import io
import json
import re
import wave
import math
import struct
from typing import List, Dict, Any, Optional, Tuple

import requests
import streamlit as st
from google import genai


# =========================================================
# 基本設定
# =========================================================

st.set_page_config(
    page_title="5分で名作",
    page_icon="📖",
    layout="wide",
)

TTS_MODEL = "gemini-3.1-flash-tts-preview"
SCRIPT_MODEL = "gemini-3.6-flash"

SAMPLE_RATE = 24000
CHANNELS = 1
SAMPLE_WIDTH = 2

DEFAULT_CHUNK_CHARS = 450
SPEED = 1.5

# YouTube / DOVA は「素材そのものを内蔵」せず、
# 脚本から検索キーワードと公式サイトへの導線だけを提示する。
RESOURCE_SITES = {
    "YouTube Audio Library": "https://www.youtube.com/audiolibrary",
    "DOVA-SYNDROME": "https://dova-s.jp/",
}


# =========================================================
# Gemini API
# =========================================================

def get_gemini_client(api_key: str):
    if not api_key or not api_key.strip():
        raise RuntimeError("Gemini API Keyが入力されていません。")
    return genai.Client(api_key=api_key.strip())


# =========================================================
# WAV / PCM
# =========================================================

def pcm_to_wav(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    if not pcm:
        raise RuntimeError("PCM音声が空です。")

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def duration(wav_bytes: bytes) -> float:
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        return w.getnframes() / w.getframerate()


def wav_params(wav_bytes: bytes):
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        return w.getparams()


def make_silence(seconds: float, sample_rate: int = SAMPLE_RATE) -> bytes:
    frames = max(0, int(seconds * sample_rate))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * frames)
    return buf.getvalue()


def normalize_wav(wav_bytes: bytes) -> bytes:
    """Aivis/Geminiで違うサンプルレート等が返っても24kHz mono 16bitへ寄せる。"""
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        channels = w.getnchannels()
        width = w.getsampwidth()
        rate = w.getframerate()
        frames = w.getnframes()
        raw = w.readframes(frames)

    if channels == 1 and width == 2 and rate == SAMPLE_RATE:
        return wav_bytes

    if width != 2:
        raise RuntimeError("16bit PCM以外のWAVは現在のミキサーでは扱えません。")

    samples = struct.unpack("<" + "h" * (len(raw) // 2), raw)

    if channels > 1:
        mono = []
        for i in range(0, len(samples), channels):
            group = samples[i:i + channels]
            mono.append(int(sum(group) / len(group)))
        samples = mono

    if rate != SAMPLE_RATE:
        old_frames = len(samples)
        new_frames = max(1, int(old_frames * SAMPLE_RATE / rate))
        resampled = []
        for i in range(new_frames):
            pos = i * rate / SAMPLE_RATE
            left = int(pos)
            right = min(left + 1, old_frames - 1)
            frac = pos - left
            value = int(samples[left] * (1 - frac) + samples[right] * frac)
            resampled.append(value)
        samples = resampled

    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(struct.pack("<" + "h" * len(samples), *samples))
    return out.getvalue()


def speed_up(wav_bytes: bytes, factor: float = 1.5) -> bytes:
    """簡易タイムストレッチ。音程は変えずに、サンプルを間引く方式。"""
    wav_bytes = normalize_wav(wav_bytes)
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        params = w.getparams()
        raw = w.readframes(w.getnframes())

    samples = array_from_pcm16(raw)
    frames = len(samples)
    out_samples = []

    pos = 0.0
    while int(pos) < frames:
        out_samples.append(samples[int(pos)])
        pos += factor

    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(params.framerate)
        w.writeframes(pcm16_from_array(out_samples))
    return out.getvalue()


def array_from_pcm16(raw: bytes) -> List[int]:
    if not raw:
        return []
    return list(struct.unpack("<" + "h" * (len(raw) // 2), raw))


def pcm16_from_array(values: List[int]) -> bytes:
    clipped = [max(-32768, min(32767, int(v))) for v in values]
    if not clipped:
        return b""
    return struct.pack("<" + "h" * len(clipped), *clipped)


# =========================================================
# タイムライン・ミキサー
# =========================================================

def mix_wavs_at_timeline(
    clips: List[Dict[str, Any]],
    tail_seconds: float = 0.25,
) -> bytes:
    """
    clips:
      {"audio": wav_bytes, "start": 秒, "gain": 0.0～1.5}
    同時発話は同じstartに置けば重ねてミックスできる。
    """
    if not clips:
        raise RuntimeError("ミックスする音声がありません。")

    normalized = []
    end_time = 0.0

    for clip in clips:
        audio = normalize_wav(clip["audio"])
        start = max(0.0, float(clip.get("start", 0.0)))
        gain = float(clip.get("gain", 1.0))

        with wave.open(io.BytesIO(audio), "rb") as w:
            raw = w.readframes(w.getnframes())
            frames = w.getnframes()

        normalized.append((start, gain, array_from_pcm16(raw), frames))
        end_time = max(end_time, start + frames / SAMPLE_RATE)

    total_frames = int((end_time + tail_seconds) * SAMPLE_RATE)
    mix = [0] * total_frames

    for start, gain, samples, frames in normalized:
        offset = int(start * SAMPLE_RATE)
        for i, sample in enumerate(samples):
            j = offset + i
            if j >= total_frames:
                break
            mix[j] += int(sample * gain)

    raw = pcm16_from_array(mix)
    return pcm_to_wav(raw)


# =========================================================
# 文章処理
# =========================================================

def normalize_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, max_chars: int = DEFAULT_CHUNK_CHARS) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks = []
    remaining = text

    while len(remaining) > max_chars:
        area = remaining[:max_chars]
        positions = [i + 1 for i, c in enumerate(area) if c in "。！？!?」』"]
        if positions:
            cut = max(positions)
        else:
            positions = [i + 1 for i, c in enumerate(area) if c in "、,\n"]
            cut = max(positions) if positions else max_chars

        part = remaining[:cut].strip()
        if part:
            chunks.append(part)
        remaining = remaining[cut:].strip()

    if remaining:
        chunks.append(remaining)
    return chunks


# =========================================================
# Gemini TTS
# =========================================================

def raw_tts(client, text: str, voice: str) -> bytes:
    """
    Gemini TTSの生PCMをWAVに変換。
    以前のNameErrorの原因だったraw_ttsを明示的に定義している。
    """
    text = normalize_text(text)
    if not text:
        raise RuntimeError("TTSに渡す文章が空です。")

    prompt = (
        "Read the following Japanese text naturally and expressively. "
        "Speak only the Japanese text. "
        "Do not explain, translate, summarize, or add any words.\n\n"
        "Japanese text:\n" + text
    )

    response = client.interactions.create(
        model=TTS_MODEL,
        input=prompt,
        response_format={"type": "audio"},
        generation_config={
            "speech_config": [{"voice": voice}]
        },
    )

    if not hasattr(response, "output_audio") or response.output_audio is None:
        raise RuntimeError("Geminiから音声データが返されませんでした。")

    data = response.output_audio.data
    if data is None:
        raise RuntimeError("Geminiから空の音声データが返されました。")

    if isinstance(data, str):
        import base64
        pcm = base64.b64decode(data)
    else:
        pcm = bytes(data)

    return pcm_to_wav(pcm)


def gemini_tts(text: str, voice: str, client) -> bytes:
    parts = split_text(text)
    if not parts:
        raise RuntimeError("音声化する文章がありません。")

    wavs = []
    for part in parts:
        wavs.append(raw_tts(client, part, voice))

    # 同じ話者内の分割は連続再生する。
    clips = []
    cursor = 0.0
    for wav in wavs:
        clips.append({"audio": wav, "start": cursor})
        cursor += duration(wav)

    return mix_wavs_at_timeline(clips, tail_seconds=0.05)


# =========================================================
# AivisSpeech Engine
# =========================================================

def get_aivis_speakers(engine_url: str) -> List[Dict[str, Any]]:
    url = engine_url.rstrip("/") + "/speakers"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()


def aivis_voice_options(speakers: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
    options = []
    for speaker in speakers:
        name = speaker.get("name", "Unknown")
        for style in speaker.get("styles", []):
            style_name = style.get("name", "ノーマル")
            sid = int(style.get("id"))
            options.append((f"{name} / {style_name}", sid))
    return options


def aivis_tts(text: str, speaker_id: int, engine_url: str) -> bytes:
    base = engine_url.rstrip("/")
    text = normalize_text(text)
    if not text:
        raise RuntimeError("TTSに渡す文章が空です。")

    # VOICEVOX互換API。
    q = requests.post(
        base + "/audio_query",
        params={"text": text, "speaker": speaker_id},
        timeout=30,
    )
    q.raise_for_status()
    query = q.json()

    s = requests.post(
        base + "/synthesis",
        params={"speaker": speaker_id},
        json=query,
        timeout=120,
    )
    s.raise_for_status()

    if not s.content:
        raise RuntimeError("AivisSpeech Engineから音声が返されませんでした。")

    return normalize_wav(s.content)


# =========================================================
# 脚本解析
# =========================================================

VOICE_CAST_RE = re.compile(r"^\s*\[([^\]]+)\]\s*$")
WAIT_RE = re.compile(r"\[WAIT\s*([0-9.]+)\]", re.I)
SE_RE = re.compile(r"^\s*\[SE(?:想定)?\s*[:：]?\s*(.*?)\]\s*$", re.I)
BGM_RE = re.compile(r"^\s*\[BGM\s*[:：]?\s*(.*?)\]\s*$", re.I)


def parse_script_text(text: str) -> Dict[str, Any]:
    """
    最初に作った脚本形式をそのまま受け付ける。
    [NARRATOR | 静かに]
    [MEROS | 怒り]
    [MEROS & SELINUNTIUS | 同時に、涙声]
    [SE想定：強い平手打ち]
    [WAIT 1.0]
    """
    lines = normalize_text(text).splitlines()

    segments = []
    se_instructions = []
    bgm_instructions = []
    image_instructions = []

    current_speaker = None
    current_note = ""

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("##") or line.startswith("#"):
            continue

        m = WAIT_RE.match(line)
        if m:
            segments.append({
                "type": "wait",
                "duration": float(m.group(1)),
            })
            continue

        m = SE_RE.match(line)
        if m:
            sound = m.group(1).strip()
            se_instructions.append({
                "time_hint": "",
                "sound": sound,
                "purpose": "脚本内で指定されたSE",
            })
            continue

        m = BGM_RE.match(line)
        if m:
            mood = m.group(1).strip()
            bgm_instructions.append({
                "time_hint": "",
                "mood": mood,
                "purpose": "脚本内で指定されたBGM",
            })
            continue

        m = VOICE_CAST_RE.match(line)
        if m:
            header = m.group(1).strip()
            parts = [p.strip() for p in header.split("|", 1)]
            current_speaker = parts[0]
            current_note = parts[1] if len(parts) > 1 else ""
            continue

        # 明示的な話者ヘッダがない通常文は、直前の話者へ。
        if current_speaker:
            # 引用符を含む本文もそのまま保持。
            segments.append({
                "type": "dialogue",
                "speaker": current_speaker,
                "note": current_note,
                "text": line,
            })

    return {
        "title": "アップロード脚本",
        "segments": segments,
        "image_instructions": image_instructions,
        "se_instructions": se_instructions,
        "bgm_instructions": bgm_instructions,
    }


# =========================================================
# Geminiによる脚本生成
# =========================================================

def make_script(client, source: str, minutes: int, roles: List[str]) -> Dict[str, Any]:
    source = normalize_text(source)
    role_text = "、".join(roles)

    prompt = f"""
あなたはYouTubeチャンネル「5分で名作」の脚本家です。

以下の原作を、視聴者が最後まで見たくなる
「短く、速く、感情の変化が明確な朗読ドラマ」
として再構成してください。

目標尺: {minutes}分

重要:
- 解説動画ではなく物語として構成する
- 原作の重要な出来事と結末を残す
- 冒頭から物語を始める
- 冗長な説明を削る
- 感情が動く場面を残す
- セリフとナレーションを適切に混ぜる
- 同時発話が効果的な場面では speaker を「A & B」の形式にできる
- 間を入れたい場合は type="wait" のsegmentを使い、durationを秒で指定する
- SE/BGMは実際の音声本文に入れず、別指示として出す
- YouTube向けに聞きやすい自然な日本語にする

使用可能な話者:
{role_text}

segments のspeakerは上記の話者名、または複数話者を「話者A & 話者B」の形で指定してください。

JSONだけを返してください。

{{
  "title": "作品タイトル",
  "segments": [
    {{
      "type": "dialogue",
      "speaker": "ナレーター",
      "text": "..."
    }},
    {{
      "type": "wait",
      "duration": 1.0
    }}
  ],
  "image_instructions": [
    {{
      "time_hint": "0:00",
      "visual": "画面に欲しいイラスト"
    }}
  ],
  "se_instructions": [
    {{
      "time_hint": "0:00",
      "sound": "欲しい効果音",
      "purpose": "何のためのSEか"
    }}
  ],
  "bgm_instructions": [
    {{
      "time_hint": "0:00",
      "mood": "BGMの雰囲気",
      "purpose": "何のためのBGMか"
    }}
  ]
}}

原作:
{source}
"""

    response = client.interactions.create(
        model=SCRIPT_MODEL,
        input=prompt,
    )

    output = getattr(response, "output_text", "")
    match = re.search(r"\{.*\}", output, re.DOTALL)
    if not match:
        raise RuntimeError("Geminiから脚本JSONを取得できませんでした。")

    try:
        return json.loads(match.group())
    except json.JSONDecodeError as e:
        raise RuntimeError("脚本JSONの解析に失敗しました。\n" + str(e))


# =========================================================
# タイムライン構築
# =========================================================

def split_speakers(speaker: str) -> List[str]:
    return [s.strip() for s in re.split(r"\s*&\s*|、|,|＆", speaker) if s.strip()]


def build_timeline_from_segments(
    segments: List[Dict[str, Any]],
    cast: Dict[str, Any],
    engine: str,
    gemini_client=None,
    aivis_engine_url: str = "",
    progress_callback=None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    重要:
    「順番に結合」ではなく、開始時刻を持つtimelineを作る。
    同時発話は同じstartに配置する。
    """
    timeline = []
    subtitle_segments = []
    cursor = 0.0

    dialogue_count = sum(
        1 for s in segments if s.get("type", "dialogue") == "dialogue"
    )
    done = 0

    for seg in segments:
        seg_type = seg.get("type", "dialogue")

        if seg_type == "wait":
            cursor += max(0.0, float(seg.get("duration", 0)))
            continue

        speaker_raw = str(seg.get("speaker", "ナレーター")).strip()
        text = normalize_text(str(seg.get("text", "")))

        if not text:
            continue

        speakers = split_speakers(speaker_raw)
        if not speakers:
            speakers = ["ナレーター"]

        # 話者ごとに別音声を作り、同じ開始時刻へ置く。
        generated = []

        for speaker in speakers:
            actual_speaker = speaker if speaker in cast else "ナレーター"
            voice = cast.get(actual_speaker)

            if engine == "Gemini TTS":
                if not gemini_client:
                    raise RuntimeError("Gemini TTSにはGemini API Keyが必要です。")
                if not isinstance(voice, str):
                    voice = str(voice)
                audio = gemini_tts(text, voice, gemini_client)

            elif engine == "AivisSpeech":
                if not aivis_engine_url:
                    raise RuntimeError(
                        "AivisSpeech Engine URLが設定されていません。"
                        "ローカルでAivisSpeech Engineを起動してください。"
                    )
                audio = aivis_tts(text, int(voice), aivis_engine_url)

            else:
                raise RuntimeError("音声エンジンが不正です。")

            generated.append({
                "speaker": actual_speaker,
                "text": text,
                "audio": audio,
                "start": cursor,
                "duration": duration(audio),
            })

        max_dur = max(g["duration"] for g in generated)

        for g in generated:
            timeline.append(g)

        subtitle_segments.append({
            "speaker": speaker_raw,
            "text": text,
            "start": cursor,
            "duration": max_dur,
        })

        cursor += max_dur
        done += 1

        if progress_callback:
            progress_callback(done / max(1, dialogue_count))

    return timeline, subtitle_segments


# =========================================================
# SRT
# =========================================================

def srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{ms:03}"


def make_srt(segments: List[Dict[str, Any]], speed_factor: float = 1.0) -> str:
    out = []

    for i, seg in enumerate(segments, 1):
        start = seg["start"] / speed_factor
        end = (seg["start"] + seg["duration"]) / speed_factor
        out.append(
            f"{i}\n"
            f"{srt_time(start)} --> {srt_time(end)}\n"
            f"{seg['speaker']}: {seg['text']}\n"
        )

    return "\n".join(out)


# =========================================================
# SE/BGM候補
# =========================================================

def make_search_url(site: str, keyword: str) -> str:
    from urllib.parse import quote_plus
    if site == "YouTube Audio Library":
        return "https://www.youtube.com/audiolibrary"
    return "https://www.google.com/search?q=" + quote_plus(
        f"site:dova-s.jp {keyword}"
    )


def suggest_se_bgm(result: Dict[str, Any]) -> Dict[str, Any]:
    se = result.get("se_instructions", [])
    bgm = result.get("bgm_instructions", [])

    for item in se:
        keyword = item.get("sound", "")
        item["recommended_sites"] = [
            {
                "site": "YouTube Audio Library",
                "url": make_search_url("YouTube Audio Library", keyword),
            },
            {
                "site": "DOVA-SYNDROME",
                "url": make_search_url("DOVA-SYNDROME", keyword),
            },
        ]

    for item in bgm:
        keyword = item.get("mood", "")
        item["recommended_sites"] = [
            {
                "site": "YouTube Audio Library",
                "url": make_search_url("YouTube Audio Library", keyword),
            },
            {
                "site": "DOVA-SYNDROME",
                "url": make_search_url("DOVA-SYNDROME", keyword),
            },
        ]

    return result


# =========================================================
# クレジット
# =========================================================

def make_credits(
    engine: str,
    cast: Dict[str, Any],
    aivis_speakers: Optional[Dict[int, str]] = None,
) -> str:
    lines = [
        "【5分で名作 - 音声クレジット】",
        "",
        f"音声エンジン: {engine}",
    ]

    if engine == "Gemini TTS":
        lines.append("Gemini TTSを使用")
    else:
        lines.append("AivisSpeech Engineを使用")
        lines.append("※使用した音声モデルの個別ライセンスを確認してください。")

    lines.append("")
    lines.append("【CAST】")

    for role, voice in cast.items():
        if aivis_speakers and isinstance(voice, int):
            lines.append(f"{role}: {aivis_speakers.get(voice, voice)}")
        else:
            lines.append(f"{role}: {voice}")

    lines += [
        "",
        "【素材】",
        "SE/BGMは各素材サイトの利用規約・個別ライセンスを確認してください。",
        "YouTube Audio Library: https://www.youtube.com/audiolibrary",
        "DOVA-SYNDROME: https://dova-s.jp/",
    ]

    return "\n".join(lines)


# =========================================================
# Session State
# =========================================================

if "cast" not in st.session_state:
    st.session_state.cast = {
        "ナレーター": "Rasalgethi",
        "メロス": "Iapetus",
        "王": "Algieba",
        "セリヌンティウス": "Alnilam",
        "妹": "Aoede",
        "フィロストラトス": "Schedar",
    }

if "aivis_cast" not in st.session_state:
    st.session_state.aivis_cast = {}

if "script_result" not in st.session_state:
    st.session_state.script_result = None

if "timeline" not in st.session_state:
    st.session_state.timeline = None

if "subtitle_segments" not in st.session_state:
    st.session_state.subtitle_segments = None

if "final_audio" not in st.session_state:
    st.session_state.final_audio = None

if "fast_audio" not in st.session_state:
    st.session_state.fast_audio = None


# =========================================================
# UI
# =========================================================

st.title("📖 5分で名作・自動音声制作")
st.caption(
    "原作 → 脚本 / 脚本アップロード → キャスト → タイムライン → "
    "SE/BGM候補 → WAV → SRT"
)

with st.sidebar:
    st.header("⚙️ 制作設定")

    engine = st.radio(
        "音声エンジン",
        ["AivisSpeech", "Gemini TTS"],
        help=(
            "AivisSpeechはローカルEngineを起動して使う方式。"
            "Gemini TTSはAPIを使用します。"
        ),
    )

    gemini_key = st.text_input(
        "Gemini API Key",
        type="password",
        help="脚本生成をGeminiで行う場合、またはGemini TTSを使う場合に必要です。",
    )

    aivis_url = st.text_input(
        "AivisSpeech Engine URL",
        value="http://127.0.0.1:10101",
        help="AivisSpeech EngineのURL。ローカル実行時は通常 http://127.0.0.1:10101",
    )

    if engine == "AivisSpeech":
        st.info(
            "AivisSpeechを選んだ場合、Streamlit CloudからあなたのPCのlocalhostへ"
            "接続することはできません。AivisSpeech Engineとこのアプリを同じPCで動かしてください。"
        )

    st.divider()
    st.subheader("🎙 キャスト")

    if engine == "Gemini TTS":
        voices = [
            "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda",
            "Orus", "Aoede", "Callirrhoe", "Autonoe", "Enceladus",
            "Iapetus", "Umbriel", "Algieba", "Despina", "Erinome",
            "Algenib", "Rasalgethi", "Laomedeia", "Achernar",
            "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird",
            "Zubenelgenubi", "Vindemiatrix", "Sadachbia",
            "Sadaltager", "Sulafat",
        ]

        for role in list(st.session_state.cast):
            current = st.session_state.cast[role]
            if current not in voices:
                current = "Kore"

            st.session_state.cast[role] = st.selectbox(
                role,
                voices,
                index=voices.index(current),
                key=f"gemini_voice_{role}",
            )

    else:
        try:
            speakers = get_aivis_speakers(aivis_url)
            options = aivis_voice_options(speakers)

            if not options:
                st.warning("AivisSpeech Engineから音声モデルを取得できませんでした。")
            else:
                label_to_id = dict(options)
                labels = list(label_to_id.keys())

                for role in list(st.session_state.cast):
                    old_id = st.session_state.aivis_cast.get(role)
                    default_index = 0
                    if old_id is not None:
                        for i, label in enumerate(labels):
                            if label_to_id[label] == old_id:
                                default_index = i
                                break

                    selected = st.selectbox(
                        role,
                        labels,
                        index=default_index,
                        key=f"aivis_voice_{role}",
                    )
                    st.session_state.aivis_cast[role] = label_to_id[selected]

                # Aivis用の実キャスト辞書へ。
                st.session_state.aivis_voice_labels = label_to_id

        except Exception as e:
            st.warning(
                "AivisSpeech Engineに接続できません。\n"
                "Engineを起動してからページを再読み込みしてください。"
            )
            st.caption(str(e))

    if st.button("＋キャラクター追加"):
        n = 1
        while f"キャラクター{n}" in st.session_state.cast:
            n += 1
        st.session_state.cast[f"キャラクター{n}"] = (
            "Kore" if engine == "Gemini TTS" else 0
        )
        st.rerun()


# =========================================================
# 入力方式
# =========================================================

st.header("① 作品を入力")

input_mode = st.radio(
    "入力方法",
    ["原作からGeminiに脚本を作らせる", "完成済み脚本をアップロード"],
    horizontal=True,
)

source = ""

if input_mode == "原作からGeminiに脚本を作らせる":
    uploaded = st.file_uploader(
        "原作 TXT / MD",
        type=["txt", "md"],
        key="source_upload",
    )

    if uploaded:
        source = uploaded.read().decode("utf-8", "replace")
    else:
        source = st.text_area(
            "または原作本文を貼り付け",
            height=260,
            placeholder="青空文庫などの本文を貼り付けます。",
        )

    minutes = st.radio(
        "動画の長さ",
        [1, 5, 10],
        index=1,
        horizontal=True,
    )

else:
    script_file = st.file_uploader(
        "完成済み脚本 TXT / MD",
        type=["txt", "md"],
        key="script_upload",
    )

    if script_file:
        source = script_file.read().decode("utf-8", "replace")
        st.success("脚本を読み込みました。")
        with st.expander("読み込んだ脚本"):
            st.text(source)

    minutes = 5


# =========================================================
# 脚本作成
# =========================================================

st.header("② 脚本")

col1, col2 = st.columns(2)

with col1:
    make_script_button = st.button(
        "📝 脚本を作成 / 読み込む",
        type="primary",
        disabled=not bool(source.strip()),
    )

with col2:
    st.caption(
        "完成脚本アップロードの場合はGeminiを使わず、その脚本をそのまま解析します。"
    )

if make_script_button:
    try:
        if input_mode == "原作からGeminiに脚本を作らせる":
            client = get_gemini_client(gemini_key)
            with st.spinner("Geminiが脚本を作成しています…"):
                result = make_script(
                    client,
                    source,
                    minutes,
                    list(st.session_state.cast.keys()),
                )
        else:
            with st.spinner("脚本を解析しています…"):
                result = parse_script_text(source)

        result = suggest_se_bgm(result)
        st.session_state.script_result = result
        st.session_state.timeline = None
        st.session_state.subtitle_segments = None
        st.session_state.final_audio = None
        st.session_state.fast_audio = None

        st.success(
            f"脚本を準備しました：{len(result.get('segments', []))}項目"
        )

    except Exception as e:
        st.error("脚本処理に失敗しました。")
        st.code(str(e))


# =========================================================
# 脚本表示
# =========================================================

result = st.session_state.script_result

if result:
    st.subheader("📜 現在の脚本")

    with st.expander("脚本を確認", expanded=True):
        for i, seg in enumerate(result.get("segments", []), 1):
            if seg.get("type") == "wait":
                st.markdown(
                    f"**{i}. ⏸ WAIT {seg.get('duration', 0)}秒**"
                )
            else:
                st.markdown(
                    f"**{i}. {seg.get('speaker', 'ナレーター')}**  "
                    f"{seg.get('text', '')}"
                )


# =========================================================
# 音声生成
# =========================================================

if result:
    st.header("③ 音声生成")

    if engine == "AivisSpeech":
        cast_for_engine = st.session_state.aivis_cast.copy()
    else:
        cast_for_engine = st.session_state.cast.copy()

    generate_audio = st.button(
        "🎙 音声を生成する",
        type="primary",
    )

    if generate_audio:
        try:
            client = None
            if engine == "Gemini TTS":
                client = get_gemini_client(gemini_key)

            progress = st.progress(0.0)
            status = st.empty()

            def update_progress(value):
                progress.progress(min(1.0, max(0.0, value)))
                status.write(f"音声生成：{int(value * 100)}%")

            timeline, subtitles = build_timeline_from_segments(
                result.get("segments", []),
                cast_for_engine,
                engine,
                gemini_client=client,
                aivis_engine_url=aivis_url,
                progress_callback=update_progress,
            )

            if not timeline:
                raise RuntimeError("音声化できるセリフがありません。")

            status.write("音声をタイムライン上でミックスしています…")

            final_audio = mix_wavs_at_timeline(timeline)
            fast_audio = speed_up(final_audio, SPEED)

            st.session_state.timeline = timeline
            st.session_state.subtitle_segments = subtitles
            st.session_state.final_audio = final_audio
            st.session_state.fast_audio = fast_audio

            progress.progress(1.0)
            status.write("完成しました。")
            st.success("🎉 音声生成が完了しました。")

        except Exception as e:
            st.error("音声生成に失敗しました。")
            st.code(str(e))


# =========================================================
# 結果
# =========================================================

if st.session_state.final_audio:
    st.header("④ 完成音声")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("通常速度")
        st.audio(st.session_state.final_audio, format="audio/wav")
        st.download_button(
            "⬇ 通常速度 WAV",
            st.session_state.final_audio,
            "audio_original.wav",
            "audio/wav",
        )

    with c2:
        st.subheader("1.5倍速")
        st.audio(st.session_state.fast_audio, format="audio/wav")
        st.download_button(
            "⬇ 1.5倍速 WAV",
            st.session_state.fast_audio,
            "audio_1.5x.wav",
            "audio/wav",
        )

    st.header("⑤ SRT字幕")

    srt = make_srt(
        st.session_state.subtitle_segments,
        SPEED,
    )

    st.download_button(
        "⬇ SRT字幕",
        srt,
        "subtitles.srt",
        "application/x-subrip",
    )

    with st.expander("SRTプレビュー"):
        st.code(srt, language="text")

    st.header("⑥ SE / BGM候補")

    st.info(
        "素材そのものをアプリへ内蔵せず、作品ごとに検索候補を提示します。"
        "利用前に各素材の個別ライセンスを確認してください。"
    )

    se_items = result.get("se_instructions", [])
    bgm_items = result.get("bgm_instructions", [])

    if se_items:
        st.subheader("🔊 SE")
        for item in se_items:
            st.markdown(
                f"**{item.get('time_hint', '')} "
                f"{item.get('sound', '')}** — {item.get('purpose', '')}"
            )
            for site in item.get("recommended_sites", []):
                st.markdown(
                    f"- [{site['site']}]({site['url']})"
                )
    else:
        st.caption("脚本からSE指定はありません。")

    if bgm_items:
        st.subheader("🎵 BGM")
        for item in bgm_items:
            st.markdown(
                f"**{item.get('time_hint', '')} "
                f"{item.get('mood', '')}** — {item.get('purpose', '')}"
            )
            for site in item.get("recommended_sites", []):
                st.markdown(
                    f"- [{site['site']}]({site['url']})"
                )
    else:
        st.caption("脚本からBGM指定はありません。")

    st.header("⑦ クレジット")

    labels = {}
    if engine == "AivisSpeech":
        labels = getattr(st.session_state, "aivis_voice_labels", {})

    credits = make_credits(
        engine,
        cast_for_engine,
        labels,
    )

    st.text(credits)

    st.download_button(
        "⬇ クレジットTXT",
        credits,
        "voice_credits.txt",
        "text/plain",
    )

    st.header("⑧ タイムライン")

    for i, clip in enumerate(st.session_state.timeline, 1):
        st.write(
            f"{i:02d}  "
            f"{clip['start']:.2f}s  "
            f"{clip['speaker']}  "
            f"{clip['duration']:.2f}s"
        )

    st.download_button(
        "⬇ 制作情報 JSON",
        json.dumps(result, ensure_ascii=False, indent=2),
        "production_notes.json",
        "application/json",
    )

st.divider()
st.caption(
    "※ YouTube収益化を目的とする場合も、原作・音声モデル・SE・BGM・画像の"
    "各ライセンスと、YouTubeの最新の収益化ポリシーを作品ごとに確認してください。"
)
