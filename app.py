import io
import json
import re
import time
import wave
import zipfile
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

import requests
import streamlit as st

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

# =========================================================
# 5分で名作 — local AivisSpeech / Gemini TTS edition
# =========================================================
# Design goals:
# - Keep original -> Gemini script generation.
# - Accept a completed script in the user's rich format.
# - Prefer FREE local AivisSpeech Engine for TTS.
# - Keep Gemini TTS as an optional cloud fallback.
# - Build an explicit timeline instead of simply concatenating files.
# - Support simultaneous dialogue, waits, SE/BGM/image cues.
# - Suggest copyright-safer SE/BGM sources without bundling copyrighted assets.
# - Export WAV, 1.5x WAV, SRT, timeline JSON and credits.

st.set_page_config(page_title="5分で名作", page_icon="📖", layout="wide")

AIVIS_DEFAULT_URL = "http://127.0.0.1:10101"
GEMINI_TTS_MODEL = "gemini-3.1-flash-tts-preview"
GEMINI_SCRIPT_MODEL = "gemini-3.6-flash"
SPEED = 1.5
SAMPLE_RATE = 24000

# Gemini voices retained as optional voices.
GEMINI_VOICES = [
    "Rasalgethi", "Algieba", "Alnilam", "Aoede", "Iapetus", "Schedar",
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus",
]

# Fallback mapping used only when a completed script does not provide explicit model/style IDs.
DEFAULT_ROLE_MAP = {
    "NARRATOR": {"label": "ナレーター", "gemini": "Rasalgethi"},
    "MEROS": {"label": "メロス", "gemini": "Iapetus"},
    "DIONYS": {"label": "ディオニス／王", "gemini": "Algieba"},
    "SELINUNTIUS": {"label": "セリヌンティウス", "gemini": "Alnilam"},
    "FILOSTRATUS": {"label": "フィロストラトス", "gemini": "Schedar"},
    "SISTER": {"label": "妹", "gemini": "Aoede"},
}

# =========================================================
# Audio utilities
# =========================================================

def pcm_to_wav(pcm: bytes, sample_rate: int = SAMPLE_RATE, channels: int = 1, sample_width: int = 2) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return out.getvalue()


def wav_info(data: bytes) -> Tuple[int, int, int, int]:
    with wave.open(io.BytesIO(data), "rb") as wf:
        return wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()


def wav_duration(data: bytes) -> float:
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception:
        return 0.0


def ensure_wav(data: bytes) -> bytes:
    if data[:4] == b"RIFF":
        return data
    return pcm_to_wav(data)


def silence_wav(seconds: float, sample_rate: int = SAMPLE_RATE) -> bytes:
    frames = max(0, int(seconds * sample_rate))
    return pcm_to_wav(b"\x00\x00" * frames, sample_rate=sample_rate)


def mix_wavs(base: bytes, overlay: bytes, start_seconds: float = 0.0) -> bytes:
    """Mix two mono 16-bit WAVs, placing overlay at start_seconds."""
    import array
    with wave.open(io.BytesIO(base), "rb") as wb:
        sr_b = wb.getframerate(); ch_b = wb.getnchannels(); sw_b = wb.getsampwidth(); raw_b = wb.readframes(wb.getnframes())
    with wave.open(io.BytesIO(overlay), "rb") as wo:
        sr_o = wo.getframerate(); ch_o = wo.getnchannels(); sw_o = wo.getsampwidth(); raw_o = wo.readframes(wo.getnframes())
    if not (sr_b == sr_o == SAMPLE_RATE and ch_b == ch_o == 1 and sw_b == sw_o == 2):
        raise ValueError("WAV format mismatch. 24kHz/mono/16bit WAV is required.")
    a = array.array("h"); a.frombytes(raw_b)
    b = array.array("h"); b.frombytes(raw_o)
    offset = int(start_seconds * SAMPLE_RATE)
    need = max(len(a), offset + len(b))
    if len(a) < need:
        a.extend([0] * (need - len(a)))
    for i, sample in enumerate(b):
        j = offset + i
        v = a[j] + sample
        a[j] = max(-32768, min(32767, v))
    return pcm_to_wav(a.tobytes())


def concat_wavs(parts: List[bytes]) -> bytes:
    parts = [ensure_wav(p) for p in parts if p]
    if not parts:
        return silence_wav(0.05)
    with wave.open(io.BytesIO(parts[0]), "rb") as w0:
        params = w0.getparams()
        frames = [w0.readframes(w0.getnframes())]
    for data in parts[1:]:
        with wave.open(io.BytesIO(data), "rb") as wf:
            p = wf.getparams()
            if (p.nchannels, p.sampwidth, p.framerate) != (params.nchannels, params.sampwidth, params.framerate):
                raise ValueError("All WAV files must have the same format.")
            frames.append(wf.readframes(wf.getnframes()))
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setparams(params)
        for fr in frames:
            wf.writeframes(fr)
    return out.getvalue()


def speed_up_wav(data: bytes, factor: float = SPEED) -> bytes:
    """Simple time compression by frame decimation/interpolation-free resampling.
    This changes pitch slightly; it is deliberately conservative and dependency-free.
    """
    import array
    with wave.open(io.BytesIO(data), "rb") as wf:
        sr = wf.getframerate(); ch = wf.getnchannels(); sw = wf.getsampwidth(); raw = wf.readframes(wf.getnframes())
    if ch != 1 or sw != 2:
        return data
    arr = array.array("h"); arr.frombytes(raw)
    step = max(1, factor)
    out = array.array("h")
    idx = 0.0
    while int(idx) < len(arr):
        out.append(arr[int(idx)])
        idx += step
    return pcm_to_wav(out.tobytes(), sr, 1, 2)

# =========================================================
# Script parsing
# =========================================================

@dataclass
class Segment:
    index: int
    speaker: str
    text: str
    direction: str = ""
    simultaneous_group: Optional[str] = None
    start: Optional[float] = None
    kind: str = "speech"  # speech / wait / cue
    cue_type: str = ""
    cue_text: str = ""


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^['\"「『]|['\"」』]$", "", text).strip()
    return text


def parse_script(text: str) -> Tuple[Dict[str, str], List[Segment], List[Dict]]:
    """Parse the user's requested format.

    Supports:
      [SPEAKER | direction]
      [SPEAKER]
      [SPEAKER & SPEAKER | direction]
      [SE想定：...]
      [BGM：...]
      [画像：...]
      [WAIT 1.5]
      [END]
    Also accepts the VOICE CAST section as metadata.
    """
    cast = {}
    segments: List[Segment] = []
    cues: List[Dict] = []
    current_speaker = None
    current_direction = ""
    buffer: List[str] = []
    idx = 1

    def flush():
        nonlocal idx, buffer
        if not buffer or not current_speaker:
            buffer = []
            return
        txt = clean_text("\n".join(buffer))
        if txt:
            segments.append(Segment(idx, current_speaker, txt, current_direction))
            idx += 1
        buffer = []

    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    in_script = False
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.upper().startswith("# SCRIPT"):
            in_script = True
            continue
        if line.upper().startswith("# VOICE CAST"):
            in_script = False
            continue
        if not in_script and line.startswith("###"):
            # Store the character description as cast metadata until SCRIPT begins.
            name = line.lstrip("# ").strip().upper()
            cast.setdefault(name, "")
            continue
        if not in_script and cast:
            # Description lines are attached to the most recent ### role.
            if cast:
                last = list(cast.keys())[-1]
                if line.startswith("#"):
                    continue
                cast[last] = (cast[last] + " " + line).strip()
            continue
        if not in_script:
            # If there is no explicit # SCRIPT, treat first dialogue marker as script start.
            if not line.startswith("["):
                continue
            in_script = True

        if line == "[END]":
            flush(); break
        m_wait = re.match(r"^\[\s*WAIT\s+([0-9]+(?:\.[0-9]+)?)\s*\]$", line, re.I)
        if m_wait:
            flush()
            segments.append(Segment(idx, "", "", kind="wait", start=float(m_wait.group(1))))
            idx += 1
            continue
        m_cue = re.match(r"^\[\s*(SE|SE想定|BGM|画像|IMAGE)\s*[:：]\s*(.*?)\s*\]$", line, re.I)
        if m_cue:
            flush()
            typ = m_cue.group(1).upper()
            typ = "SE" if typ.startswith("SE") else ("BGM" if typ == "BGM" else "IMAGE")
            cues.append({"index": idx, "type": typ, "text": m_cue.group(2)})
            segments.append(Segment(idx, "", "", kind="cue", cue_type=typ, cue_text=m_cue.group(2)))
            idx += 1
            continue
        m = re.match(r"^\[\s*(.*?)\s*(?:\|\s*(.*?))?\s*\]$", line)
        if m:
            flush()
            current_speaker = m.group(1).strip().upper()
            current_direction = (m.group(2) or "").strip()
            continue
        if line.startswith("#"):
            continue
        if current_speaker:
            buffer.append(line)
    flush()

    # If no explicit script markers were found, support simple "NAME: text" lines.
    if not segments:
        for raw in lines:
            m = re.match(r"^\s*([^:：]{1,30})[:：]\s*(.+)$", raw.strip())
            if m:
                segments.append(Segment(len(segments)+1, m.group(1).strip().upper(), clean_text(m.group(2))))

    # Add default cast descriptions for roles not explicitly described.
    for role, meta in DEFAULT_ROLE_MAP.items():
        cast.setdefault(role, "")
    return cast, segments, cues


def speaker_label(name: str) -> str:
    key = name.upper().strip()
    if key in DEFAULT_ROLE_MAP:
        return DEFAULT_ROLE_MAP[key]["label"]
    aliases = {
        "王": "ディオニス／王", "ディオニス": "ディオニス／王",
        "メロス": "メロス", "セリヌンティウス": "セリヌンティウス",
        "フィロストラトス": "フィロストラトス", "老人": "老人", "山賊": "山賊",
        "妹": "妹", "ナレーター": "ナレーター",
    }
    return aliases.get(key, name)


def normalize_role(name: str) -> str:
    n = name.strip().upper()
    aliases = {
        "ナレーター": "NARRATOR", "NARRATION": "NARRATOR", "ナレーション": "NARRATOR",
        "メロス": "MEROS", "王": "DIONYS", "ディオニス": "DIONYS",
        "セリヌンティウス": "SELINUNTIUS", "フィロストラトス": "FILOSTRATUS",
        "妹": "SISTER",
    }
    return aliases.get(n, n)

# =========================================================
# AivisSpeech local API
# =========================================================

def aivis_get_speakers(base_url: str) -> List[Dict]:
    r = requests.get(base_url.rstrip("/") + "/speakers", timeout=5)
    r.raise_for_status()
    return r.json()


def flatten_aivis_speakers(raw: List[Dict]) -> List[Dict]:
    out = []
    for sp in raw:
        for style in sp.get("styles", []):
            out.append({
                "speaker_id": style.get("id"),
                "speaker_name": sp.get("name", ""),
                "style_name": style.get("name", ""),
                "uuid": sp.get("speaker_uuid", ""),
            })
    return [x for x in out if x.get("speaker_id") is not None]


def aivis_synthesize(base_url: str, text: str, speaker_id: int, speed: float = 1.0, intonation: float = 1.0) -> bytes:
    base = base_url.rstrip("/")
    q = requests.post(base + "/audio_query", params={"speaker": speaker_id}, data=text.encode("utf-8"), timeout=60)
    q.raise_for_status()
    query = q.json()
    query["speedScale"] = float(speed)
    # AivisSpeech uses intonationScale for style emotion strength.
    query["intonationScale"] = max(0.0, min(2.0, float(intonation)))
    s = requests.post(base + "/synthesis", params={"speaker": speaker_id}, json=query, timeout=120)
    s.raise_for_status()
    return ensure_wav(s.content)

# =========================================================
# Gemini TTS optional
# =========================================================

def gemini_synthesize(client, text: str, voice: str, direction: str = "") -> bytes:
    prompt = text
    if direction:
        prompt = f"演技指示: {direction}\n\n{text}"
    response = client.models.generate_content(
        model=GEMINI_TTS_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                )
            ),
        ),
    )
    parts = getattr(response, "candidates", [])
    for cand in parts:
        content = getattr(cand, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []) or []:
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                return ensure_wav(inline.data)
    raise RuntimeError("Gemini TTSから音声データが返されませんでした。")

# =========================================================
# Timeline
# =========================================================

def build_timeline(segments: List[Segment], audio_by_index: Dict[int, bytes], pause: float = 0.12) -> List[Dict]:
    timeline = []
    cursor = 0.0
    simultaneous_cursor: Dict[str, float] = {}
    for seg in segments:
        if seg.kind == "wait":
            cursor += float(seg.start or 0.0)
            timeline.append({"index": seg.index, "kind": "wait", "start": cursor, "duration": float(seg.start or 0.0)})
            continue
        if seg.kind == "cue":
            timeline.append({"index": seg.index, "kind": seg.cue_type, "cue": seg.cue_text, "start": cursor})
            continue
        audio = audio_by_index.get(seg.index)
        if not audio:
            continue
        dur = wav_duration(audio)
        group = seg.simultaneous_group
        if group:
            # Same group starts at the same cursor. Group ends at max duration.
            start = cursor
            timeline.append({"index": seg.index, "kind": "speech", "speaker": seg.speaker,
                              "direction": seg.direction, "text": seg.text, "start": start, "duration": dur,
                              "simultaneous_group": group})
            simultaneous_cursor[group] = max(simultaneous_cursor.get(group, start), start + dur)
            # Do not advance cursor until the last group member; parser sets group members consecutively.
            next_seg = segments[segments.index(seg)+1] if segments.index(seg)+1 < len(segments) else None
            if not next_seg or next_seg.simultaneous_group != group:
                cursor = simultaneous_cursor[group] + pause
        else:
            start = cursor
            timeline.append({"index": seg.index, "kind": "speech", "speaker": seg.speaker,
                              "direction": seg.direction, "text": seg.text, "start": start, "duration": dur})
            cursor += dur + pause
    return timeline


def render_mix(timeline: List[Dict], audio_by_index: Dict[int, bytes]) -> bytes:
    # Render all speech into one mono track. Cues are intentionally not auto-downloaded.
    max_end = 0.05
    for item in timeline:
        if item.get("kind") == "speech":
            max_end = max(max_end, item["start"] + item["duration"])
    base = silence_wav(max_end + 0.05)
    for item in timeline:
        if item.get("kind") == "speech":
            base = mix_wavs(base, audio_by_index[item["index"]], item["start"])
    return base

# =========================================================
# SRT
# =========================================================

def srt_time(sec: float) -> str:
    ms = int(round(sec * 1000))
    h = ms // 3600000; ms %= 3600000
    m = ms // 60000; ms %= 60000
    s = ms // 1000; ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def make_srt(timeline: List[Dict], speed_factor: float = 1.0) -> str:
    out = []
    n = 1
    for item in timeline:
        if item.get("kind") != "speech":
            continue
        start = item["start"] / speed_factor
        end = (item["start"] + item["duration"]) / speed_factor
        label = speaker_label(item["speaker"])
        text = item["text"].strip()
        out.append(f"{n}\n{srt_time(start)} --> {srt_time(end)}\n[{label}] {text}\n")
        n += 1
    return "\n".join(out)

# =========================================================
# SE / BGM recommendations
# =========================================================

SE_KEYWORDS = {
    "豪雨": ["heavy rain", "rainstorm", "rain"],
    "雨": ["rain"],
    "川": ["river", "water", "stream"],
    "濁流": ["river", "water", "flood"],
    "泳": ["water splash", "splash"],
    "平手打ち": ["slap", "hit"],
    "殴": ["punch", "hit"],
    "走": ["footsteps", "running"],
    "足音": ["footsteps"],
    "歓声": ["crowd cheer", "cheering"],
    "扉": ["door", "door close"],
    "剣": ["sword", "blade"],
    "倒れ": ["fall", "body fall"],
    "風": ["wind"],
    "山賊": ["footsteps", "forest", "wind"],
}

BGM_KEYWORDS = {
    "不穏": ["dark", "tension", "suspense"],
    "緊張": ["tension", "suspense", "dramatic"],
    "怒": ["dramatic", "dark", "intense"],
    "走": ["action", "adventure", "fast"],
    "絶望": ["sad", "dramatic", "dark"],
    "希望": ["hope", "uplifting", "inspiring"],
    "感動": ["emotional", "piano", "inspiring"],
    "クライマックス": ["cinematic", "epic", "dramatic"],
    "コミカル": ["comedy", "funny", "light"],
}


def cue_suggestions(cue_type: str, cue_text: str) -> Dict:
    text = cue_text.lower()
    mapping = SE_KEYWORDS if cue_type == "SE" else BGM_KEYWORDS
    keywords = []
    for k, vals in mapping.items():
        if k.lower() in text:
            keywords.extend(vals)
    if not keywords:
        keywords = [cue_text]
    keywords = list(dict.fromkeys(keywords))[:5]
    q = "+".join(re.sub(r"[^a-zA-Z0-9 +_-]", " ", x).strip().replace(" ", "+") for x in keywords)
    return {
        "keywords": keywords,
        "youtube": "https://www.youtube.com/audiolibrary/music?nv=1",
        "dova": f"https://dova-s.jp/" if cue_type == "BGM" else "https://dova-s.jp/se/",
        "query": q,
    }

# =========================================================
# Gemini script generation
# =========================================================

def generate_script(client, source: str, minutes: int) -> str:
    prompt = f"""
あなたは日本語の朗読ドラマ脚本家です。
原作を約{minutes}分のYouTube向け朗読ドラマに再構成してください。

必ず以下のフォーマットで出力してください。

# VOICE CAST
### NARRATOR
成人。落ち着いた日本語男性ナレーター。
### MEROS
キャラクターの声質・演技方針を書く。
### DIONYS
キャラクターの声質・演技方針を書く。

# SCRIPT
[NARRATOR | 演技指示]
本文

[MEROS | 演技指示]
本文

[SE想定：必要なら効果音の内容]
[BGM：必要なら音楽の雰囲気]
[画像：必要なら画面の内容]

同時発話は [MEROS & SELINUNTIUS | 同時に、涙声] のように書く。
必要な間は [WAIT 1.0] のように書く。
最後は [END]。

AI音声でそのまま演技できる程度に、短いセリフ単位で構成する。
SE/BGM/画像指示は音声として読ませない。

原作:
{source}
"""
    response = client.models.generate_content(model=GEMINI_SCRIPT_MODEL, contents=prompt)
    return response.text

# =========================================================
# UI
# =========================================================

st.title("📖 5分で名作 — 朗読ドラマ制作アプリ")
st.caption("原作→脚本→音声→タイムライン→SRT。普段は無料のローカルAivisSpeech、必要時のみGemini TTS。")

with st.sidebar:
    st.header("⚙️ 設定")
    engine = st.radio("音声エンジン", ["AivisSpeech（無料・ローカル）", "Gemini TTS（クラウド）"], index=0)
    speed = st.slider("完成音声の速度", 1.0, 2.0, 1.5, 0.05)
    pause = st.slider("セリフ間の基本の間（秒）", 0.0, 0.8, 0.12, 0.02)
    if engine.startswith("Aivis"):
        aivis_url = st.text_input("AivisSpeech Engine URL", AIVIS_DEFAULT_URL)
        st.caption("AivisSpeech EngineをPCで起動してください。既定ポートは10101です。")
    else:
        aivis_url = AIVIS_DEFAULT_URL
        gemini_key = st.text_input("Gemini API Key", type="password")

st.divider()

mode = st.radio("入力方法", ["原作からGeminiで脚本を作る", "完成した脚本をそのまま使う"], horizontal=True)
source_text = ""

if mode == "原作からGeminiで脚本を作る":
    uploaded = st.file_uploader("原作ファイル（txt / md）", type=["txt", "md"])
    if uploaded:
        source_text = uploaded.read().decode("utf-8", errors="replace")
    else:
        source_text = st.text_area("原作を貼り付け", height=260)
    minutes = st.slider("目標動画時間（分）", 3, 10, 5)
    if st.button("🪄 Geminiで脚本を生成", type="primary", disabled=not source_text.strip()):
        if not gemini_key:
            st.error("脚本生成にはGemini API Keyが必要です。")
        elif genai is None:
            st.error("google-genai がインストールされていません。requirements.txtを確認してください。")
        else:
            try:
                client = genai.Client(api_key=gemini_key)
                with st.spinner("脚本を生成しています…"):
                    generated = generate_script(client, source_text, minutes)
                st.session_state["script"] = generated
            except Exception as e:
                st.error(f"脚本生成に失敗しました：{e}")
else:
    uploaded_script = st.file_uploader("完成脚本（txt / md）", type=["txt", "md"])
    if uploaded_script:
        st.session_state["script"] = uploaded_script.read().decode("utf-8", errors="replace")
    else:
        st.session_state.setdefault("script", "")

script = st.text_area("現在の脚本", value=st.session_state.get("script", ""), height=500)
st.session_state["script"] = script

if script.strip():
    cast, segments, cues = parse_script(script)
    st.subheader("🎭 脚本解析")
    speech_count = sum(1 for s in segments if s.kind == "speech")
    st.write(f"セリフ {speech_count}件 / 指示 {len(cues)}件")

    # Detect simultaneous speech markers and normalize speakers.
    for seg in segments:
        if seg.kind == "speech":
            roles = [normalize_role(x) for x in re.split(r"\s*&\s*|＆", seg.speaker)]
            if len(roles) > 1:
                # Expand later in generation UI. The first role keeps this segment's index;
                # subsequent roles share the same index with a synthetic key.
                pass

    st.subheader("🎙️ キャスト")
    role_names = sorted({normalize_role(s.speaker) for s in segments if s.kind == "speech"})

    aivis_speakers = []
    if engine.startswith("Aivis"):
        try:
            aivis_speakers = flatten_aivis_speakers(aivis_get_speakers(aivis_url))
            st.success(f"AivisSpeech接続OK：{len(aivis_speakers)}スタイルを取得")
        except Exception:
            st.warning("AivisSpeech Engineに接続できません。PCでAivisSpeechを起動してから再試行してください。")

    cast_config = {}
    for role in role_names:
        meta = DEFAULT_ROLE_MAP.get(role, {"label": role, "gemini": GEMINI_VOICES[0]})
        label = meta.get("label", role)
        with st.expander(f"{label}（{role}）", expanded=True):
            if engine.startswith("Aivis"):
                if aivis_speakers:
                    options = [f"{x['speaker_name']} / {x['style_name']} / ID {x['speaker_id']}" for x in aivis_speakers]
                    default = 0
                    if role == "DIONYS":
                        default = min(2, len(options)-1)
                    elif role in ("MEROS", "FILOSTRATUS"):
                        default = min(1, len(options)-1)
                    choice = st.selectbox("AivisSpeechモデル/スタイル", options, index=default, key=f"aiv_{role}")
                    cast_config[role] = {"engine": "aivis", "style": aivis_speakers[options.index(choice)]}
                else:
                    cast_config[role] = {"engine": "aivis", "style": None}
            else:
                default_voice = meta.get("gemini", GEMINI_VOICES[0])
                voice = st.selectbox("Gemini Voice", GEMINI_VOICES, index=GEMINI_VOICES.index(default_voice) if default_voice in GEMINI_VOICES else 0, key=f"gem_{role}")
                cast_config[role] = {"engine": "gemini", "voice": voice}
            intensity = st.slider("感情強度", 0.0, 2.0, 1.0, 0.1, key=f"int_{role}")
            cast_config[role]["intonation"] = intensity

    st.subheader("🎬 SE / BGM / 画像指示")
    if not cues:
        st.info("脚本にSE/BGM/画像指示がありません。必要なら [SE想定：...] / [BGM：...] / [画像：...] を追加してください。")
    else:
        for cue in cues:
            sug = cue_suggestions(cue["type"], cue["text"])
            st.markdown(f"**{cue['type']}：{cue['text']}**")
            st.write("検索キーワード：", ", ".join(sug["keywords"]))
            if cue["type"] in ("SE", "BGM"):
                c1, c2 = st.columns(2)
                with c1:
                    st.link_button("YouTube Audio Libraryを開く", sug["youtube"])
                with c2:
                    st.link_button("DOVAを開く", sug["dova"])
                st.caption("YouTube Audio LibraryはYouTube公式の音楽・効果音。CCの場合は帰属表示が必要です。素材を採用したら下のクレジット欄に記録してください。")

    st.divider()
    if st.button("🎙️ 音声を生成して一本化", type="primary"):
        if engine.startswith("Aivis") and not aivis_speakers:
            st.error("AivisSpeech Engineに接続できません。まずAivisSpeechを起動してください。")
        elif engine.startswith("Gemini") and not gemini_key:
            st.error("Gemini API Keyを入力してください。")
        else:
            client = None
            if engine.startswith("Gemini"):
                client = genai.Client(api_key=gemini_key)
            audio_by_index = {}
            progress = st.progress(0.0)
            speech_segments = [s for s in segments if s.kind == "speech"]
            errors = []
            for n, seg in enumerate(speech_segments, start=1):
                role_parts = [normalize_role(x) for x in re.split(r"\s*&\s*|＆", seg.speaker)]
                # For simultaneous dialogue, generate the first speaker's audio here and
                # the second speaker's audio under a synthetic segment key.
                for part_i, role in enumerate(role_parts):
                    key = seg.index if part_i == 0 else seg.index * 1000 + part_i
                    cfg = cast_config.get(role)
                    if not cfg:
                        cfg = {"engine": engine.split("（")[0].lower()}
                    try:
                        if engine.startswith("Aivis"):
                            style = cfg.get("style")
                            if not style:
                                raise RuntimeError(f"{role}のAivisSpeechスタイルが選択されていません。")
                            audio_by_index[key] = aivis_synthesize(
                                aivis_url, seg.text, int(style["speaker_id"]),
                                speed=1.0, intonation=cfg.get("intonation", 1.0)
                            )
                        else:
                            voice = cfg.get("voice", GEMINI_VOICES[0])
                            audio_by_index[key] = gemini_synthesize(client, seg.text, voice, seg.direction)
                    except Exception as e:
                        errors.append(f"#{seg.index} {role}: {e}")
                progress.progress(n / max(1, len(speech_segments)))
            if errors:
                st.error("一部の音声生成に失敗しました。")
                for e in errors[:10]:
                    st.code(e)
            else:
                # Build timeline, including synthetic simultaneous voices.
                timeline = []
                cursor = 0.0
                for seg in segments:
                    if seg.kind == "wait":
                        cursor += float(seg.start or 0)
                        timeline.append({"index": seg.index, "kind": "wait", "start": cursor, "duration": float(seg.start or 0)})
                        continue
                    if seg.kind == "cue":
                        timeline.append({"index": seg.index, "kind": seg.cue_type, "cue": seg.cue_text, "start": cursor})
                        continue
                    role_parts = [normalize_role(x) for x in re.split(r"\s*&\s*|＆", seg.speaker)]
                    if len(role_parts) > 1:
                        durs = [wav_duration(audio_by_index[seg.index if i == 0 else seg.index*1000+i]) for i in range(len(role_parts))]
                        dur = max(durs or [0])
                        for i, role in enumerate(role_parts):
                            key = seg.index if i == 0 else seg.index*1000+i
                            timeline.append({"index": key, "source_index": seg.index, "kind": "speech", "speaker": role,
                                             "direction": seg.direction, "text": seg.text, "start": cursor, "duration": wav_duration(audio_by_index[key]),
                                             "simultaneous": True})
                        cursor += dur + pause
                    else:
                        dur = wav_duration(audio_by_index[seg.index])
                        timeline.append({"index": seg.index, "source_index": seg.index, "kind": "speech", "speaker": role_parts[0],
                                         "direction": seg.direction, "text": seg.text, "start": cursor, "duration": dur})
                        cursor += dur + pause
                # Mix using synthetic keys.
                max_end = max([x.get("start", 0)+x.get("duration", 0) for x in timeline if x.get("kind") == "speech"] + [0.1])
                final = silence_wav(max_end + 0.2)
                for item in timeline:
                    if item.get("kind") == "speech":
                        final = mix_wavs(final, audio_by_index[item["index"]], item["start"])
                final_fast = speed_up_wav(final, speed)
                srt = make_srt(timeline, speed)
                credits = [
                    "5分で名作 — 音声・素材クレジット", "",
                    f"Voice engine: {engine}",
                    f"Playback speed: {speed}x", "",
                ]
                for role, cfg in cast_config.items():
                    if cfg.get("engine") == "aivis" and cfg.get("style"):
                        x = cfg["style"]
                        credits.append(f"{role}: AivisSpeech / {x['speaker_name']} / {x['style_name']} / style_id={x['speaker_id']}")
                    elif cfg.get("engine") == "gemini":
                        credits.append(f"{role}: Google Gemini TTS / {cfg.get('voice')}")
                credits += ["", "SE/BGM: 使用素材ごとに、提供元のライセンス条件と必要な帰属表示を動画説明欄へ追加してください."]
                st.session_state["final_wav"] = final
                st.session_state["final_fast"] = final_fast
                st.session_state["srt"] = srt
                st.session_state["timeline"] = timeline
                st.session_state["credits"] = "\n".join(credits)
                st.success("音声をタイムライン化して一本にしました。")

if st.session_state.get("final_wav"):
    st.divider()
    st.subheader("🎧 完成音声")
    st.audio(st.session_state["final_fast"], format="audio/wav")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("WAV（元速度）", st.session_state["final_wav"], "master.wav", "audio/wav")
    with c2:
        st.download_button("WAV（1.5倍など設定速度）", st.session_state["final_fast"], "master_fast.wav", "audio/wav")
    with c3:
        st.download_button("SRT字幕", st.session_state["srt"], "subtitles.srt", "application/x-subrip")

    st.subheader("📋 タイムライン")
    for item in st.session_state.get("timeline", []):
        if item.get("kind") == "speech":
            st.write(f"{item['start']:.2f}s — {speaker_label(item['speaker'])} — {item['text']}")
        elif item.get("kind") in ("SE", "BGM", "IMAGE"):
            st.write(f"{item['start']:.2f}s — {item['kind']} — {item['cue']}")
        else:
            st.write(f"{item['start']:.2f}s — WAIT {item.get('duration', 0):.2f}s")

    st.subheader("🧾 クレジット記録")
    st.code(st.session_state.get("credits", ""), language="text")
    st.download_button("クレジットTXT", st.session_state.get("credits", ""), "credits.txt", "text/plain")

st.divider()
st.caption("素材の採用前に各素材ページのライセンスを確認してください。YouTube Audio LibraryはYouTube公式の音楽・効果音ライブラリです。")
