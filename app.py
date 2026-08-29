import io
import json
import base64
import wave
import re
import array
import time

import streamlit as st
from google import genai


# =========================================================
# 基本設定
# =========================================================

st.set_page_config(
    page_title="5分で名作",
    page_icon="📖",
    layout="wide"
)

TTS_MODEL = "gemini-3.1-flash-tts-preview"
SCRIPT_MODEL = "gemini-3.6-flash"

# Gemini TTSの公式例に合わせる
SAMPLE_RATE = 24000
CHANNELS = 1
SAMPLE_WIDTH = 2

# 通常時の1チャンクの目安
# 長すぎる入力を避け、声の安定性と失敗率のバランスを取る
DEFAULT_CHUNK_CHARS = 450

# Safety等で失敗したとき、さらに細かくする
RETRY_CHUNK_CHARS = 220

MAX_RETRIES = 3

SPEED = 1.5


# =========================================================
# キャスト
# =========================================================

VOICES = [
    "Zephyr",
    "Puck",
    "Charon",
    "Kore",
    "Fenrir",
    "Leda",
    "Orus",
    "Aoede",
    "Callirrhoe",
    "Autonoe",
    "Enceladus",
    "Iapetus",
    "Umbriel",
    "Algieba",
    "Despina",
    "Erinome",
    "Algenib",
    "Rasalgethi",
    "Laomedeia",
    "Achernar",
    "Alnilam",
    "Schedar",
    "Gacrux",
    "Pulcherrima",
    "Achird",
    "Zubenelgenubi",
    "Vindemiatrix",
    "Sadachbia",
    "Sadaltager",
    "Sulafat"
]

DEFAULT = {
    "ナレーター": "Rasalgethi",
    "メロス": "Iapetus",
    "王": "Algieba",
    "セリヌンティウス": "Alnilam",
    "妹": "Aoede",
    "フィロストラトス": "Schedar"
}


# =========================================================
# WAV関連
# =========================================================

def pcm_to_wav(pcm: bytes) -> bytes:
    """
    Gemini TTSの生PCMをWAVに変換する。
    Gemini公式サンプルと同じ24kHz / 16bit / mono。
    """

    wav_buffer = io.BytesIO()

    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)

    return wav_buffer.getvalue()


def duration(wav_bytes: bytes) -> float:
    """WAVの長さを秒で取得"""

    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        return w.getnframes() / w.getframerate()


def join_wav(parts):
    """複数のWAVを一本に結合"""

    if not parts:
        return b""

    first = io.BytesIO(parts[0])

    with wave.open(first, "rb") as w:
        params = w.getparams()

    output = io.BytesIO()

    with wave.open(output, "wb") as out:
        out.setparams(params)

        for part in parts:
            with wave.open(io.BytesIO(part), "rb") as w:
                frames = w.readframes(w.getnframes())
                out.writeframes(frames)

    return output.getvalue()


def speed_up(wav_bytes: bytes, factor=1.5) -> bytes:
    """
    音声を単純なサンプル間引きで高速化。
    今回は編集時に1.5倍速前提なので、
    ここでは音程を大きく変えずに簡易高速化する。
    """

    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        params = w.getparams()
        raw = w.readframes(w.getnframes())

    if params.sampwidth != 2:
        return wav_bytes

    samples = array.array("h")
    samples.frombytes(raw)

    frames = len(samples) // params.nchannels

    result = array.array("h")

    index = 0.0

    while int(index) < frames:
        frame_index = int(index) * params.nchannels

        result.extend(
            samples[
                frame_index:
                frame_index + params.nchannels
            ]
        )

        index += factor

    output = io.BytesIO()

    with wave.open(output, "wb") as w:
        w.setnchannels(params.nchannels)
        w.setsampwidth(2)
        w.setframerate(params.framerate)
        w.writeframes(result.tobytes())

    return output.getvalue()


# =========================================================
# 文章分割
# =========================================================

def normalize_text(text):
    """TTSに渡す文章を軽く整理"""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # 連続空白を整理
    text = re.sub(r"[ \t]+", " ", text)

    # 連続改行を整理
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def split_text(text, max_chars=DEFAULT_CHUNK_CHARS):
    """
    日本語文章を意味がなるべく切れないように分割。

    優先順位:
    1. 句点
    2. 読点
    3. 改行
    4. それでも長ければ文字数
    """

    text = normalize_text(text)

    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    chunks = []

    remaining = text

    punctuation = "。！？!?」』"

    while len(remaining) > max_chars:

        search_area = remaining[:max_chars]

        cut_positions = [
            i + 1
            for i, char in enumerate(search_area)
            if char in punctuation
        ]

        if cut_positions:
            cut = max(cut_positions)

        else:
            # 読点・改行を探す
            comma_positions = [
                i + 1
                for i, char in enumerate(search_area)
                if char in "、,\n"
            ]

            if comma_positions:
                cut = max(comma_positions)

            else:
                cut = max_chars

        chunk = remaining[:cut].strip()

        if chunk:
            chunks.append(chunk)

        remaining = remaining[cut:].strip()

    if remaining:
        chunks.append(remaining)

    return chunks


# =========================================================
# SRT
# =========================================================

def srt_time(seconds):
    ms = int(round(seconds * 1000))

    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    secs, ms = divmod(ms, 1000)

    return f"{hours:02}:{minutes:02}:{secs:02},{ms:03}"


def make_srt(segments, speed_factor=1.0):
    """
    speed_factor=1.5なら、
    1.5倍速後の時間に合わせてSRTを作る。
    """

    current = 0.0
    output = []

    for index, seg in enumerate(segments, 1):

        original_duration = seg["duration"]

        # 1.5倍速なら実時間は1/1.5
        actual_duration = original_duration / speed_factor

        start = current
        end = current + actual_duration

        speaker = seg["speaker"]
        text = seg["text"]

        output.append(
            f"{index}\n"
            f"{srt_time(start)} --> {srt_time(end)}\n"
            f"{speaker}: {text}\n"
        )

        current = end

    return "\n".join(output)


# =========================================================
# Gemini TTS
# =========================================================

def tts(client, text, voice):
    """
    Gemini TTSで日本語音声を生成する。
    返り値はWAVではなく、Geminiが返したPCMデータをWAV化したもの。
    """

    text = str(text).strip()

    if not text:
        raise RuntimeError("TTSに渡す文章が空です。")

    prompt = (
        "Read the following Japanese text naturally and expressively. "
        "Speak only the Japanese text. "
        "Do not explain it, translate it, or add any words.\n\n"
        "Japanese text:\n"
        + text
    )

    r = client.interactions.create(
        model="gemini-3.1-flash-tts-preview",
        input=prompt,
        response_format={"type": "audio"},
        generation_config={
            "speech_config": [
                {
                    "voice": voice
                }
            ]
        }
    )

    # Geminiの音声データを取得
    if not hasattr(r, "output_audio") or r.output_audio is None:
        raise RuntimeError("Geminiから音声データが返されませんでした。")

    audio_data = r.output_audio.data

    if audio_data is None:
        raise RuntimeError("Geminiから空の音声データが返されました。")

    # Base64文字列の場合
    if isinstance(audio_data, str):
        pcm_data = base64.b64decode(audio_data)
    else:
        pcm_data = bytes(audio_data)

    if not pcm_data:
        raise RuntimeError("音声データが空です。")

    # PCM → WAV
    wav_buffer = io.BytesIO()

    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(pcm_data)

    return wav_buffer.getvalue()


def tts_with_recovery(client, text, voice):
    """
    TTS生成。
    
    1. 通常サイズで生成
    2. 失敗したら短くして再試行
    3. それでも失敗したらさらに分割
    """

    text = normalize_text(text)

    if not text:
        return []

    # -----------------------------------------------------
    # まず通常サイズで分割
    # -----------------------------------------------------

    chunks = split_text(
        text,
        DEFAULT_CHUNK_CHARS
    )

    results = []

    for chunk in chunks:

        success = False

        # -------------------------------------------------
        # 通常リトライ
        # -------------------------------------------------

        for attempt in range(MAX_RETRIES):

            try:

                audio = raw_tts(
                    client,
                    chunk,
                    voice
                )

                results.append(
                    {
                        "audio": audio,
                        "text": chunk
                    }
                )

                success = True
                break

            except Exception as e:

                error_text = str(e)

                # Safety / stream / 400系など
                # 同じ文章を何度も投げ続けない
                if attempt < MAX_RETRIES - 1:

                    time.sleep(
                        1.5 * (attempt + 1)
                    )

                else:

                    # -------------------------------------
                    # ここから細分化
                    # -------------------------------------

                    smaller_chunks = split_text(
                        chunk,
                        RETRY_CHUNK_CHARS
                    )

                    if len(smaller_chunks) <= 1:
                        raise RuntimeError(
                            "音声生成に失敗しました。\n\n"
                            f"キャラクター: {voice}\n"
                            f"文章: {chunk}\n\n"
                            f"Geminiエラー:\n{error_text}"
                        )

                    # 細分化したものを再帰的に生成
                    smaller_results = []

                    for small_chunk in smaller_chunks:

                        small_audio = tts_with_recovery(
                            client,
                            small_chunk,
                            voice
                        )

                        smaller_results.extend(
                            small_audio
                        )

                    results.extend(
                        smaller_results
                    )

                    success = True

        if not success:
            raise RuntimeError(
                "音声生成に失敗しました。"
            )

    return results


# =========================================================
# 脚本生成
# =========================================================

def make_script(client, source, minutes, roles):

    source = normalize_text(source)

    role_text = "、".join(roles)

    prompt = f"""
あなたはYouTubeチャンネル「5分で名作」の脚本家です。

以下の原作を、視聴者が最後まで見たくなる
「短く、速く、インパクトのある物語」
として再構成してください。

解説動画にはしないでください。
あくまで「物語」として語ってください。

目標尺:
{minutes}分

重要:
- 原作の重要な出来事だけ残す
- 冗長な説明は削る
- 冒頭からすぐ物語を始める
- 展開を早める
- 感情が動く場面は残す
- 最後のオチ・結末はしっかり描く
- キャラクターのセリフを適切に入れる
- ナレーションだけで全部説明しない
- セリフは自然な日本語にする
- 原作の雰囲気をなるべく残す
- YouTubeで聞きやすい文章にする

使用可能な話者:
{role_text}

話者名は必ず上記の名前のどれかを使用してください。

以下のJSONだけを返してください。

{{
  "title": "作品タイトル",
  "segments": [
    {{
      "speaker": "ナレーター",
      "text": "..."
    }},
    {{
      "speaker": "メロス",
      "text": "..."
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
        input=prompt
    )

    output = response.output_text

    # JSON部分だけ取り出す
    match = re.search(
        r"\{.*\}",
        output,
        re.DOTALL
    )

    if not match:
        raise RuntimeError(
            "脚本JSONを取得できませんでした。"
        )

    try:
        return json.loads(
            match.group()
        )

    except json.JSONDecodeError as e:
        raise RuntimeError(
            "脚本JSONの解析に失敗しました。\n"
            + str(e)
        )


# =========================================================
# Streamlit UI
# =========================================================

if "cast" not in st.session_state:
    st.session_state.cast = DEFAULT.copy()


st.title("📖 5分で名作・自動音声制作")

st.caption(
    "原作 → 脚本 → キャラ別TTS → 自動分割 → 結合 → 1.5倍速 → SRT"
)


# =========================================================
# サイドバー
# =========================================================

with st.sidebar:

    st.subheader("🔑 Gemini")

    api_key = st.text_input(
        "Gemini API Key",
        type="password"
    )

    st.caption(
        "APIキーはこの画面だけに入力してください。"
    )

    st.divider()

    st.subheader("🎙 キャスト")

    for role in list(st.session_state.cast):

        current_voice = st.session_state.cast[role]

        if current_voice not in VOICES:
            current_voice = "Kore"

        st.session_state.cast[role] = st.selectbox(
            role,
            VOICES,
            index=VOICES.index(current_voice),
            key=f"voice_{role}"
        )

    st.divider()

    if st.button("＋キャラクター追加"):

        n = 1

        while f"キャラクター{n}" in st.session_state.cast:
            n += 1

        st.session_state.cast[
            f"キャラクター{n}"
        ] = "Kore"

        st.rerun()


# =========================================================
# 原作入力
# =========================================================

uploaded = st.file_uploader(
    "📚 原作 TXT / MD",
    type=["txt", "md"]
)

if uploaded:

    source = uploaded.read().decode(
        "utf-8",
        "replace"
    )

else:

    source = st.text_area(
        "または本文を貼り付け",
        height=220,
        placeholder="ここに青空文庫などの本文を貼り付けます。"
    )


# =========================================================
# 尺
# =========================================================

minutes = st.radio(
    "🎬 動画の長さ",
    [1, 5, 10],
    index=1,
    horizontal=True
)


# =========================================================
# 生成
# =========================================================

generate = st.button(
    "🚀 生成開始",
    type="primary",
    disabled=not (
        api_key and
        source.strip()
    )
)


if generate:

    try:

        client = genai.Client(
            api_key=api_key
        )

        with st.status(
            "🎬 制作中…",
            expanded=True
        ):

            # ---------------------------------------------
            # STEP 1
            # ---------------------------------------------

            st.write(
                "① 脚本を生成しています…"
            )

            result = make_script(
                client,
                source,
                minutes,
                list(st.session_state.cast.keys())
            )

            segments = result.get(
                "segments",
                []
            )

            if not segments:
                raise RuntimeError(
                    "脚本にセグメントがありません。"
                )

            st.write(
                f"脚本完成：{len(segments)}セグメント"
            )

            # ---------------------------------------------
            # STEP 2
            # 音声生成
            # ---------------------------------------------

            st.write(
                "② キャラクター別音声を生成しています…"
            )

            all_audio = []
            timed_segments = []

            total_segments = len(segments)

            progress = st.progress(0)

            for index, segment in enumerate(
                segments
            ):

                speaker = segment.get(
                    "speaker",
                    "ナレーター"
                )

                text = normalize_text(
                    segment.get(
                        "text",
                        ""
                    )
                )

                # 存在しないキャラならナレーター
                if speaker not in st.session_state.cast:
                    speaker = "ナレーター"

                if not text:
                    progress.progress(
                        (index + 1) / total_segments
                    )
                    continue

                voice = st.session_state.cast[
                    speaker
                ]

                st.write(
                    f"🎙 {speaker} / {voice}"
                )

                generated_parts = tts_with_recovery(
                    client,
                    text,
                    voice
                )

                for part in generated_parts:

                    audio = part["audio"]

                    part_text = part["text"]

                    part_duration = duration(
                        audio
                    )

                    all_audio.append(
                        audio
                    )

                    timed_segments.append(
                        {
                            "speaker": speaker,
                            "text": part_text,
                            "duration": part_duration
                        }
                    )

                progress.progress(
                    (index + 1) / total_segments
                )

            if not all_audio:
                raise RuntimeError(
                    "音声を1つも生成できませんでした。"
                )

            # ---------------------------------------------
            # STEP 3
            # 結合
            # ---------------------------------------------

            st.write(
                "③ 音声を一本に結合しています…"
            )

            original_audio = join_wav(
                all_audio
            )

            # ---------------------------------------------
            # STEP 4
            # 1.5倍速
            # ---------------------------------------------

            st.write(
                "④ 1.5倍速版を作っています…"
            )

            fast_audio = speed_up(
                original_audio,
                SPEED
            )

            # ---------------------------------------------
            # STEP 5
            # SRT
            # ---------------------------------------------

            st.write(
                "⑤ 1.5倍速版に合わせて字幕を作っています…"
            )

            srt = make_srt(
                timed_segments,
                SPEED
            )

            # ---------------------------------------------
            # 保存
            # ---------------------------------------------

            result["timed_segments"] = (
                timed_segments
            )

            st.session_state.result = result
            st.session_state.original = (
                original_audio
            )
            st.session_state.fast = (
                fast_audio
            )
            st.session_state.srt = srt

        st.success(
            "🎉 完成しました！"
        )

    except Exception as e:

        st.error(
            "音声生成中にエラーが発生しました。"
        )

        st.code(
            str(e)
        )


# =========================================================
# 結果表示
# =========================================================

if "result" in st.session_state:

    result = st.session_state.result

    st.divider()

    st.header(
        "🎬 " + result.get(
            "title",
            "完成"
        )
    )

    # ---------------------------------------------
    # 音声
    # ---------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "🎧 元音声"
        )

        st.audio(
            st.session_state.original,
            format="audio/wav"
        )

        st.download_button(
            "⬇ 元音声 WAV",
            st.session_state.original,
            "audio_original.wav",
            "audio/wav"
        )

    with col2:

        st.subheader(
            "⚡ 1.5倍速版"
        )

        st.audio(
            st.session_state.fast,
            format="audio/wav"
        )

        st.download_button(
            "⬇ 1.5倍速 WAV",
            st.session_state.fast,
            "audio_1.5x.wav",
            "audio/wav"
        )

    # ---------------------------------------------
    # SRT
    # ---------------------------------------------

    st.subheader(
        "💬 字幕"
    )

    st.download_button(
        "⬇ SRT字幕",
        st.session_state.srt,
        "subtitles.srt",
        "application/x-subrip"
    )

    # ---------------------------------------------
    # 制作指示
    # ---------------------------------------------

    st.subheader(
        "🎨 編集用指示"
    )

    notes = {
        "images": result.get(
            "image_instructions",
            []
        ),
        "SE": result.get(
            "se_instructions",
            []
        ),
        "BGM": result.get(
            "bgm_instructions",
            []
        )
    }

    st.download_button(
        "⬇ 画像・SE・BGM指示 JSON",
        json.dumps(
            notes,
            ensure_ascii=False,
            indent=2
        ),
        "production_notes.json",
        "application/json"
    )

    # ---------------------------------------------
    # 脚本
    # ---------------------------------------------

    with st.expander(
        "📜 完成した脚本を見る"
    ):

        for segment in result.get(
            "segments",
            []
        ):

            speaker = segment.get(
                "speaker",
                "ナレーター"
            )

            text = segment.get(
                "text",
                ""
            )

            st.markdown(
                f"**{speaker}**　{text}"
            )

    # ---------------------------------------------
    # 画像指示
    # ---------------------------------------------

    with st.expander(
        "🖼 画像指示を見る"
    ):

        for item in result.get(
            "image_instructions",
            []
        ):

            st.markdown(
                f"**{item.get('time_hint', '')}** "
                f"{item.get('visual', '')}"
            )

    # ---------------------------------------------
    # SE
    # ---------------------------------------------

    with st.expander(
        "🔊 SE指示を見る"
    ):

        for item in result.get(
            "se_instructions",
            []
        ):

            st.markdown(
                f"**{item.get('time_hint', '')}** "
                f"{item.get('sound', '')} "
                f"— {item.get('purpose', '')}"
            )

    # ---------------------------------------------
    # BGM
    # ---------------------------------------------

    with st.expander(
        "🎵 BGM指示を見る"
    ):

        for item in result.get(
            "bgm_instructions",
            []
        ):

            st.markdown(
                f"**{item.get('time_hint', '')}** "
                f"{item.get('mood', '')} "
                f"— {item.get('purpose', '')}"
            )
