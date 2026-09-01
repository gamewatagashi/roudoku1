import io
import re
import wave
import array
from dataclasses import dataclass

import requests
import streamlit as st

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None


st.set_page_config(page_title="5分で名作", page_icon="📖", layout="wide")

AIVIS_DEFAULT = "http://127.0.0.1:10101"
SCRIPT_MODEL = "gemini-3.6-flash"
TTS_MODEL = "gemini-3.1-flash-tts-preview"
SR = 24000

ROLES = {
    "NARRATOR": "ナレーター",
    "MEROS": "メロス",
    "DIONYS": "ディオニス／王",
    "SELINUNTIUS": "セリヌンティウス",
    "SISTER": "妹",
    "FILOSTRATUS": "フィロストラトス",
}

ALIASES = {
    "ナレーター": "NARRATOR",
    "ナレーション": "NARRATOR",
    "メロス": "MEROS",
    "王": "DIONYS",
    "ディオニス": "DIONYS",
    "セリヌンティウス": "SELINUNTIUS",
    "妹": "SISTER",
    "フィロストラトス": "FILOSTRATUS",
}

DEFAULT_AIVIS = {
    "NARRATOR": "阿井田 茂",
    "MEROS": "にせ",
    "DIONYS": "ろてじん",
    "SELINUNTIUS": "澤原 玄二郎",
    "SISTER": "まお",
    "FILOSTRATUS": "らせつん",
}

GEMINI_VOICES = [
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede",
    "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba",
    "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar",
    "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi",
    "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat"
]

AOZORA = [
    ("走れメロス", "太宰治", "https://www.aozora.gr.jp/cards/000035/card1567.html"),
    ("羅生門", "芥川龍之介", "https://www.aozora.gr.jp/cards/000879/card127.html"),
    ("蜘蛛の糸", "芥川龍之介", "https://www.aozora.gr.jp/cards/000879/card92.html"),
    ("注文の多い料理店", "宮沢賢治", "https://www.aozora.gr.jp/cards/000081/card43754.html"),
    ("銀河鉄道の夜", "宮沢賢治", "https://www.aozora.gr.jp/cards/000081/card456.html"),
    ("坊っちゃん", "夏目漱石", "https://www.aozora.gr.jp/cards/000148/card752.html"),
    ("吾輩は猫である", "夏目漱石", "https://www.aozora.gr.jp/cards/000148/card789.html"),
    ("檸檬", "梶井基次郎", "https://www.aozora.gr.jp/cards/000074/card423.html"),
    ("セロ弾きのゴーシュ", "宮沢賢治", "https://www.aozora.gr.jp/cards/000081/card470.html"),
]

@dataclass
class Seg:
    i: int
    speaker: str
    text: str
    direction: str = ""
    kind: str = "speech"
    cue: str = ""
    cue_type: str = ""
    wait: float = 0.0


def normalize_role(value):
    value = str(value).strip()
    return ALIASES.get(value, value.upper())


def role_label(value):
    return ROLES.get(normalize_role(value), str(value))


def clean_text(value):
    value = re.sub(r"\s+", " ", str(value).strip())
    return re.sub(r'^[「『"\'“”]|[」』"\'“”]$', "", value).strip()


def parse_script(text):
    segments = []
    current = None
    direction = ""
    buffer = []
    n = 1

    def flush():
        nonlocal buffer, n
        if current and buffer:
            body = clean_text("\n".join(buffer))
            if body:
                segments.append(Seg(n, current, body, direction))
                n += 1
        buffer = []

    for raw in text.replace("\r", "").split("\n"):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        wait_match = re.match(r"^\[\s*WAIT\s+([0-9.]+)\s*\]$", line, re.I)
        if wait_match:
            flush()
            segments.append(Seg(n, "", "", kind="wait", wait=float(wait_match.group(1))))
            n += 1
            continue

        cue_match = re.match(
            r"^\[\s*(SE想定|SE|BGM|画像|画像指示|映像)\s*[：:]\s*(.*?)\s*\]$",
            line,
            re.I,
        )
        if cue_match:
            flush()
            raw_type = cue_match.group(1).upper()
            cue_type = "SE" if raw_type.startswith("SE") else (
                "BGM" if raw_type == "BGM" else "IMAGE"
            )
            segments.append(
                Seg(n, "", "", kind="cue", cue=cue_match.group(2), cue_type=cue_type)
            )
            n += 1
            continue

        speaker_match = (
            re.match(r"^\[\s*(.*?)\s*(?:\|\s*(.*?))?\s*\]$", line)
            or re.match(r"^【\s*(.*?)\s*(?:\|\s*(.*?))?\s*】$", line)
        )
        if speaker_match:
            flush()
            current = speaker_match.group(1).strip()
            direction = (speaker_match.group(2) or "").strip()
            continue

        if line == "[END]":
            flush()
            current = None
            direction = ""
            continue

        if current:
            buffer.append(line)

    flush()
    return segments


def ensure_wav(data):
    if data[:4] == b"RIFF":
        return data
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data)
    return out.getvalue()


def wav_duration(data):
    with wave.open(io.BytesIO(data), "rb") as w:
        return w.getnframes() / w.getframerate()


def silence(seconds):
    frames = max(0, int(seconds * SR))
    return ensure_wav(b"\0\0" * frames)


def overlay_wav(base, over, start_seconds):
    with wave.open(io.BytesIO(base), "rb") as a:
        a_frames = a.readframes(a.getnframes())
    with wave.open(io.BytesIO(over), "rb") as b:
        b_frames = b.readframes(b.getnframes())

    x = array.array("h")
    x.frombytes(a_frames)
    y = array.array("h")
    y.frombytes(b_frames)

    offset = int(start_seconds * SR)
    needed = offset + len(y)
    if len(x) < needed:
        x.extend([0] * (needed - len(x)))

    for i, value in enumerate(y):
        x[offset + i] = max(-32768, min(32767, x[offset + i] + value))

    return ensure_wav(x.tobytes())


def speed_up(data, factor):
    # Simple resampling by sample selection. The resulting WAV is intentionally
    # shorter and remains playable; pitch rises slightly, which is acceptable
    # for the 1.5x editing workflow.
    if factor <= 1:
        return data

    with wave.open(io.BytesIO(data), "rb") as w:
        raw = w.readframes(w.getnframes())

    samples = array.array("h")
    samples.frombytes(raw)
    output = array.array("h")
    pos = 0.0
    while int(pos) < len(samples):
        output.append(samples[int(pos)])
        pos += factor
    return ensure_wav(output.tobytes())


def get_aivis_speakers(base_url):
    response = requests.get(base_url.rstrip("/") + "/speakers", timeout=10)
    response.raise_for_status()
    result = []
    for speaker in response.json():
        for style in speaker.get("styles", []):
            result.append({
                "speaker_name": speaker.get("name", ""),
                "style_name": style.get("name", ""),
                "id": int(style["id"]),
            })
    return result


def aivis_synthesize(base_url, text, style_id, intonation=1.0):
    query = requests.post(
        base_url.rstrip("/") + "/audio_query",
        params={"speaker": style_id},
        data=text.encode("utf-8"),
        timeout=60,
    )
    query.raise_for_status()
    query_json = query.json()
    query_json["intonationScale"] = max(0.0, min(2.0, float(intonation)))

    response = requests.post(
        base_url.rstrip("/") + "/synthesis",
        params={"speaker": style_id},
        json=query_json,
        timeout=180,
    )
    response.raise_for_status()
    return ensure_wav(response.content)


def gemini_tts(client, text, voice, direction=""):
    prompt = f"演技指示：{direction}\n\n{text}" if direction else text
    response = client.models.generate_content(
        model=TTS_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice
                    )
                )
            ),
        ),
    )

    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                return ensure_wav(inline.data)

    raise RuntimeError("Gemini TTSから音声データが返りませんでした。")


def generate_script(client, source, minutes):
    prompt = f"""あなたは「5分で名作」の脚本家です。
以下の原作を約{minutes}分の朗読ドラマ脚本にしてください。

条件：
- ナレーションと会話を明確に分ける
- 会話は短めの発話単位にする
- 登場人物ごとに声を割り当てやすくする
- 必要に応じて[SE想定：...]、[BGM：...]、[画像：...]、[WAIT 1.0]を使う
- 同時発話は[人物A & 人物B | 同時に]の形式にする
- 最後は[END]
- 余計な説明は付けず、VOICE CASTとSCRIPTだけ返す

原作：
{source}"""
    return client.models.generate_content(
        model=SCRIPT_MODEL,
        contents=prompt
    ).text


def srt_timestamp(seconds):
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3600000
    total_ms %= 3600000
    minutes = total_ms // 60000
    total_ms %= 60000
    secs = total_ms // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def build_srt(timeline, speed):
    speech = [x for x in timeline if x["kind"] == "speech"]
    blocks = []
    for n, item in enumerate(speech, 1):
        start = item["start"] / speed
        end = (item["start"] + item["duration"]) / speed
        blocks.append(
            f"{n}\n{srt_timestamp(start)} --> {srt_timestamp(end)}\n"
            f"[{role_label(item['speaker'])}] {item['text']}\n"
        )
    return "\n".join(blocks)


st.title("📖 5分で名作")
st.caption("原作 → 脚本 → キャスト → 音声 → タイムライン → WAV / SRT / クレジット")

if "script" not in st.session_state:
    st.session_state.script = ""
if "final" not in st.session_state:
    st.session_state.final = None
if "fast" not in st.session_state:
    st.session_state.fast = None
if "srt" not in st.session_state:
    st.session_state.srt = ""
if "credits" not in st.session_state:
    st.session_state.credits = ""
if "timeline" not in st.session_state:
    st.session_state.timeline = []

with st.sidebar:
    st.header("⚙️ 音声設定")

    engine = st.radio(
        "音声エンジン",
        ["AivisSpeech（無料・ローカル）", "Gemini TTS（クラウド）"],
    )
    speed = st.slider("完成音声速度", 1.0, 2.0, 1.5, 0.05)
    pause = st.slider("通常のセリフ間", 0.0, 0.8, 0.12, 0.02)

    aivis_url = AIVIS_DEFAULT
    gemini_key = ""

    if engine.startswith("Aivis"):
        st.subheader("🔌 AivisSpeech接続")
        aivis_url = st.text_input("AivisSpeech Engine URL", AIVIS_DEFAULT)

        if st.button("接続テスト"):
            try:
                found = get_aivis_speakers(aivis_url)
                st.session_state.aivis_styles = found
                st.success(f"接続成功：{len(found)}スタイル")
            except Exception as exc:
                st.error(f"接続失敗：{exc}")

        st.caption(
            "AivisSpeechをPCで起動している場合は、通常 "
            "http://127.0.0.1:10101 を指定します。"
        )
        st.warning(
            "Streamlit Cloud上のアプリから、あなたのPCの127.0.0.1へは接続できません。"
            "AivisSpeechを使う場合は、アプリも同じPCで動かすか、到達可能なEngine URLを指定してください。"
        )
    else:
        st.subheader("☁️ Gemini TTS")
        gemini_key = st.text_input("Gemini API Key", type="password")
        st.caption("Gemini TTSはAPIクォータを消費します。429が出た場合は待ってから再実行してください。")


st.subheader("📚 作品を用意")

tab_a, tab_b, tab_c = st.tabs(["📚 青空文庫", "📄 原作", "📝 完成脚本"])

with tab_a:
    choices = [f"{title} — {author}" for title, author, _ in AOZORA]
    selected = st.selectbox("参考作品", choices)
    url = next(u for t, a, u in AOZORA if f"{t} — {a}" == selected)

    st.link_button("🔗 青空文庫で開く", url)
    st.link_button("🔎 青空文庫総合インデックス", "https://www.aozora.gr.jp/")
    st.caption("公開前に、作品の著作権状態と青空文庫の利用規準を確認してください。")

with tab_b:
    original_file = st.file_uploader(
        "原作 TXT / MD", type=["txt", "md"], key="original_upload"
    )
    if original_file:
        original_text = original_file.read().decode("utf-8", "replace")
    else:
        original_text = st.text_area("原作本文", height=220)

with tab_c:
    script_file = st.file_uploader(
        "完成脚本 TXT / MD", type=["txt", "md"], key="script_upload"
    )
    if script_file:
        st.session_state.script = script_file.read().decode("utf-8", "replace")

    st.session_state.script = st.text_area(
        "完成脚本",
        value=st.session_state.script,
        height=320,
        key="script_editor",
    )
    st.caption(
        "形式例：[NARRATOR | 静かに]、[MEROS & SELINUNTIUS | 同時に]、"
        "[WAIT 1.0]、[SE想定：足音]、[BGM：緊張感]、[END]"
    )


if original_text.strip():
    st.divider()
    st.subheader("🪄 原作から脚本生成")

    target_minutes = st.slider("目標分数", 1, 15, 5)

    if st.button("Geminiで脚本を生成"):
        if genai is None:
            st.error("google-genai がインストールされていません。requirements.txtを確認してください。")
        elif not gemini_key:
            st.error("Gemini API Keyを入力してください。")
        else:
            try:
                client = genai.Client(api_key=gemini_key)
                st.session_state.script = generate_script(
                    client, original_text, target_minutes
                )
                st.rerun()
            except Exception as exc:
                st.error(f"脚本生成失敗：{exc}")


if st.session_state.script.strip():
    st.divider()
    st.subheader("📜 脚本を確認")

    segments = parse_script(st.session_state.script)
    speech_segments = [x for x in segments if x.kind == "speech"]

    st.info(f"音声セグメント：{len(speech_segments)}件")

    if "aivis_styles" not in st.session_state:
        try:
            if engine.startswith("Aivis"):
                st.session_state.aivis_styles = get_aivis_speakers(aivis_url)
        except Exception:
            st.session_state.aivis_styles = []

    styles = st.session_state.get("aivis_styles", [])

    used_roles = []
    for segment in speech_segments:
        for raw_role in re.split(r"\s*&\s*|＆", segment.speaker):
            r = normalize_role(raw_role)
            if r not in used_roles:
                used_roles.append(r)

    st.subheader("🎙️ キャスト")
    cast = {}

    for r in used_roles:
        with st.expander(f"🎙 {role_label(r)}", expanded=True):
            if engine.startswith("Aivis"):
                if not styles:
                    st.warning("AivisSpeechの話者一覧を取得できていません。接続テストを押してください。")
                    cast[r] = {"style": None}
                else:
                    options = [
                        f"{x['speaker_name']} / {x['style_name']} / ID {x['id']}"
                        for x in styles
                    ]
                    preferred = DEFAULT_AIVIS.get(r)
                    default_index = next(
                        (i for i, x in enumerate(styles)
                         if x["speaker_name"] == preferred),
                        0,
                    )
                    selected_option = st.selectbox(
                        "AivisSpeechモデル / スタイル",
                        options,
                        index=default_index,
                        key=f"aivis_cast_{r}",
                    )
                    cast[r] = {
                        "style": styles[options.index(selected_option)]
                    }
            else:
                defaults = {
                    "NARRATOR": "Rasalgethi",
                    "MEROS": "Iapetus",
                    "DIONYS": "Algieba",
                    "SELINUNTIUS": "Alnilam",
                    "SISTER": "Aoede",
                    "FILOSTRATUS": "Schedar",
                }
                default_voice = defaults.get(r, "Rasalgethi")
                selected_voice = st.selectbox(
                    "Gemini Voice",
                    GEMINI_VOICES,
                    index=GEMINI_VOICES.index(default_voice),
                    key=f"gemini_cast_{r}",
                )
                cast[r] = {"voice": selected_voice}

            cast[r]["intonation"] = st.slider(
                "抑揚",
                0.0,
                2.0,
                1.0,
                0.1,
                key=f"intonation_{r}",
            )

    st.subheader("🎬 SE / BGM / 画像")
    cue_segments = [x for x in segments if x.kind == "cue"]

    if cue_segments:
        for cue in cue_segments:
            st.markdown(f"**{cue.cue_type}**：{cue.cue}")
            if cue.cue_type in ("SE", "BGM"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.link_button(
                        "YouTube Audio Library",
                        "https://www.youtube.com/audiolibrary",
                    )
                with c2:
                    st.link_button(
                        "DOVA-SYNDROME",
                        "https://dova-s.jp/",
                    )
                with c3:
                    st.link_button(
                        "効果音ラボ",
                        "https://soundeffect-lab.info/",
                    )
    else:
        st.caption("脚本にSE / BGM / 画像指示があるとここに表示されます。")

    if st.button("🎙️ 音声を生成して順番を維持して一本化", type="primary"):
        if engine.startswith("Aivis") and not styles:
            st.error("AivisSpeechに接続できません。左の「接続テスト」を実行してください。")
        elif engine.startswith("Gemini") and not gemini_key:
            st.error("Gemini API Keyを入力してください。")
        else:
            client = (
                genai.Client(api_key=gemini_key)
                if engine.startswith("Gemini") else None
            )

            generated = {}
            errors = []
            jobs = []

            # 同時発話は別話者として生成し、同じ開始時刻に重ねる。
            for segment in speech_segments:
                speakers = [
                    normalize_role(x)
                    for x in re.split(r"\s*&\s*|＆", segment.speaker)
                ]
                for local_index, r in enumerate(speakers):
                    jobs.append((segment, local_index, r))

            progress = st.progress(0.0)

            for number, (segment, local_index, r) in enumerate(jobs, 1):
                try:
                    if engine.startswith("Aivis"):
                        style = cast[r].get("style")
                        if not style:
                            raise RuntimeError(f"{role_label(r)}の声が未選択です。")
                        audio = aivis_synthesize(
                            aivis_url,
                            segment.text,
                            style["id"],
                            cast[r]["intonation"],
                        )
                    else:
                        audio = gemini_tts(
                            client,
                            segment.text,
                            cast[r]["voice"],
                            segment.direction,
                        )

                    generated[(segment.i, local_index)] = audio

                except Exception as exc:
                    errors.append(
                        f"#{segment.i} {role_label(r)}: {exc}"
                    )

                progress.progress(number / max(1, len(jobs)))

            if errors:
                st.error("音声生成に失敗しました。")
                for error in errors:
                    st.code(error)
            else:
                # タイムラインを先に作り、同時発話を同じstartに置く。
                timeline = []
                current_time = 0.0

                for segment in segments:
                    if segment.kind == "wait":
                        current_time += segment.wait
                        continue

                    if segment.kind == "cue":
                        timeline.append({
                            "kind": "cue",
                            "start": current_time,
                            "cue": segment.cue,
                            "cue_type": segment.cue_type,
                        })
                        continue

                    speakers = [
                        normalize_role(x)
                        for x in re.split(r"\s*&\s*|＆", segment.speaker)
                    ]
                    durations = [
                        wav_duration(generated[(segment.i, idx)])
                        for idx in range(len(speakers))
                    ]
                    longest = max(durations)

                    for idx, r in enumerate(speakers):
                        timeline.append({
                            "kind": "speech",
                            "index": (segment.i, idx),
                            "speaker": r,
                            "text": segment.text,
                            "start": current_time,
                            "duration": durations[idx],
                        })

                    current_time += longest + pause

                speech_end = max(
                    [
                        x["start"] + x["duration"]
                        for x in timeline
                        if x["kind"] == "speech"
                    ] + [0.1]
                )

                final_audio = silence(speech_end + 0.2)

                for item in timeline:
                    if item["kind"] == "speech":
                        final_audio = overlay_wav(
                            final_audio,
                            generated[item["index"]],
                            item["start"],
                        )

                fast_audio = speed_up(final_audio, speed)

                credits = [
                    "5分で名作 — 音声・素材クレジット",
                    "",
                    f"音声エンジン：{engine}",
                    f"完成音声速度：{speed}x",
                    "",
                ]

                for r, info in cast.items():
                    if engine.startswith("Aivis") and info.get("style"):
                        style = info["style"]
                        credits.append(
                            f"{role_label(r)}：AivisSpeech / "
                            f"{style['speaker_name']} / {style['style_name']} / "
                            f"style_id={style['id']}"
                        )
                    else:
                        credits.append(
                            f"{role_label(r)}：Google Gemini TTS / "
                            f"{info.get('voice', '')}"
                        )

                credits.extend([
                    "",
                    "SE/BGM：作品内で使用した各素材サイト・作者・ライセンス条件を別途記載してください。",
                    "青空文庫：原作を使用した場合は作品ごとの利用条件を確認してください。",
                ])

                st.session_state.final = final_audio
                st.session_state.fast = fast_audio
                st.session_state.timeline = timeline
                st.session_state.srt = build_srt(timeline, speed)
                st.session_state.credits = "\n".join(credits)

                st.success(
                    "完成しました。セリフの順番を維持し、同時発話は同じ時刻に重ねています。"
                )


if st.session_state.final:
    st.divider()
    st.subheader("🎧 完成音声")

    st.audio(st.session_state.fast, format="audio/wav")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.download_button(
            "WAV（元速度）",
            st.session_state.final,
            "master.wav",
            "audio/wav",
        )

    with c2:
        st.download_button(
            f"WAV（{speed}倍）",
            st.session_state.fast,
            "master_fast.wav",
            "audio/wav",
        )

    with c3:
        st.download_button(
            "SRT字幕",
            st.session_state.srt,
            "subtitles.srt",
            "application/x-subrip",
        )

    with c4:
        st.download_button(
            "クレジットTXT",
            st.session_state.credits,
            "credits.txt",
            "text/plain",
        )

    with st.expander("📋 タイムライン"):
        for item in st.session_state.timeline:
            if item["kind"] == "speech":
                st.write(
                    f"{item['start']:.2f}s｜"
                    f"{role_label(item['speaker'])}｜"
                    f"{item['text']}"
                )
            else:
                st.write(
                    f"{item['start']:.2f}s｜"
                    f"{item['cue_type']}｜{item['cue']}"
                )

    with st.expander("🧾 クレジット"):
        st.code(st.session_state.credits)
