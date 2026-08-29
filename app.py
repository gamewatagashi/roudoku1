
import io, json, base64, wave, re, array
import streamlit as st
from google import genai

st.set_page_config(page_title="5分で名作", page_icon="📖")
VOICES = ["Zephyr","Puck","Charon","Kore","Fenrir","Leda","Orus","Aoede","Callirrhoe","Autonoe","Enceladus","Iapetus","Umbriel","Algieba","Despina","Erinome","Algenib","Rasalgethi","Laomedeia","Achernar","Alnilam","Schedar","Gacrux","Pulcherrima","Achird","Zubenelgenubi","Vindemiatrix","Sadachbia","Sadaltager","Sulafat"]
DEFAULT = {"ナレーター":"Rasalgethi","メロス":"Iapetus","王":"Algieba","セリヌンティウス":"Alnilam","妹":"Aoede","フィロストラトス":"Schedar"}

def dur(b):
    with wave.open(io.BytesIO(b),"rb") as w: return w.getnframes()/w.getframerate()

def join_wav(parts):
    if not parts: return b""
    bio=io.BytesIO()
    with wave.open(io.BytesIO(parts[0]),"rb") as w: p=w.getparams()
    with wave.open(bio,"wb") as out:
        out.setparams(p)
        for b in parts:
            with wave.open(io.BytesIO(b),"rb") as w: out.writeframes(w.readframes(w.getnframes()))
    return bio.getvalue()

def speed(b,f=1.5):
    with wave.open(io.BytesIO(b),"rb") as w:
        p=w.getparams(); raw=w.readframes(w.getnframes())
    if p.sampwidth != 2: return b
    x=array.array("h"); x.frombytes(raw)
    frames=len(x)//p.nchannels; y=array.array("h"); i=0.0
    while int(i)<frames:
        k=int(i)*p.nchannels; y.extend(x[k:k+p.nchannels]); i+=f
    o=io.BytesIO()
    with wave.open(o,"wb") as w:
        w.setnchannels(p.nchannels); w.setsampwidth(2); w.setframerate(p.framerate); w.writeframes(y.tobytes())
    return o.getvalue()

def stime(s):
    ms=int(round(s*1000)); h,ms=divmod(ms,3600000); m,ms=divmod(ms,60000); sec,ms=divmod(ms,1000)
    return f"{h:02}:{m:02}:{sec:02},{ms:03}"

def make_srt(segs):
    t=0; out=[]
    for i,s in enumerate(segs,1):
        d=s["duration"]
        out.append(f"{i}\n{stime(t)} --> {stime(t+d)}\n{s['speaker']}: {s['text']}\n")
        t+=d
    return "\n".join(out)

def tts(client,text,voice):
    prompt = (
        "Synthesize ONLY the Japanese spoken text after the marker. "
        "Use expressive, natural, fast-paced audiobook narration. "
        "Do not read instructions aloud.\n=== SPOKEN TEXT ===\n" + text
    )
    r=client.interactions.create(
        model="gemini-3.1-flash-tts-preview",
        input=prompt,
        response_format={"type":"audio"},
        generation_config={"speech_config":[{"voice":voice}]})
    raw=r.output_audio.data
    return base64.b64decode(raw) if isinstance(raw,str) else raw

def make_script(client,src,minutes,roles):
    prompt = (
        "You are a Japanese YouTube scriptwriter for the channel '5分で名作'.\n"
        f"Rewrite the supplied public-domain story as a FAST, IMPACTFUL STORY, not an explanation.\n"
        f"Target {minutes} minutes before the editor's later 1.5x speed-up. "
        "Keep only essential plot, conflict and emotional payoff.\n"
        "Use narration for prose and character dialogue for characters. "
        f"Use only these speaker names: {roles}.\n"
        "Return JSON only with this schema:\n"
        '{"title":"...","segments":[{"speaker":"ナレーター","text":"..."},{"speaker":"キャラクター名","text":"..."}],'
        '"image_instructions":[{"time_hint":"...","visual":"..."}],'
        '"se_instructions":[{"time_hint":"...","sound":"...","purpose":"..."}],'
        '"bgm_instructions":[{"time_hint":"...","mood":"...","purpose":"..."}]}\n'
        "SOURCE:\n" + src
    )
    r=client.interactions.create(model="gemini-3.6-flash",input=prompt)
    m=re.search(r"\{.*\}",r.output_text,re.S)
    if not m: raise RuntimeError("脚本JSONを取得できませんでした")
    return json.loads(m.group())

if "cast" not in st.session_state: st.session_state.cast=DEFAULT.copy()

st.title("📖 5分で名作・自動音声制作")
st.caption("原作 → 脚本 → キャラ別TTS → 結合 → 1.5倍速 → SRT")

with st.sidebar:
    key=st.text_input("Gemini API Key",type="password")
    st.caption("APIキーはチャットには貼らず、この画面にだけ入力してください。")
    st.subheader("🎙 キャスト")
    for role in list(st.session_state.cast):
        st.session_state.cast[role]=st.selectbox(
            role,VOICES,index=VOICES.index(st.session_state.cast[role]),key="v_"+role)
    if st.button("＋キャラクター追加"):
        n=1
        while f"キャラクター{n}" in st.session_state.cast: n+=1
        st.session_state.cast[f"キャラクター{n}"]="Kore"
        st.rerun()

up=st.file_uploader("原作 TXT / MD",type=["txt","md"])
src=up.read().decode("utf-8","replace") if up else st.text_area("または本文を貼り付け",height=180)
minutes=st.radio("尺",[1,5,10],index=1,horizontal=True)

if st.button("🚀 生成開始",type="primary",disabled=not(key and src.strip())):
    try:
        c=genai.Client(api_key=key)
        with st.status("制作中…",expanded=True):
            st.write("脚本を生成中…")
            result=make_script(c,src,minutes,list(st.session_state.cast))
            segs=result["segments"]; aud=[]; timed=[]; bar=st.progress(0)
            for i,s in enumerate(segs):
                speaker=s["speaker"] if s["speaker"] in st.session_state.cast else "ナレーター"
                s["speaker"]=speaker
                a=tts(c,s["text"],st.session_state.cast[speaker])
                aud.append(a); timed.append({"speaker":speaker,"text":s["text"],"duration":dur(a)})
                bar.progress((i+1)/len(segs))
            original=join_wav(aud); fast=speed(original,1.5)
            result["timed_segments"]=timed
            st.session_state.result=result
            st.session_state.original=original
            st.session_state.fast=fast
            st.session_state.srt=make_srt(timed)
        st.success("完成！")
    except Exception as e:
        st.error(str(e))

if "result" in st.session_state:
    r=st.session_state.result
    st.header(r.get("title","完成"))
    a,b=st.columns(2)
    with a:
        st.write("元音声"); st.audio(st.session_state.original,"audio/wav")
        st.download_button("元音声 WAV",st.session_state.original,"audio_original.wav","audio/wav")
    with b:
        st.write("1.5倍速版"); st.audio(st.session_state.fast,"audio/wav")
        st.download_button("1.5倍速 WAV",st.session_state.fast,"audio_1.5x.wav","audio/wav")
    st.download_button("SRT字幕",st.session_state.srt,"subtitles.srt","application/x-subrip")
    notes={"images":r.get("image_instructions",[]),"SE":r.get("se_instructions",[]),"BGM":r.get("bgm_instructions",[])}
    st.download_button("画像・SE・BGM指示 JSON",json.dumps(notes,ensure_ascii=False,indent=2),"production_notes.json","application/json")
    with st.expander("脚本"):
        for s in r["segments"]: st.markdown(f"**{s['speaker']}**　{s['text']}")
