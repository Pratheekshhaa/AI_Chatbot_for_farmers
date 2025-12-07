import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import tempfile
import requests
import soundfile as sf
import librosa

from PIL import Image
from gtts import gTTS
import whisper
from langchain_ollama import ChatOllama

# ===========================
# CONFIG
# ===========================

OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "llama3.1:8b"
DATA_DIR = "data"

LANG_CONFIG = {
    "English (English)": {"code": "en", "name": "English"},
    "हिंदी (Hindi)": {"code": "hi", "name": "Hindi"},
    "ಕನ್ನಡ (Kannada)": {"code": "kn", "name": "Kannada"},
}

# ===========================
# LOAD MODELS
# ===========================

@st.cache_resource
def get_llm():
    return ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_URL,
        temperature=0.3,
    )

@st.cache_resource
def get_whisper():
    return whisper.load_model("tiny")

@st.cache_resource
def load_knowledge():
    """Load RAG data (subsidy + CSVs)."""
    subsidy_path = os.path.join(DATA_DIR, "subsidy.txt")
    subsidy_text = ""
    if os.path.exists(subsidy_path):
        with open(subsidy_path, "r", encoding="utf-8") as f:
            subsidy_text = f.read()

    def load_csv(name):
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                df["__combined__"] = df.astype(str).agg(" | ".join, axis=1)
                return df
            except:
                pass
        return None

    return {
        "subsidy_text": subsidy_text,
        "crop_soil_df": load_csv("data_core.csv"),
        "mandi_df": load_csv("mandi_data_2000_r.csv"),
        "rainfall_df": load_csv("Indian Rainfall Dataset.csv"),
    }


# ===========================
# RAG UTILITIES
# ===========================

STOPWORDS = {
    "the","is","are","was","were","for","and","with","to","in","on","at","of",
    "how","what","when","where","why","which","my","your","his","her","their",
    "a","an","do","does","can","could","should","will","would","from","it"
}

def keywords(text: str):
    return [
        t.strip(" ,.!?()[]{}'\"").lower()
        for t in text.split()
        if t.lower() not in STOPWORDS
    ]

def search_text(q, textdata, top=6):
    if not textdata:
        return ""
    keys = keywords(q)
    lines = textdata.splitlines()
    scored = []
    for ln in lines:
        score = sum(1 for k in keys if k in ln.lower())
        if score > 0:
            scored.append((score, ln))
    scored.sort(reverse=True)
    return "\n".join([ln for _, ln in scored[:top]])

def search_df(q, df, label, top=6):
    if df is None:
        return ""
    keys = keywords(q)
    mask = False
    for k in keys:
        mask = mask | df["__combined__"].str.contains(k, case=False, na=False)
    hits = df[mask].head(top)
    if hits.empty:
        return ""
    return f"--- {label} ---\n" + "\n".join(hits["__combined__"].tolist())

def build_context(q):
    data = load_knowledge()
    parts = []

    sub = search_text(q, data["subsidy_text"])
    if sub:
        parts.append("Subsidy Info:\n" + sub)

    soil = search_df(q, data["crop_soil_df"], "Crop & Soil Data")
    if soil:
        parts.append(soil)

    mandi = search_df(q, data["mandi_df"], "Mandi Prices")
    if mandi:
        parts.append(mandi)

    rain = search_df(q, data["rainfall_df"], "Rainfall Data")
    if rain:
        parts.append(rain)

    return "\n\n".join(parts)


# ===========================
# WEATHER
# ===========================

def get_weather(city):
    if not city:
        return None
    try:
        geo = requests.get(
            f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
        ).json()
        if "results" not in geo or not geo["results"]:
            return None
        lat = geo["results"][0]["latitude"]
        lon = geo["results"][0]["longitude"]

        w = requests.get(
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}&hourly=temperature_2m,relativehumidity_2m,rain"
        ).json()

        return {
            "temp": w["hourly"]["temperature_2m"][0],
            "humidity": w["hourly"]["relativehumidity_2m"][0],
            "rain": w["hourly"]["rain"][0],
        }
    except:
        return None


def fert_reco(crop, w):
    if not w:
        return f"For {crop}, use recommended dose based on soil test."
    t, h, r = w["temp"], w["humidity"], w["rain"]
    if r > 0:
        return "Rain expected → avoid fertilizer today."
    if t > 32 and h > 70:
        return "Hot & humid → reduce nitrogen; high disease risk."
    return "Weather OK → apply normal recommended fertilizer dose."


# ===========================
# VOICE TRANSCRIPTION — NO FFMPEG
# ===========================

def transcribe(audio_bytes):
    """
    Convert microphone audio → numpy → 16kHz mono → Whisper.
    No FFmpeg required.
    """
    model = get_whisper()

    # Load raw audio bytes with soundfile
    audio_np, sr = sf.read(io.BytesIO(audio_bytes))

    # Convert stereo → mono
    if audio_np.ndim > 1:
        audio_np = np.mean(audio_np, axis=1)

    # Resample to 16kHz for Whisper
    if sr != 16000:
        audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=16000)

    audio_np = audio_np.astype(np.float32)

    result = model.transcribe(audio_np)
    return result.get("text", "")


# ===========================
# LEAF ANALYSIS
# ===========================

def leaf_basic(file):
    try:
        img = Image.open(file).convert("RGB")
        arr = np.array(img)
        r, g, b = arr.mean(axis=(0, 1))

        if g > r + 10 and g > b + 10:
            status = "Leaf looks healthy green."
        elif r > g:
            status = "Leaf is yellow/brown → nutrient issue or disease."
        else:
            status = "Leaf shows mixed stress."

        return f"{status}\nRGB avg: {int(r)}, {int(g)}, {int(b)}"
    except Exception as e:
        return f"Leaf analysis error: {e}"


# ===========================
# ANSWER GENERATION
# ===========================

def answer_multilingual(q, city, crop, lang_name):
    llm = get_llm()

    w = get_weather(city)
    context = ""
    if w:
        context += f"Weather in {city}: {w['temp']}°C, humidity {w['humidity']}%, rain {w['rain']} mm.\n"
    context += "Fertilizer advice: " + fert_reco(crop, w) + "\n\n"
    context += build_context(q)

    prompt = (
        "You are 'Farmer Friend', an Indian agriculture expert.\n"
        f"Your response language must be ONLY: {lang_name}.\n"
        "- If Hindi → use Devanagari script.\n"
        "- If Kannada → use Kannada script.\n"
        "- If English → use English.\n"
        "Never mix languages.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{q}\n\n"
        f"ANSWER in {lang_name}:\n"
    )

    resp = llm.invoke(prompt)
    return resp.content.strip()


# ===========================
# STREAMLIT UI
# ===========================

def main():
    st.set_page_config(page_title="Farmer Friend", page_icon="🌾", layout="wide")
    st.title("🌾 Farmer Friend – Multilingual Smart Agriculture Assistant")

    st.sidebar.header("Settings")

    lang_label = st.sidebar.selectbox("Language", list(LANG_CONFIG.keys()))
    lang_code = LANG_CONFIG[lang_label]["code"]
    lang_name = LANG_CONFIG[lang_label]["name"]

    city = st.sidebar.text_input("City")
    crop = st.sidebar.text_input("Crop")

    tab1, tab2 = st.tabs(["💬 Chat / Voice", "🌿 Leaf Analysis"])

    # -----------------------------------
    # TAB 1 — CHAT + VOICE
    # -----------------------------------
    with tab1:

        st.subheader(f"Ask your question ({lang_label})")

        st.markdown("### 🎤 Voice Input")
        mic = st.audio_input("Click the mic and speak...")

        text_in = st.text_area("Or type here:")

        if st.button("Get Advice"):
            if mic is not None:
                st.info("🎧 Transcribing voice...")
                audio_bytes = mic.getvalue()
                text_in = transcribe(audio_bytes)

            if not text_in.strip():
                st.warning("Please type or speak something.")
                st.stop()

            with st.spinner("Thinking like a pro farmer..."):
                ans = answer_multilingual(text_in, city, crop, lang_name)

            st.markdown("### 📝 Answer:")
            st.write(ans)

            try:
                tts = gTTS(ans, lang=lang_code)
                buf = io.BytesIO()
                tts.write_to_fp(buf)
                buf.seek(0)
                st.audio(buf, format="audio/mp3")
            except:
                st.error("TTS audio failed.")

    # -----------------------------------
    # TAB 2 — LEAF ANALYSIS
    # -----------------------------------
    with tab2:
        st.subheader("Leaf Image Analysis")
        img = st.file_uploader("Upload leaf image", type=["jpg","png","jpeg"])

        if img:
            st.image(Image.open(img), width=450)
            if st.button("Analyze Leaf"):
                basic = leaf_basic(img)
                st.markdown("### Basic Analysis")
                st.write(basic)

                llm = get_llm()
                detail = llm.invoke(
                    f"You are a crop disease expert.\nLeaf status: {basic}\nExplain in {lang_name}:"
                ).content
                st.markdown("### Expert Insight")
                st.write(detail)


if __name__ == "__main__":
    main()
