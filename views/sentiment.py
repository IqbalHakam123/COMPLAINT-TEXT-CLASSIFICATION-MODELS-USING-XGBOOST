import streamlit as st
import joblib
import numpy as np
import re
import json
import os
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.sparse import hstack
from streamlit_extras.colored_header import colored_header

from source.booster_utils import load_booster_config, is_booster_aduan as token_regex_booster

BOOSTER_JSON_PATH = "/Users/iqbal/Documents/deploy-TA-Streamlit/source/booster.json"
BOOSTER_CONFIG_PATH = "/Users/iqbal/Documents/deploy-TA-Streamlit/source/booster_config.json"
VECT_PATH = "/Users/iqbal/Documents/deploy-TA-Streamlit/model/vectorizer-fiks-1.pkl"
XGB_PATH = "/Users/iqbal/Documents/deploy-TA-Streamlit/model/xgboost_model-fiks-1.pkl"
KEYW_PATH = "/Users/iqbal/Documents/deploy-TA-Streamlit/model/aduan_keywords-1.npy"
SLANG_PATH = "/Users/iqbal/Documents/deploy-TA-Streamlit/source/slang-kamus.txt"
SENTI_PATH = "/Users/iqbal/Documents/deploy-TA-Streamlit/source/sentiwords_id.txt"
THRESHOLD = 0.4

@st.cache_data
def load_slang_dict():
    d = {}
    try:
        with open(SLANG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    k, v = line.strip().split(":")
                    d[k] = v
    except:
        pass
    return d

@st.cache_data
def load_senti_dict():
    d = {}
    try:
        with open(SENTI_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    k, v = line.strip().split(":")
                    d[k] = float(v)
    except:
        pass
    return d

@st.cache_data
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ========== NORMALIZATION ==========
def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())

def normalize_slang(text: str, slang_dict) -> str:
    return " ".join(slang_dict.get(w, w) for w in word_tokenize(text.lower()))

def stem_and_check_affix(word: str, stemmer, aduan_keywords):
    cnt = 0
    stem = stemmer.stem(word)
    if stem in aduan_keywords:
        cnt += 1
    return cnt, stem

def preprocess_text(text: str, slang_dict, stemmer, stop_words, aduan_keywords):
    t = normalize_slang(text, slang_dict)
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\d+", "", t)
    tokens = [w for w in word_tokenize(t) if w not in stop_words]
    cleaned = []
    score = 0
    for w in tokens:
        c, s = stem_and_check_affix(w, stemmer, aduan_keywords)
        cleaned.append(s)
        score += c
    return " ".join(cleaned), score

def is_keyword_aduan(text: str, aduan_keywords) -> int:
    return int(bool(set(word_tokenize(text.lower())) & aduan_keywords))

# ========== PREDIC==========
def predict_aduan(
    text: str,
    booster_aduan_patterns,           # dari booster.json
    booster_bukan_aduan_patterns,     # dari booster.json
    token_pairs,                      # dari booster_config.json
    regex_patterns,                   # dari booster_config.json
    vectorizer,
    xgb_model,
    aduan_keywords,
    slang_dict,
    stemmer,
    stop_words,
    THRESHOLD=0.4,
):
    for pat in booster_aduan_patterns:
        if re.search(pat, text.lower()):
            return 1, 1.0, 0, text, f"[AduanBooster: {pat}]"

    ok, note = token_regex_booster(text, token_pairs, regex_patterns)
    if ok:
        return 1, 1.0, 0, text, note

    txt_clean, senti_score = preprocess_text(text, slang_dict, stemmer, stop_words, aduan_keywords)
    Xv = vectorizer.transform([txt_clean])
    Xk = np.array([is_keyword_aduan(txt_clean, aduan_keywords)]).reshape(-1, 1)
    X = hstack([Xv, Xk])
    prob = xgb_model.predict_proba(X)[0, 1]
    lbl = int(prob >= THRESHOLD)
    if lbl == 1:
        return lbl, prob, senti_score, txt_clean, "ML-Prediksi"

    for pat in booster_bukan_aduan_patterns:
        if re.search(pat, text.lower()):
            return 0, prob, senti_score, txt_clean, f"[BukanAduanBooster: {pat}]"

    return lbl, prob, senti_score, txt_clean, "ML-Prediksi"

# ========== MAIN STREAMLIT APP ==========
def main():
    # -- LOAD ASSET & CONFIG --
    booster_cfg = load_json(BOOSTER_JSON_PATH)
    booster_aduan_patterns = booster_cfg["booster_aduan"]
    booster_bukan_aduan_patterns = booster_cfg["booster_bukan_aduan"]

    token_pairs, regex_patterns = load_booster_config(BOOSTER_CONFIG_PATH)

    vectorizer = joblib.load(VECT_PATH)
    xgb_model = joblib.load(XGB_PATH)
    aduan_keywords = set(np.load(KEYW_PATH, allow_pickle=True))
    slang_dict = load_slang_dict()
    senti_dict = load_senti_dict()
    senti_indo = SentimentIntensityAnalyzer()
    senti_indo.lexicon.update(senti_dict)
    stemmer = StemmerFactory().create_stemmer()
    stop_words = set(stopwords.words("indonesian")) | set(stopwords.words("english"))
    stop_words.update(["iya"])

    colored_header(
        label="🕵️ Sampaikan Aduan Anda",
        description="Sistem ini akan mendeteksi apakah teks Anda merupakan aduan atau bukan.",
        color_name="violet-70"
    )

    user_input = st.text_area(
        "📝 Masukkan teks aduan:",
        height=150, placeholder="Ketik aduan Anda di sini…"
    )

    if st.button("🔍 Prediksi"):
        if not user_input.strip():
            st.warning("⚠️ Silakan masukkan teks terlebih dahulu.")
        else:
            pred, prob, senti_score, txt_cleaned, info = predict_aduan(
                user_input, 
                booster_aduan_patterns, booster_bukan_aduan_patterns,
                token_pairs, regex_patterns,
                vectorizer, xgb_model, aduan_keywords, slang_dict,
                stemmer, stop_words, THRESHOLD
            )
            st.markdown("---")
            st.subheader("📊 Hasil Prediksi:")
            if pred == 1:
                st.success("✅ Aduan Terdeteksi!")
                st.toast("👍 Aduan berhasil diidentifikasi!", icon="📣")
                st.balloons()
            else:
                st.error("❌ Bukan Aduan")
                st.toast("ℹ️ Teks tidak terdeteksi sebagai aduan.", icon="💡")

    if st.button("⬅️ Kembali ke Beranda"):
        st.query_params["page"] = "about"
        st.rerun()

if __name__ == "__main__":
    main()