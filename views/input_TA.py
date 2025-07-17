import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import json
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.sparse import hstack
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

for resource in ['stopwords', 'punkt', 'wordnet']:
    nltk.download(resource)

def app():
    vectorizer = joblib.load("model/vectorizer-fiks-1.pkl")
    xgb_model  = joblib.load("model/xgboost_model-fiks-1.pkl")
    aduan_keywords = set(np.load(
        "model/aduan_keywords-1.npy",
        allow_pickle=True
    ))
    with open("source/booster.json", "r", encoding="utf-8") as f:
        boost = json.load(f)
    booster_aduan = boost["booster_aduan"]
    booster_bukan = boost["booster_bukan_aduan"]

    THRESHOLD = 0.4

    @st.cache_data
    def load_slang_dict():
        slang = {}
        try:
            with open("source/slang-kamus.txt", "r", encoding="utf-8") as f:
                for line in f:
                    k, v = line.strip().split(":")
                    slang[k] = v
        except:
            pass
        return slang

    @st.cache_data
    def load_senti_dict():
        senti = {}
        try:
            with open("source/sentiwords_id.txt", "r", encoding="utf-8") as f:
                for line in f:
                    k, v = line.strip().split(":")
                    senti[k] = float(v)
        except:
            pass
        return senti

    slang_dict = load_slang_dict()
    senti_dict = load_senti_dict()
    senti_indo  = SentimentIntensityAnalyzer()
    senti_indo.lexicon.update(senti_dict)
    stemmer = StemmerFactory().create_stemmer()
    stop_words = set(stopwords.words('indonesian')) | set(stopwords.words('english'))
    stop_words.update(["iya"])

    def normalize_text(text: str) -> str:
        return re.sub(r'\s+', ' ', text.lower().strip())

    def normalize_slang(text: str) -> str:
        return " ".join(slang_dict.get(w, w) for w in word_tokenize(text))

    def preprocess_text(text: str) -> str:
        t = text.lower()
        t = normalize_slang(t)
        t = re.sub(r'[^\w\s]', '', t)
        t = re.sub(r'\d+', '', t)
        tokens = [w for w in word_tokenize(t) if w not in stop_words]
        return " ".join(tokens)

    def is_keyword_aduan(tclean: str) -> int:
        return int(bool(set(word_tokenize(tclean)) & aduan_keywords))

    # ====== BOOSTER FUNCTIONS ======
    def is_booster_aduan(text: str):
        t = normalize_text(text)
        for pat in booster_aduan:
            if re.search(pat, t):
                return True, f"[Booster Aduan: {pat}]"
        return False, None

    def is_booster_bukan(text: str):
        t = normalize_text(text)
        for pat in booster_bukan:
            if re.search(pat, t):
                return True, f"[Booster Bukan: {pat}]"
        return False, None

    def predict_aduan(text: str):
        ok, note = is_booster_aduan(text)
        if ok:
            return 1, 1.0, note
        ok, note = is_booster_bukan(text)
        if ok:
            return 0, 0.0, note
        clean = preprocess_text(text)
        X_vec = vectorizer.transform([clean])
        X_key = np.array([is_keyword_aduan(clean)]).reshape(-1,1)
        X     = hstack([X_vec, X_key])
        prob  = xgb_model.predict_proba(X)[0,1]
        lbl   = int(prob >= THRESHOLD)
        return lbl, prob, "ML-Prediksi"

    # ====== METRIC CARD ======
    def metric_card(title, value, delta=None, color="#316398"):
        st.markdown(f"""
            <div style='
                background:{color}22;
                border:2px solid {color};
                border-radius:12px;
                padding:1.2em;
                text-align:center;
                box-shadow:0 3px 10px #00000010;
                margin-bottom:6px;
            '>
                <div style='font-size:16px;font-weight:600;opacity:0.8'>{title}</div>
                <div style='font-size:32px;font-weight:700;color:{color}'>{value}</div>
                {f"<div style='color:grey;font-size:13px'>{delta}</div>" if delta else ""}
            </div>
        """, unsafe_allow_html=True)

    # ====== HEADER ======
    st.markdown("""
        <style>
        .big-title {font-size:34px; font-weight:700; color:#0A2342; margin-bottom:6px;}
        .highlight {background-color:#D0E2FF; padding:2px 6px; border-radius:4px;}
        </style>
    """, unsafe_allow_html=True)
    st.markdown("<div class='big-title'>🚦 Analisis Pengaduan Transportasi Surabaya</div>",
                unsafe_allow_html=True)
    st.caption(
        "📊 Unggah CSV Anda untuk memunculkan dashboard sederhana dan insight aduan secara instan!",
        unsafe_allow_html=True
    )

    # ====== UPLOAD & PROCESS ======
    up = st.file_uploader("📂 Pilih file CSV...", type=["csv"])
    if up:
        df = pd.read_csv(up)
        if "text" not in df.columns:
            st.error("Kolom `text` tidak ditemukan.")
            return

        with st.spinner("🔎 Memproses..."):
            def apply_pred(r):
                lbl, pr, note = predict_aduan(r["text"])
                return pd.Series({
                    "text_cleaned": preprocess_text(r["text"]),
                    "prob_aduan":  pr,
                    "label":       "aduan" if lbl==1 else "bukan aduan",
                    "rule_note":   note
                })
            df[["text_cleaned","prob_aduan","label","rule_note"]] = df.apply(apply_pred, axis=1)
        st.success("✅ Klasifikasi selesai!")

        # ====== METRICS ======
        n1 = (df.label=="aduan").sum()
        n2 = (df.label=="bukan aduan").sum()
        tot = len(df)
        aw  = int(df.text.str.split().str.len().mean())

        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("Total Data", tot, color="#316398")
        with c2: metric_card("Aduan", n1, color="#DE4C4A")
        with c3: metric_card("Bukan Aduan", n2, color="#48A14D")
        with c4: metric_card("Rata-rata Kata", aw, color="#316398")

        # ====== DISTRIBUSI & CONTOH KALIMAT BERSEBELAHAN ======
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Distribusi Aduan vs Bukan")
            fig, ax = plt.subplots(figsize=(3, 3))
            wedges, _, autotxt = ax.pie(
                [n1, n2],
                labels=["Aduan", "Bukan"],
                autopct="%1.1f%%",
                startangle=120,
                colors=["#DE4C4A", "#A1D48F"],
                explode=(0.05, 0.05),
                wedgeprops={"edgecolor": "#fff", "linewidth": 1.5},
                textprops={"fontsize": 12, "weight": "bold"}
            )
            for t in autotxt:
                t.set_color("#333")
            ax.add_artist(plt.Circle((0, 0), 0.65, fc="white", linewidth=1.2, edgecolor="#ececec"))
            ax.axis("equal")
            st.pyplot(fig)


        with col2:
            st.markdown("### 🧾 Contoh Kalimat", unsafe_allow_html=True)
            tab1, tab2 = st.tabs(["Aduan", "Bukan Aduan"])
            with tab1:
                df_aduan = df[df.label=="aduan"][["text","label"]]
                df_aduan.columns = ["Contoh Kalimat","Label"]
                st.dataframe(df_aduan, height=250, use_container_width=True, hide_index=True)
            with tab2:
                df_bukan = df[df.label=="bukan aduan"][["text","label"]]
                df_bukan.columns = ["Contoh Kalimat","Label"]
                st.dataframe(df_bukan, height=250, use_container_width=True, hide_index=True)

        # ====== WORDCLOUD ======
        st.markdown("### ☁️ WordCloud", unsafe_allow_html=True)
        wc1, wc2 = st.columns(2)
        with wc1:
            st.write("Aduan")
            ad_text = " ".join(df[df.label=="aduan"]["text_cleaned"])
            if ad_text:
                wc = WordCloud(width=400, height=200, background_color='white')\
                    .generate(ad_text)
                fig, ax = plt.subplots(figsize=(6,3))
                ax.imshow(wc); ax.axis("off")
                st.pyplot(fig)
        with wc2:
            st.write("Bukan Aduan")
            nonad_text = " ".join(df[df.label=="bukan aduan"]["text_cleaned"])
            if nonad_text:
                wc = WordCloud(width=400, height=200, background_color='white')\
                    .generate(nonad_text)
                fig, ax = plt.subplots(figsize=(6,3))
                ax.imshow(wc); ax.axis("off")
                st.pyplot(fig)

        # ====== DUPLIKAT ======
        st.markdown("### 🔁 Duplikat", unsafe_allow_html=True)
        dup = (
            df.groupby('text_cleaned')
                .agg(Text_Asli=('text','first'),
                    Label=('label','first'),
                    Jumlah_Duplikat=('text','count'))
                .query('Jumlah_Duplikat>1')
                .reset_index()
                .rename(columns={'text_cleaned':'Contoh Kalimat'})
        )
        if not dup.empty:
            st.warning(f"🚨 {len(dup)} kalimat berulang", icon="♻️")
            def style_lbl(v):
                c = '#48A14D' if v=='aduan' else '#DE4C4A'
                return f'background:{c}22; color:{c}; font-weight:600;'
            st.dataframe(dup.style.applymap(style_lbl, subset=['Label']),
                            use_container_width=True, hide_index=True)
        else:
            st.info("Tidak ada duplikat.")

        # ====== DOWNLOAD ======
        st.markdown("### ⬇️ Unduh Hasil", unsafe_allow_html=True)
        download_df = df[["text", "text_cleaned", "label"]]
        csv_data = download_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="hasil_sederhana.csv",
            mime="text/csv"
        )

    else:
        st.info(
            "📂 Pastikan file CSV Anda memiliki:\n"
            "- Kolom `text` (berisi kalimat yang akan diproses)\n"
            "- Kolom `label` (biarkan kosong)\n"
        )

    if st.button("⬅️ Kembali ke Beranda"):
        st.query_params["page"] = "about"
        st.rerun()
