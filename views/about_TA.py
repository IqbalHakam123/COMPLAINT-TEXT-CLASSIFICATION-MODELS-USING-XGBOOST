import streamlit as st
import pandas as pd

def app():

    COLOR_PRIMARY = "#45607c"   # Biru navy muted
    COLOR_BG = "#f8fafc"        # Abu putih modern
    COLOR_BORDER = "#d3dbe6"    # Abu kebiruan
    COLOR_TEXT = "#233143"      # Abu tua kebiruan

    st.markdown(f"""
    <style>
    body {{ background: {COLOR_BG}; }}
    .headline-prof {{
        text-align:center;
        background: linear-gradient(90deg, #eaf1fa 10%, #fff 90%);
        color: {COLOR_PRIMARY};
        font-weight:900;
        font-size:2.3em;
        letter-spacing: 0.7px;
        border-radius: 22px;
        padding: 22px 26px 15px 26px;
        margin-bottom: 16px;
        border: 2px solid {COLOR_BORDER};
        box-shadow: 0 2px 14px #31639811;
    }}
    .caption-prof {{
        text-align:center;
        color: {COLOR_PRIMARY};
        font-size:1.12em;
        font-weight:600;
        margin-bottom:27px;
        letter-spacing: 0.05px;
        padding: 4px 0 0 0;
        border-radius: 7px;
    }}
    .section-title {{
        font-size:1.19em; font-weight:700; color:{COLOR_PRIMARY}; margin-top:1.5em; margin-bottom:0.85em;
        border-left:4px solid {COLOR_PRIMARY}33; padding-left:11px;
        letter-spacing:0.01em;
        text-align:left;
    }}
    .content-box {{
        background: #fafbfc;
        border: 1.3px solid {COLOR_BORDER};
        border-radius: 13px;
        padding: 1.45rem 1.18rem 1.12rem 1.18rem;
        margin-bottom: 2.15rem;
        box-shadow: 0 2px 14px #31639809;
        color: {COLOR_TEXT};
        font-size: 1.01em;
        text-align: justify;
    }}
    .highlight-key {{
        color: {COLOR_PRIMARY};
        font-weight:700;
        font-size:1.07em;
    }}
    .kontribusi-box {{
        background: #f1f5fb;
        border: 1.3px solid {COLOR_PRIMARY}22;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 2rem;
        color: #234;
        font-size: 1.02em;
        box-shadow: 0 1px 8px #31639810;
        text-align: justify;
    }}
    .stat-box {{
        background: #f1f5fb;
        border-radius:9px;
        padding:12px 19px 10px 19px;
        margin-bottom:17px;
        font-size:1.01em;
        border:1.2px solid #e0e0e0;
        color:#234;
        text-align: justify;
    }}
    .css-1m91lqg {{background: #f1f5fb;}}
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="headline-prof">
            Klasifikasi Otomatis Aduan Transportasi & Lalu Lintas berbasis AI 
        </div>
        <div class="caption-prof">
            Platform otomatisasi aduan transportasi dan lalu lintas dengan <span class="highlight-key">Machine Learning</span> modern
        </div>
    """, unsafe_allow_html=True)

    # ======= Kontribusi Penelitian =======
    st.markdown('<div class="section-title">Kontribusi Penelitian</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="content-box">
        Penelitian ini menghadirkan <b>aplikasi website cerdas dan interaktif</b> untuk <span class="highlight-key">klasifikasi otomatis</span> aduan transportasi dan lalu lintas dari media sosial menggunakan <b>XGBoost</b> dan teknik NLP Bahasa Indonesia. Model ini mempercepat dan mempermudah pengelolaan aduan, membantu <b>identifikasi pola masalah publik secara real-time</b>, serta mendorong pengambilan keputusan berbasis data. Inovasi ini juga <b>potensial diimplementasi</b> untuk pengelolaan aduan di berbagai kota dan instansi lain di Indonesia.
    </div>
    """, unsafe_allow_html=True)

    # ======= INFO STATISTIK DATASET =======
    st.markdown(f'''
    <div class="content-box">
        <b>Statistik Dataset:</b><br>
        Total tweet: <b>23.000+</b> &nbsp;|&nbsp; Periode: <b>Feb–Des 2023</b> &nbsp;|&nbsp; Sumber: <code>@e100ss</code>
    </div>
    ''', unsafe_allow_html=True)

    # ======= PENDEKATAN CRISP-DM & DESKRIPSI =======
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(
            "images/Metodologi-data-science.png",
            caption="Alur CRISP-DM dalam Penelitian",
            use_container_width=True
        )
        st.markdown('<div class="section-title">Pendekatan CRISP-DM</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="content-box">Menggunakan kerangka kerja <span class="highlight-key">CRISP-DM</span> secara iteratif: '
            'Business Understanding → Data Understanding → Data Preparation → Modeling → '
            'Evaluation → Deployment untuk memaksimalkan kualitas model klasifikasi aduan.</div>',
            unsafe_allow_html=True
        )
    with col2:
        st.markdown('<div class="section-title">Deskripsi Penelitian</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="content-box">
        <p>
        <b>Peningkatan volume aduan masyarakat di media sosial telah menciptakan kebutuhan mendesak akan model pengelolaan data yang otomatis dan responsif.</b>
        Proses manual tidak lagi memadai untuk menganalisis ribuan aduan yang masuk setiap hari, sehingga inovasi berbasis AI menjadi kebutuhan utama layanan publik modern.
        </p>
        <p>
        Penelitian ini bertujuan mengembangkan <b>model klasifikasi otomatis</b> berbasis <strong>XGBoost</strong> untuk mendeteksi dan mengelompokkan aduan transportasi & lalu lintas pada data tweet dari akun <code>@e100ss</code>. Data diperoleh melalui <em>web scraping</em> selama Februari–Desember 2023, menghasilkan lebih dari <strong>23.000</strong> tweet yang mencerminkan berbagai permasalahan masyarakat—mulai dari kemacetan, infrastruktur jalan, hingga perilaku pengguna.
        </p>
        <ul>
            <li><strong>Case Folding</strong>: ubah semua huruf ke lowercase</li>
            <li><strong>Pembersihan</strong>: hapus simbol, angka, URL, dan mention</li>
            <li><strong>Stopword Removal</strong>: buang kata-kata umum tidak bermakna</li>
            <li><strong>Tokenizing & Stemming</strong>: pecah kalimat menjadi kata dasar</li>
            <li><strong>Norm. Slang & OOV</strong>: mapping istilah informal dan kata di luar kosakata</li>
        </ul>
        <p>
        Fitur diekstraksi menggunakan <b>CountVectorizer</b>, <b>TF-IDF</b>, dan <b>Word2Vec</b> untuk merepresentasikan data secara numerik. Evaluasi model menggunakan metrik <b>accuracy</b>, <b>precision</b>, <b>recall</b>, <b>F1-score</b>, serta visualisasi <b>confusion matrix</b> dan <b>ROC Curve</b>. Kombinasi <span class="highlight-key">CountVectorizer + XGBoost</span> terbukti paling optimal—meraih akurasi 93%, precision 90%, recall 86%, dan F1-score 88%.
        </p>
        <p>
            <span style="color:#3c3f46;">
                Model terbaik akan diimplementasikan dalam aplikasi website Streamlit yang interaktif, mendukung prediksi aduan baik satuan maupun batch (CSV), serta memudahkan instansi untuk melakukan monitoring isu secara real-time. Solusi ini tak hanya meningkatkan kecepatan layanan publik, namun juga dapat diadaptasi untuk sektor dan wilayah lain—mewujudkan pengelolaan aduan yang efektif, efisien, dan berbasis data.
            </span>
        </p>
        </div>
        """, unsafe_allow_html=True)

    # ======= DATASET & EVALUASI =======
    st.markdown('<div class="section-title">Informasi Dataset & Evaluasi Model</div>', unsafe_allow_html=True)
    df = pd.read_csv("source/data-20rb.csv")
    df_display = df.drop(columns=["is_aduan", "text_clean"], errors="ignore")
    with st.expander("📊 Tampilkan Data Aduan (20.000+ tweet)", expanded=False):
        st.dataframe(df_display, use_container_width=True)
        csv = df_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Dataset (CSV)", 
            csv, "dataset.csv", mime="text/csv", 
            help="Unduh data aduan untuk eksplorasi lebih lanjut."
        )

    # ======= data tabel =======
    dummy_reports = {
        "CountVectorizer + XGBoost": pd.DataFrame({
            "": ["Bukan Aduan","Aduan","Accuracy"],
            "Precision": [0.94, 0.90, ""],
            "Recall":    [0.96, 0.86, ""],
            "F1-Score":  [0.95, 0.88, 0.93],
            "Support":   [3308, 1365, 4673]
        }),
        "TF-IDF + XGBoost": pd.DataFrame({
            "": ["Bukan Aduan","Aduan","Accuracy"],
            "Precision": [0.93, 0.92, ""],
            "Recall":    [0.97, 0.83, ""],
            "F1-Score":  [0.95, 0.87, 0.93],
            "Support":   [3308, 1365, 4673]
        }),
        "TF-IDF + Word2Vect + XGBoost": pd.DataFrame({
            "": ["Bukan Aduan","Aduan","Accuracy"],
            "Precision": [0.93, 0.88, ""],
            "Recall":    [0.95, 0.83, ""],
            "F1-Score":  [0.94, 0.86, 0.92],
            "Support":   [3308, 1365, 4673]
        }),
        "CountVectorizer + Word2Vect + XGBoost": pd.DataFrame({
            "": ["Bukan Aduan","Aduan","Accuracy"],
            "Precision": [0.93, 0.88, ""],
            "Recall":    [0.95, 0.83, ""],
            "F1-Score":  [0.94, 0.85, 92],
            "Support":   [3308, 1365, 4673]
        }),
        "TF-IDF + CountVectorizer + XGBoost": pd.DataFrame({
            "": ["Bukan Aduan","Aduan","Accuracy"],
            "Precision": [0.94, 0.88, ""],
            "Recall":    [0.95, 0.86, ""],
            "F1-Score":  [0.95, 0.87, 0.92],
            "Support":   [3308, 1365, 4673]
        }),
    }
    report_images = {
        "CountVectorizer + XGBoost":            "images/cv-fiks-gambar.png",
        "TF-IDF + XGBoost":                     "images/tfidf-xgb.png",
        "TF-IDF + Word2Vect + XGBoost":         "images/w2v+tfidf.png",
        "CountVectorizer + Word2Vect + XGBoost":"images/w2w+cv.png",
        "TF-IDF + CountVectorizer + XGBoost":   "images/tfidf-cv.png",
    }

    choice = st.selectbox("Pilih Model Evaluasi:", list(dummy_reports.keys()))
    st.markdown(f"#### {choice}", unsafe_allow_html=True)
    st.image(report_images[choice], use_container_width=True)
    st.markdown("<b>Classification Report</b>", unsafe_allow_html=True)
    df_report = dummy_reports[choice].set_index("", drop=True)
    for col in ["Precision","Recall","F1-Score"]:
        df_report[col] = df_report[col].apply(lambda x: f"{float(x):.2f}" if isinstance(x,(int,float)) else x)
    st.table(df_report)


    # ====== tombl kembali ====
    if st.button("⬅️ Kembali ke Beranda"):
        st.query_params["page"] = "about"
        st.rerun()
