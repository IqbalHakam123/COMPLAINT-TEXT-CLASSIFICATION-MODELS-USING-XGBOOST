import streamlit as st
import importlib.util
import os

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Aduan Masyarakat Transportasi Surabaya",
    page_icon="🚦"
)

# ─── Read query‐params ─────────────────────────────────────────────────────────
query_params = st.query_params
page = query_params.get("page", "about")

# ─── GLOBAL CSS (with responsive media query) ─────────────────────────────────
st.markdown("""
    <style>
    /* Title */
    .adn-title {
        font-size: 2.7em;
        font-weight: 700;
        color: #2C3E50;
        margin-bottom: 1rem;
        line-height: 1.1;
        letter-spacing: -1px;
        animation: fadeInTitle 1s ease-out both;
    }
    @keyframes fadeInTitle {
      from { opacity: 0; transform: translateY(-10px); }
      to { opacity: 1; transform: translateY(0); }
    }

    /* Highlighted words */
    .highlight {
        color: #3498DB !important;
        font-size: 1.1em;
        font-weight: 600;
        letter-spacing: -0.3px;
    }

    /* Description text */
    .adn-desc {
        font-size: 1.15em;
        color: #566573;
        margin-bottom: 2rem;
        animation: fadeInDesc 1s .3s both;
    }
    @keyframes fadeInDesc {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }

    /* Button container */
    .adn-btns {
        display: flex;
        flex-wrap: wrap;
        gap: 1.5rem;
        justify-content: center;
        margin: 2rem 0;
    }

    /* Buttons */
    .adn-btn {
        flex: 1 1 auto;
        min-width: 140px;
        max-width: 220px;
        height: 48px;
        border-radius: 8px;
        background: #ECF0F1;
        border: 1px solid #BDC3C7;
        color: #2C3E50;
        font-size: 1em;
        font-weight: 600;
        transition: background .2s, transform .2s, box-shadow .2s;
        display: flex; align-items: center; justify-content: center;
        cursor: pointer;
    }
    .adn-btn:hover {
        background: #BDC3C7;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }

    /* Media query for narrow (mobile) screens */
    @media (max-width: 600px) {
      /* Stack the two Streamlit columns */
      .stColumns {
        flex-direction: column !important;
      }
      /* Center the image in the .mobile-img wrapper */
      .mobile-img img {
        display: block !important;
        margin: 0 auto !important;
      }
      /* Keep buttons centered */
      .adn-btns {
        justify-content: center !important;
      }
    }
    </style>
""", unsafe_allow_html=True)

# ─── Helper to load other pages (views/*.py) ────────────────────────────────────
def load_page_from_views(page_file):
    page_path = os.path.join("views", page_file)
    if os.path.exists(page_path):
        spec = importlib.util.spec_from_file_location("viewpage", page_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, "app"):
            module.app()
        elif hasattr(module, "main"):
            module.main()
        else:
            st.warning(f"Module '{page_file}' has no app()/main().")
    else:
        st.error(f"File not found: views/{page_file}")

# ─── MULTI‐PAGE LOGIC ──────────────────────────────────────────────────────────
if page == "about":
    # ─── Loading overlay (first 2s) ─────────────────────────────────────────
    st.markdown("""
    <style>
      .loader-wrapper {
        position: fixed; top:0; left:0; width:100vw; height:100vh;
        background:#fff; display:flex; flex-direction:column;
        align-items:center; justify-content:center; z-index:9999;
        /* fade out after 5s */
        animation: fadeOut 0.5s ease-out 5s forwards;
      }
      .loader {
        border:12px solid #f3f3f3; border-top:12px solid #316398;
        border-radius:50%; width:100px; height:100px;
        animation: spin 1s linear infinite;
      }
      .loader-text {
        margin-top:1.2rem;
        font-size:1.4rem; color:#316398;
        font-weight:700; letter-spacing:1px;
        /* make room for the text width */
        width:16ch;           /* adjust to text length */
        white-space:nowrap;
        overflow:hidden;
        border-right:2px solid #316398;
        /* type, then blink, loop until fade-out */
        animation:
          typing 3s steps(16) infinite,
          blink 0.8s step-end infinite;
      }

      @keyframes spin {
        0%   { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
      }
      @keyframes fadeOut {
        to { opacity: 0; visibility: hidden; }
      }
      @keyframes typing {
        from { width: 0; }
        to   { width: 16ch; }
      }
      @keyframes blink {
        50% { border-color: transparent; }
      }
    </style>
    <div class="loader-wrapper">
      <div class="loader"></div>
      <div class="loader-text">Memuat Website </div>
    </div>
""", unsafe_allow_html=True)


    # ─── Main about layout ────────────────────────────────────────────────────
    col1, col2 = st.columns([1.7, 2])
    # add some vertical spacing
    for _ in range(4):
        col1.write("")
        col2.write("")

    # left column: image wrapped in .mobile-img for centering on mobile
    with col1:
        st.markdown('<div class="mobile-img">', unsafe_allow_html=True)
        st.image("/images/www.jpg", width=800, use_container_width=False)
        st.markdown('</div>', unsafe_allow_html=True)

    # right column: title, description, buttons
    with col2:
        st.markdown("""
            <div class="adn-title">
                Aduan Masyarakat<br>
                <span class="highlight">Transportasi & Lalu Lintas</span>
            </div>
            <div class="adn-desc">
                Sampaikan aduan, pelanggaran, atau kendala lalu lintas.<br>
                Sistem ini siap mendeteksi dan mengklasifikasikan setiap laporan Anda secara otomatis, berbasis AI.<br>
                <strong style="color:#2C3E50;">
                    Bersuara untuk perubahan, bersama tingkatkan kualitas perjalanan
                </strong>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="adn-btns">
            <form action="" method="get" style="margin:0;">
                <button name="page" value="sentiment" class="adn-btn" type="submit">
                    🔎 Coba Deteksi
                </button>
            </form>
            <form action="" method="get" style="margin:0;">
                <button name="page" value="input" class="adn-btn" type="submit">
                    📥 Input Data
                </button>
            </form>
            <form action="" method="get" style="margin:0;">
                <button name="page" value="about_penulis" class="adn-btn" type="submit">
                    👨🏻‍💻 Tentang Penelitian
                </button>
            </form>
        </div>
        """, unsafe_allow_html=True)

elif page == "sentiment":
    load_page_from_views("sentiment.py")

elif page == "input":
    load_page_from_views("input_TA.py")

elif page == "about_penulis":
    load_page_from_views("about_TA.py")

# ─── GLOBAL FOOTER ─────────────────────────────────────────────────────────────
st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.3/font/bootstrap-icons.css">
<style>
.footer-wrap {
    width: 100%; padding: 1.5rem 0; background: #F4F6F7; border-top: 1px solid #D5DBDB;
    text-align: center; margin-top: 3rem;
}
.footer-links { display: inline-flex; gap: 1.25rem; font-size: 1.3rem; margin-bottom: 0.5rem; }
.footer-links a { color: #2C3E50; transition: color .2s; }
.footer-links a:hover { color: #3498DB; }
.footer-text { font-size: 0.85rem; color: #566573; }
</style>
<div class="footer-wrap">
    <div class="footer-links">
        <a href="https://www.linkedin.com/in/iqbal-hakam-495120296" target="_blank"><i class="bi bi-linkedin"></i></a>
        <a href="https://www.instagram.com/ibaaal08" target="_blank"><i class="bi bi-instagram"></i></a>
    </div>
    <div class="footer-text">
        © 2025 Iqbal Hakam | Sistem Informasi, Telkom University<br>
        Dibangun dengan ❤️ menggunakan Python & Streamlit
    </div>
</div>
""", unsafe_allow_html=True)
