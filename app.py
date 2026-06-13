import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import os
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def preprocess_text(text: str) -> str:
    """Simple preprocessing:
    - lowercase
    - remove digits
    - remove punctuation and symbols
    - collapse multiple spaces
    """
    if not isinstance(text, str):
        return ""
    s = text.lower()
    s = re.sub(r"\d+", "", s)  # remove digits
    s = re.sub(r"[^\w\s]", "", s)  # remove punctuation/symbols
    s = re.sub(r"_", " ", s)  # replace underscores with space
    s = re.sub(r"\s+", " ", s).strip()
    return s

# =====================
# KONFIGURASI HALAMAN
# =====================
st.set_page_config(
    page_title="Academic Performance Monitoring System",
    page_icon="🎓",
    layout="wide"
)

# Custom academic styling (blue & green)
st.markdown(
    """
    <style>
    /* Page background */
    .stApp { background-color: #f4f8fb; }
    /* Container */
    .block-container { background-color: #ffffff; padding: 2rem; border-radius: 8px; max-width: 1100px; margin: 1rem auto; }
    /* Title and headers */
    h1 { color: #0b3d91; font-family: 'Segoe UI', Roboto, sans-serif; }
    h2, h3 { color: #0b6b3a; font-family: 'Segoe UI', Roboto, sans-serif; }
    /* Header boxed style */
    .header-box { background: linear-gradient(90deg, #0b3d91 0%, #0b6b3a 100%); color: #fff; padding: 1.5rem 2rem; border-radius: 8px; box-shadow: 0 6px 22px rgba(11,61,145,0.12); margin: 0 auto 1rem; max-width: 1100px; text-align: center; }
    .header-box h1 { margin: 0; color: #ffffff; font-size: 2rem; font-weight: 700; }
    .header-box .subtitle { margin: 0.5rem 0 0; color: rgba(255,255,255,0.95); font-size: 1rem; line-height: 1.4; }
    /* Buttons */
    .stButton>button { background-color: #0b3d91; color: white; border-radius:6px; }
    /* Input focus */
    input:focus, textarea:focus { outline: 2px solid #0b6b3a; }
    </style>
    """,
    unsafe_allow_html=True,
)
# =====================
# LOAD MODEL
# =====================
model_akademik = joblib.load("model_akademik.pkl")
model_sentiment = joblib.load("model_sentiment.pkl")

# =====================
# (HEADER GLOBAL) 
# =====================
# Dihapus pada halaman welcome sesuai permintaan.


# =====================
# NAV / WELCOME
# =====================
if "page" not in st.session_state:
    st.session_state["page"] = "welcome"

# Initialize sentiment history in session_state
if "riwayat_sentimen" not in st.session_state:
    st.session_state["riwayat_sentimen"] = []

# =====================
# Halaman: WELCOME
# =====================
if st.session_state["page"] == "welcome":
    st.markdown("""
    <div class="header-box" style="max-width: 1200px; padding: 2.6rem 2.8rem;">
        <h1 style="margin: 0 0 0.45rem 0; font-size: 2.85rem; font-weight: 800; letter-spacing: 0.2px;">
            Selamat Datang
        </h1>
        <p class="subtitle" style="margin: 0; font-size: 1.1rem; line-height: 1.6;">
            Sistem prediksi risiko akademik mahasiswa dan analisis sentimen feedback dengan Machine Learning.
        </p>
    </div>
    """, unsafe_allow_html=True)


    st.markdown("""
    <div style="max-width: 980px; margin: 0 auto; background: #ffffff; padding: 1.5rem 2rem; border-radius: 8px; box-shadow: 0 6px 22px rgba(11,61,145,0.08);">
        <h2 style="color:#0b6b3a; margin-top:0;">Mulai dari sini</h2>
        <p style="margin-bottom: 1rem; color:#1f2d3d;">
            Klik tombol <b>Prediksi akademik</b> untuk mengisi data mahasiswa dan melihat hasil prediksi.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Prediksi akademik", use_container_width=True):
        # reset hasil sebelumnya agar alur rapi
        st.session_state.pop("hasil_akademik", None)
        st.session_state.pop("hasil_sentiment", None)
        st.session_state["page"] = "data_akademik"
        st.rerun()

# =====================
# Halaman: DATA AKADEMIK
# =====================
if st.session_state["page"] == "data_akademik":

    st.markdown("""
    <div class="header-box" style="max-width: 1200px; padding: 2.2rem 2.8rem;">
        <h1 style="margin: 0; font-size: 2.35rem; font-weight: 800; letter-spacing: 0.2px;">
            Academic Performance Monitoring System
        </h1>
        <p class="subtitle" style="margin: 0.55rem 0 0; font-size: 1.05rem; line-height: 1.55;">
            Sistem prediksi risiko akademik mahasiswa dan analisis sentimen feedback menggunakan Machine Learning
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("⬅️ Kembali ke Home", use_container_width=False):
        st.session_state["page"] = "welcome"
        st.session_state.pop("hasil_akademik", None)
        st.session_state.pop("hasil_sentiment", None)
        st.rerun()

    st.header("Data Akademik Mahasiswa")

    col1, col2 = st.columns(2)

    with col1:
        ipk = st.number_input(
            "IPK",
            min_value=0.0,
            max_value=4.0,
            value=3.0
        )
        mata_kuliah = st.text_input(
            "Mata Kuliah",
            value=""
        )

        kehadiran = st.slider(
            "Kehadiran (%)",
            0,
            100,
            80
        )

        nilai = st.slider(
            "Nilai Rata-rata",
            0,
            100,
            75
        )

    with col2:
        semester = st.selectbox(
            "Semester",
            [1, 2, 3, 4, 5, 6, 7, 8]
        )

        sks_lulus = st.number_input(
            "SKS Lulus",
            min_value=20,
            max_value=144,
            value=20
        )

    # =====================
    # PREDIKSI AKADEMIK (dengan validasi input)
    # =====================
    if st.button("Prediksi"):
        # Validasi field wajib
        missing = []
        if not (isinstance(mata_kuliah, str) and mata_kuliah.strip()):
            missing.append("Mata Kuliah")

        # Jika ada field yang kosong, beri peringatan dan jangan jalankan model
        if missing:
            st.warning("Isi data wajib terlebih dahulu: " + ", ".join(missing))
        else:
            data = [[ipk, kehadiran, nilai, semester, sks_lulus]]
            hasil_akademik = model_akademik.predict(data)
            st.session_state["hasil_akademik"] = hasil_akademik[0]

    # =====================
    # HASIL AKADEMIK
    # =====================
    if "hasil_akademik" in st.session_state:

        st.markdown("---")
        st.subheader("📌 Hasil Prediksi Akademik")

        if st.session_state["hasil_akademik"] == "Berisiko":
            st.error("⚠️ Status Akademik: BERISIKO")
        else:
            st.success("✅ Status Akademik: TIDAK BERISIKO")

    # =====================
    # FEEDBACK
    # =====================
    if "hasil_akademik" in st.session_state:
        st.markdown("---")
        st.header("Feedback Mahasiswa")

        feedback = st.text_area("Masukkan feedback mahasiswa")
        # Tampilkan hasil preprocessing di bawah input
        processed_feedback = preprocess_text(feedback)
        if feedback and processed_feedback:
            st.markdown("**Teks hasil preprocessing:**")
            st.write(processed_feedback)

        # =====================
        # PREDIKSI SENTIMEN (dengan validasi)
        # =====================
        if st.button("📝 Analisis Sentimen"):
            if not (isinstance(feedback, str) and feedback.strip()):
                st.warning("Masukkan feedback mahasiswa terlebih dahulu.")
            else:
                # Simpan komentar ke CSV (buat jika belum ada, atau tambahkan baris)
                csv_path = "dataset_komentar_mahasiswa.csv"
                try:
                    df_new = pd.DataFrame({"komentar": [feedback]})
                    if not os.path.exists(csv_path):
                        df_new.to_csv(csv_path, index=False, encoding='utf-8-sig')
                    else:
                        df_new.to_csv(csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')
                    st.success("Komentar berhasil disimpan.")
                except Exception as e:
                    st.error(f"Gagal menyimpan komentar: {e}")

                # Lakukan analisis sentimen pada teks yang sudah dipreproses
                hasil_sentiment = model_sentiment.predict([processed_feedback])
                st.session_state["hasil_sentiment"] = hasil_sentiment[0]
                # Simpan hasil sentimen ke riwayat_sentimen (session)
                st.session_state.setdefault("riwayat_sentimen", []).append(st.session_state["hasil_sentiment"])

    # =====================
    # HASIL AKHIR
    # =====================
    if (
        "hasil_akademik" in st.session_state
        and "hasil_sentiment" in st.session_state
    ):

        st.markdown("---")
        st.header("📋 Hasil Akhir Analisis")

        st.write(f"**Status Akademik:** {st.session_state['hasil_akademik']}")
        st.write(f"**Sentimen Feedback:** {st.session_state['hasil_sentiment']}")

        if (
            st.session_state["hasil_akademik"] == "Berisiko"
            and st.session_state["hasil_sentiment"] == "Negatif"
        ):
            st.error("⚠️ Mahasiswa membutuhkan perhatian lebih.")

            st.warning("""
Rekomendasi:
• Tingkatkan kehadiran
• Konsultasi dengan dosen wali
• Ikuti kelompok belajar
""")

        elif st.session_state["hasil_sentiment"] == "Positif":
            st.success("""
Rekomendasi:
• Pertahankan prestasi akademik
• Tetap aktif dalam perkuliahan
• Tingkatkan kompetensi diri
""")

        else:
            st.info("📌 Tetap pantau perkembangan akademik secara berkala.")

        # ----------------- Dashboard Analisis Sentimen -----------------
        riwayat_s = st.session_state.get("riwayat_sentimen", [])
        if riwayat_s:
            counts = pd.Series(riwayat_s).value_counts()
            pos = int(counts.get("Positif", 0))
            neu = int(counts.get("Netral", 0))
            neg = int(counts.get("Negatif", 0))
        else:
            pos = neu = neg = 0

        s1, s2, s3 = st.columns(3)
        s1.metric("Sentimen Positif", pos)
        s2.metric("Sentimen Netral", neu)
        s3.metric("Sentimen Negatif", neg)

        # Pie chart
        df_sent = pd.DataFrame({
            "sentiment": ["Positif", "Netral", "Negatif"],
            "count": [pos, neu, neg]
        })
        fig_pie_sent = px.pie(df_sent, names="sentiment", values="count", title="Distribusi Sentimen Feedback", color_discrete_sequence=["#0b3d91", "#6fbf73", "#0b6b3a"])
        fig_pie_sent.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie_sent, use_container_width=True)

        # Bar chart
        fig_bar_sent = px.bar(df_sent, x="sentiment", y="count", color="sentiment", title="Jumlah per Kategori Sentimen", color_discrete_map={"Positif": "#0b3d91", "Netral": "#6fbf73", "Negatif": "#0b6b3a"})
        fig_bar_sent.update_layout(showlegend=False)
        st.plotly_chart(fig_bar_sent, use_container_width=True)

        # Insight AI: interpretasi otomatis berdasarkan sentimen dominan
        total_s = pos + neu + neg
        if total_s > 0:
            dominant = max((pos, "Positif"), (neu, "Netral"), (neg, "Negatif"))[1]
            if dominant == "Negatif":
                st.error("Insight: Sentimen mayoritas negatif — perlu tindakan cepat. Pertimbangkan pengumpulan umpan balik lebih lanjut, sesi konseling, atau perbaikan kualitas pengajaran.")
            elif dominant == "Positif":
                st.success("Insight: Sentimen mayoritas positif — mahasiswa menunjukkan kepuasan. Pertahankan praktik yang efektif dan komunikasi yang baik.")
            else:
                st.info("Insight: Sentimen mayoritas netral — pantau perkembangan dan kumpulkan detail tambahan bila perlu.")
        # ---------------------------------------------------------------

        # ----------------- WordCloud Komentar Mahasiswa -----------------
        st.markdown("## ☁️ WordCloud Komentar Mahasiswa")
        csv_path = "dataset_komentar_mahasiswa.csv"
        if not os.path.exists(csv_path):
            st.info("Belum ada data komentar untuk WordCloud.")
        else:
            try:
                df_comments = pd.read_csv(csv_path, encoding='utf-8-sig')
                if "komentar" not in df_comments.columns or df_comments["komentar"].dropna().empty:
                    st.info("Belum ada data komentar untuk WordCloud.")
                else:
                    text = " ".join(df_comments["komentar"].astype(str).dropna().tolist())
                    wc = WordCloud(background_color="white", width=800, height=400).generate(text)
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"Gagal membuat WordCloud: {e}")

        st.markdown("---")
        st.caption("Academic Performance Monitoring System | Machine Learning Project")

