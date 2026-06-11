
import streamlit as st
import joblib

# =====================
# KONFIGURASI HALAMAN
# =====================

st.set_page_config(
    page_title="Student Academic Prediction",
    page_icon="🎓",
    layout="wide"
)

# =====================
# LOAD MODEL
# =====================

model_akademik = joblib.load("model_akademik.pkl")
model_sentiment = joblib.load("model_sentiment.pkl")

# =====================
# JUDUL
# =====================

st.title("🎓 Student Academic Risk Prediction & Sentiment Analysis")

st.markdown("""
Prediksi risiko akademik mahasiswa dan analisis sentimen feedback
menggunakan Machine Learning.
""")

st.markdown("---")

# =====================
# INPUT AKADEMIK
# =====================

st.header("📊 Data Akademik Mahasiswa")

col1, col2 = st.columns(2)

with col1:

    ipk = st.number_input(
        "IPK",
        min_value=0.0,
        max_value=4.0,
        value=3.0
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
        [1,2,3,4,5,6,7,8]
    )

    sks = st.number_input(
        "SKS Lulus",
        min_value=0,
        max_value=144,
        value=80
    )

if st.button("🔍 Prediksi Akademik"):

    data = [[
        ipk,
        kehadiran,
        nilai,
        semester,
        sks
    ]]

    hasil_akademik = model_akademik.predict(data)

    st.subheader("📌 Hasil Prediksi Akademik")

    if hasil_akademik[0] == "Berisiko":

        st.error("⚠️ Mahasiswa Berisiko")

    else:

        st.success("✅ Mahasiswa Tidak Berisiko")

st.markdown("---")

# =====================
# SENTIMENT ANALYSIS
# =====================

st.header("💬 Feedback Mahasiswa")

feedback = st.text_area(
    "Masukkan feedback Anda"
)

if st.button("📝 Analisis Sentimen"):

    hasil_sentiment = model_sentiment.predict(
        [feedback]
    )

    st.subheader("📌 Hasil Analisis Sentimen")

    if hasil_sentiment[0] == "Positif":

        st.success("😊 Sentimen Positif")

        st.success("""
Rekomendasi:
- Pertahankan performa akademik
- Tetap aktif mengikuti perkuliahan
- Terus tingkatkan prestasi
""")

    elif hasil_sentiment[0] == "Negatif":

        st.error("😟 Sentimen Negatif")

        st.warning("""
Rekomendasi:
- Tingkatkan konsistensi belajar
- Konsultasi dengan dosen wali
- Ikuti kelompok belajar
""")

    else:

        st.info("😐 Sentimen Netral")

        st.info("""
Rekomendasi:
- Tetap pantau perkembangan akademik
- Pertahankan rutinitas belajar
""")

st.markdown("---")

st.caption("Dibuat oleh Princess Awa 👑 | Machine Learning Project")
