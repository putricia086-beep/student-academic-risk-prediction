import streamlit as st
import joblib

# =====================

# KONFIGURASI HALAMAN

# =====================

st.set_page_config(
page_title="Academic Performance Monitoring System",
page_icon="🎓",
layout="wide"
)

# =====================

# LOAD MODEL

# =====================

model_akademik = joblib.load("model_akademik.pkl")
model_sentiment = joblib.load("model_sentiment.pkl")

# =====================

# HEADER

# =====================

st.title("🎓 Academic Performance Monitoring System")

st.markdown("""
Sistem prediksi risiko akademik mahasiswa dan analisis sentimen feedback
menggunakan Machine Learning.
""")

st.markdown("---")

# =====================

# STEP 1

# =====================

st.header("📊 STEP 1 - Data Akademik Mahasiswa")

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
```

with col2:

```
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
```

if st.button("🔍 Prediksi Akademik"):

```
data = [[
    ipk,
    kehadiran,
    nilai,
    semester,
    sks
]]

hasil_akademik = model_akademik.predict(data)

st.session_state["hasil_akademik"] = hasil_akademik[0]
```

# =====================

# HASIL AKADEMIK

# =====================

if "hasil_akademik" in st.session_state:

```
st.markdown("---")

st.subheader("📌 Hasil Prediksi Akademik")

if st.session_state["hasil_akademik"] == "Berisiko":

    st.error("⚠️ Status Akademik: BERISIKO")

else:

    st.success("✅ Status Akademik: TIDAK BERISIKO")

# =====================
# STEP 2
# =====================

st.markdown("---")

st.header("💬 STEP 2 - Feedback Mahasiswa")

feedback = st.text_area(
    "Masukkan feedback mahasiswa"
)

if st.button("📝 Analisis Sentimen"):

    hasil_sentiment = model_sentiment.predict(
        [feedback]
    )

    st.session_state["hasil_sentiment"] = hasil_sentiment[0]
```

# =====================

# HASIL AKHIR

# =====================

if (
"hasil_akademik" in st.session_state
and
"hasil_sentiment" in st.session_state
):

```
st.markdown("---")

st.header("📋 Hasil Akhir Analisis")

st.write(
    f"**Status Akademik:** {st.session_state['hasil_akademik']}"
)

st.write(
    f"**Sentimen Feedback:** {st.session_state['hasil_sentiment']}"
)

if (
    st.session_state["hasil_akademik"] == "Berisiko"
    and
    st.session_state["hasil_sentiment"] == "Negatif"
):

    st.error("""
```

⚠️ Mahasiswa terindikasi membutuhkan perhatian lebih.
""")

```
    st.warning("""
```

Rekomendasi:
• Tingkatkan kehadiran
• Konsultasi dengan dosen wali
• Ikuti kelompok belajar
""")

```
elif st.session_state["hasil_sentiment"] == "Positif":

    st.success("""
```

✅ Performa akademik dan feedback menunjukkan hasil yang baik.
""")

```
    st.success("""
```

Rekomendasi:
• Pertahankan prestasi akademik
• Tetap aktif dalam perkuliahan
• Tingkatkan kompetensi diri
""")

```
else:

    st.info("""
```

📌 Tetap pantau perkembangan akademik secara berkala.
""")

st.markdown("---")

st.caption("Academic Performance Monitoring System | Machine Learning Project")
