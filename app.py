import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from datetime import datetime
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

# Custom modern styling (professional, elegant, no sidebar)
st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
        :root{
            --bg: #f6f8fb;
            --card: #ffffff;
            --primary: #0b3d91;
            --accent: #0b6b3a;
            --muted: #6b7280;
        }
        html, body, [class*="stApp"] {
            font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
            background: linear-gradient(180deg, var(--bg) 0%, #ffffff 100%);
            color: #0f1724;
        }
        .block-container{ background: transparent; padding: 0; max-width: 1180px; margin: 1.25rem auto; }
        .app-card{ background: var(--card); padding: 1.25rem; border-radius: 14px; box-shadow: 0 10px 30px rgba(15,23,42,0.06); border: 1px solid rgba(15,23,42,0.04); }
        .header-box{ background: linear-gradient(90deg, rgba(11,61,145,0.95) 0%, rgba(11,107,58,0.95) 100%); color: #fff; padding: 2rem; border-radius: 12px; box-shadow: 0 12px 40px rgba(11,61,145,0.08); text-align:center; }
        .header-box h1{ margin:0; font-size:2.4rem; font-weight:800; letter-spacing:0.2px; }
        .header-box .subtitle{ margin-top:0.6rem; color: rgba(255,255,255,0.95); font-size:1rem }
        .muted { color: var(--muted); }
        .stButton>button{ background: linear-gradient(90deg,var(--primary),var(--accent)); color:#fff; border-radius:10px; padding:.55rem 1rem; border:none; box-shadow: 0 8px 24px rgba(11,61,145,0.08); }
        .stButton>button:hover{ transform: translateY(-1px); }
        input, textarea, .stSelectbox>div>div{ border-radius:10px; }
        .metric-card{ padding:1rem; border-radius:12px; background:linear-gradient(180deg,#fff,#fbfdff); text-align:center; box-shadow: 0 8px 24px rgba(15,23,42,0.04); }
        .metric-card .label{ color:var(--muted); font-size:0.9rem }
        .metric-card .value{ font-weight:700; font-size:1.4rem; color:var(--primary) }
        .stPlotlyChart{ border-radius:12px; }
        @media (max-width:600px){ .header-box h1{ font-size:1.6rem } }
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

            # Simpan riwayat prediksi ke CSV
            try:
                os.makedirs("data", exist_ok=True)
                csv_path = os.path.join("data", "hasil_prediksi.csv")
                df_new = pd.DataFrame({
                    "IPK": [ipk],
                    "Semester": [semester],
                    "SKS_Lulus": [sks_lulus],
                    "Kehadiran": [kehadiran],
                    "Hasil_Prediksi": [st.session_state["hasil_akademik"]],
                    "Tanggal_Prediksi": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
                })
                if not os.path.exists(csv_path):
                    df_new.to_csv(csv_path, index=False, encoding='utf-8-sig')
                else:
                    df_new.to_csv(csv_path, mode='a', header=False, index=False, encoding='utf-8-sig')
            except Exception as e:
                st.error(f"Gagal menyimpan riwayat prediksi: {e}")

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
        # EVALUASI MODEL (TERSEMBUNYI DI EXPANDER)
        # =====================
        with st.expander("📊 Lihat Detail Evaluasi Model", expanded=False):
            st.markdown("---")
            st.subheader("📊 Evaluasi Model")
            try:
                # Load dataset for evaluation
                df_eval = pd.read_csv("dataset_mahasiswa.csv")
                # prepare X and y (use requested features)
                X = df_eval[["IPK", "Kehadiran", "Nilai_Rata", "Semester", "SKS_Lulus"]]
                # normalize target labels (handle both 'Tidak_Berisiko' and 'Tidak Berisiko')
                y = df_eval["Status"].astype(str).str.replace("_", " ").str.strip()

                # split and evaluate
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
                y_pred = model_akademik.predict(X_test)
                # normalize predictions labels too
                y_pred = pd.Series(y_pred).astype(str).str.replace("_", " ").str.strip()
                y_test = pd.Series(y_test).astype(str).str.replace("_", " ").str.strip()

                # classification report (use weighted avg to summarize multiclass)
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                precision = float(report.get("weighted avg", {}).get("precision", 0.0))
                recall = float(report.get("weighted avg", {}).get("recall", 0.0))
                f1 = float(report.get("weighted avg", {}).get("f1-score", 0.0))
                acc = accuracy_score(y_test, y_pred)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Accuracy", f"{acc:.3f}")
                m2.metric("Precision", f"{precision:.3f}")
                m3.metric("Recall", f"{recall:.3f}")
                m4.metric("F1-Score", f"{f1:.3f}")

                # Confusion matrix
                st.markdown("---")
                st.subheader("🔢 Confusion Matrix Model Prediksi Risiko Akademik")
                labels = ["Tidak Berisiko", "Berisiko"]
                cm = confusion_matrix(y_test, y_pred, labels=labels)
                fig_cm, ax = plt.subplots(figsize=(5, 4))
                im = ax.imshow(cm, cmap='Blues')
                for (i, j), val in np.ndenumerate(cm):
                    ax.text(j, i, int(val), ha='center', va='center', color='black')
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                ax.set_xticklabels(labels)
                ax.set_yticklabels(labels)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title("Confusion Matrix Model Prediksi Risiko Akademik")
                fig_cm.colorbar(im, ax=ax)
                st.pyplot(fig_cm)

            except Exception as e:
                st.info(f"Evaluasi model tidak tersedia: {e}")
            
            # =====================
            # STATISTIK HASIL PREDIKSI
            # =====================
            st.markdown("---")
            st.subheader("📈 Statistik Hasil Prediksi")
            hist_path = os.path.join("data", "hasil_prediksi.csv")
            if os.path.exists(hist_path):
                try:
                    df_hist = pd.read_csv(hist_path, encoding='utf-8-sig')
                    # normalize label variants
                    df_hist["Hasil_Prediksi"] = df_hist["Hasil_Prediksi"].astype(str).replace({"Tidak_Berisiko": "Tidak Berisiko"})
                    total_pred = len(df_hist)
                    jumlah_berisiko = int((df_hist["Hasil_Prediksi"] == "Berisiko").sum())
                    jumlah_tidak = int((df_hist["Hasil_Prediksi"] == "Tidak Berisiko").sum())

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Prediksi", total_pred)
                    c2.metric("Jumlah Mahasiswa Berisiko", jumlah_berisiko)
                    c3.metric("Jumlah Mahasiswa Tidak Berisiko", jumlah_tidak)

                    # Grafik distribusi
                    st.markdown("---")
                    st.subheader("📊 Distribusi Hasil Prediksi Mahasiswa")
                    df_dist = df_hist["Hasil_Prediksi"].value_counts().reset_index()
                    df_dist.columns = ["Hasil", "Jumlah"]
                    fig_pie_pred = px.pie(df_dist, names="Hasil", values="Jumlah", title="Distribusi Hasil Prediksi Mahasiswa", color_discrete_sequence=["#0b3d91", "#0b6b3a"]) 
                    fig_pie_pred.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie_pred, use_container_width=True)

                    fig_bar_pred = px.bar(df_dist, x="Hasil", y="Jumlah", color="Hasil", title="Distribusi Hasil Prediksi Mahasiswa", color_discrete_map={"Berisiko": "#d62828", "Tidak Berisiko": "#2a9d8f"})
                    fig_bar_pred.update_layout(showlegend=False)
                    st.plotly_chart(fig_bar_pred, use_container_width=True)

                    # Dataset hasil prediksi
                    st.markdown("---")
                    st.subheader("📚 Dataset Hasil Prediksi Mahasiswa")
                    st.dataframe(df_hist)

                    csv_bytes = df_hist.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                    st.download_button(label="Download CSV Hasil Prediksi", data=csv_bytes, file_name="hasil_prediksi.csv", mime="text/csv")

                except Exception as e:
                    st.error(f"Gagal membaca data hasil prediksi: {e}")
            else:
                st.info("Belum ada riwayat prediksi. Tekan tombol Prediksi untuk mulai menyimpan hasil.")

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

