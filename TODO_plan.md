# Plan (Header Welcome Page)

## Information Gathered
- Aplikasi Streamlit berada di `app.py`.
- Welcome page sudah ada pada kondisi `st.session_state["page"] == "welcome"`.
- Pada welcome page saat ini terdapat: header-box “Selamat Datang” + deskripsi, lalu section “Mulai dari sini”.
- Bagian yang dimaksud tugas: “header di halaman welcome page di hapus, lalu kolom selamat datang di perbesar”. Pada `app.py`, header-box pada welcome adalah “Selamat Datang” (kolom) dan ada juga global header-box “Academic Performance Monitoring System” yang selalu tampil.

## Plan
1. Hapus header-box global yang muncul di seluruh halaman (yang berisi “Academic Performance Monitoring System”).
2. Perbesar “kolom Selamat Datang” di welcome page:
   - Naikkan `max-width` menjadi lebih besar.
   - Besarkan ukuran font h1.
   - Tambahkan padding lebih besar dan spacing lebih lega.
   - Tingkatkan styling agar terlihat lebih menarik (tetap konsisten dengan tema hijau-biru).
3. Jalankan `streamlit` untuk memverifikasi tidak ada error dan tampilan welcome sesuai.

## Dependent Files to be edited
- `app.py`

## Followup steps
- Jalankan: `streamlit run app.py` dan verifikasi welcome page.

<ask_followup_question>
Apakah yang dimaksud “header di halaman welcome page di hapus” adalah menghapus header-box global “Academic Performance Monitoring System” yang tampil di semua halaman? Jika iya, saya akan menghapusnya dan fokus membesarkan kolom “Selamat Datang”.
</ask_followup_question>

