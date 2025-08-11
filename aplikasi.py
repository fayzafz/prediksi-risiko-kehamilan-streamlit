import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
# Menggunakan ImbPipeline karena model Anda dilatih dengan SMOTE di dalamnya
from imblearn.pipeline import Pipeline as ImbPipeline

# set_page_config harus menjadi perintah Streamlit pertama setelah import
st.set_page_config(page_title="Klasifikasi Risiko Kehamilan", layout="centered")


# --- 1. Konfigurasi dan Pemuatan Model ---
# Jalur tempat file model dan encoder disimpan.
# Saat di Streamlit Cloud, file-file ini harus berada di direktori yang sama
# atau Anda perlu mengaturnya agar bisa diakses.
# Untuk demo lokal, pastikan file .pkl ada di folder yang sama dengan app.py,
# atau sesuaikan path di bawah.

MODEL_PATH = 'model_risiko_kehamilan_best_overall.pkl'
LABEL_ENCODER_PATH = 'label_encoder_risiko.pkl'

# Periksa apakah file ada (penting untuk Streamlit Cloud)
if not os.path.exists(MODEL_PATH):
    st.error(f"File model tidak ditemukan: {MODEL_PATH}. Pastikan file .pkl ada di direktori yang benar.")
    st.stop()
if not os.path.exists(LABEL_ENCODER_PATH):
    st.error(f"File label encoder tidak ditemukan: {LABEL_ENCODER_PATH}. Pastikan file .pkl ada di direktori yang benar.")
    st.stop()

# Memuat model dan label encoder
@st.cache_resource # Cache model agar tidak dimuat ulang setiap kali aplikasi refresh
def load_artifacts():
    try:
        model = joblib.load(MODEL_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)

        # Mendapatkan nama kolom fitur yang digunakan saat training dari model
        fitted_preprocessor = model.named_steps['preprocessor']
        
        numeric_features = []
        categorical_features_ohe = []
        for name, transformer, original_cols in fitted_preprocessor.transformers_:
            if name == 'num':
                numeric_features.extend(original_cols)
            elif name == 'cat':
                if isinstance(transformer, OneHotEncoder):
                    categorical_features_ohe.extend(transformer.get_feature_names_out(original_cols))
                else:
                    categorical_features_ohe.extend(original_cols)

        expected_model_features = numeric_features + categorical_features_ohe
        
        return model, label_encoder, expected_model_features
    except Exception as e:
        st.error(f"Gagal memuat artefak model. Error: {e}")
        st.stop()

model, label_encoder, expected_model_features = load_artifacts()

# --- 2. Antarmuka Streamlit ---
st.title("🤰 Klasifikasi Risiko Kesehatan Ibu Hamil")
st.markdown("Aplikasi ini menggunakan model Machine Learning (Extreme Gradient Boosting) untuk mengklasifikasi tingkat risiko kesehatan ibu hamil berdasarkan parameter yang diberikan.")

# --- PERUBAHAN DI SINI: Layout Input Menggunakan Kolom ---
st.subheader("Harap masukkan nilai untuk setiap fitur di bawah ini:")

# Membuat dua kolom untuk input
col1, col2 = st.columns(2)

with col1:
    usia = st.number_input("Usia Ibu (tahun)", min_value=15, max_value=50, value=28, step=1)
    tekanan_darah_sistolik = st.number_input("Tekanan Darah Sistolik (mmHg)", min_value=70, max_value=180, value=120, step=1)
    kehamilan_ke = st.number_input("Kehamilan Ke- (Gravida)", min_value=1, max_value=10, value=1, step=1)

with col2:
    tekanan_darah_diastolik = st.number_input("Tekanan Darah Diastolik (mmHg)", min_value=40, max_value=120, value=80, step=1)
    kadar_hb = st.number_input("Kadar Hemoglobin (gr%)", min_value=5.0, max_value=19.0, value=12.0, step=0.1, format="%.1f")
    status_gizi = st.selectbox("Status Gizi", ("Normal", "Kurang", "Lebih", "Obesitas", "Sangat Kurus"))

# --- Tombol Prediksi (Sekarang di area utama) ---
if st.button("Klasifikasi Risiko"): # Mengubah st.sidebar.button menjadi st.button
    st.subheader("Hasil Klasifikasi:") # Mengubah subheader agar lebih umum
    
    try:
        # Mengumpulkan input dalam bentuk DataFrame
        input_data_raw = pd.DataFrame({
            'usia': [usia],
            'tekanan_darah_sistolik': [tekanan_darah_sistolik],
            'tekanan_darah_diastolik': [tekanan_darah_diastolik],
            'kehamilan_ke': [kehamilan_ke],
            'kadar_hb': [kadar_hb],
            'status_gizi': [status_gizi]
        })

        # Membuat DataFrame dengan semua kolom yang diharapkan oleh preprocessor
        processed_input_df = pd.DataFrame(columns=[
            'usia', 'tekanan_darah_sistolik', 'tekanan_darah_diastolik',
            'kehamilan_ke', 'kadar_hb', 'status_gizi'
        ])
        
        # Isi dengan data input dari user
        for col in processed_input_df.columns:
            if col in input_data_raw.columns:
                processed_input_df[col] = input_data_raw[col]

        # Lakukan prediksi menggunakan model yang sudah dilatih
        prediction_encoded = model.predict(processed_input_df)
        prediction_label = label_encoder.inverse_transform(prediction_encoded)[0]

        # Menampilkan hasil prediksi dari model ML
        st.success(f"Tingkat Risiko: **{prediction_label}**")

        # --- BAGIAN RULE-BASED DIHAPUS SEPENUHNYA ---
        # st.subheader("1. Prediksi Berdasarkan Aturan (Rule-Based)")
        # st.write("Tingkat Risiko (Rule-Based): ...")
        # st.write("Penyebab (Rule-Based): ...")
        # --- AKHIR BAGIAN RULE-BASED YANG DIHAPUS ---
        
    except Exception as e:
        st.error(f"Terjadi kesalahan saat mengklasifikasi: {e}")
        st.info("Pastikan semua nilai input valid.")

# --- Bagian Footer (bisa tetap di sidebar atau dipindahkan ke main area) ---
# Untuk menjaga kesederhanaan, saya pindahkan ke main area jika sidebar tidak digunakan banyak
st.markdown("---")
st.markdown("Dibuat oleh Fayza Fatimattuzahra")
