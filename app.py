import streamlit as st
import pandas as pd
import joblib

# Load model (pastikan file 'model_nb.pkl' ada di repositori GitHub Anda)
@st.cache_resource
def load_model():
    return joblib.load('model_nb.pkl')

nb = load_model()

# Judul aplikasi
st.title("Prediksi Kelolosan Ujian Mahasiswa")
st.write("Masukkan data di bawah untuk memprediksi apakah akan **Lolos** atau **Tidak Lolos**.")

# Input pengguna
sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
motivation_level = st.selectbox("Motivation Level", options=[0, 1, 2], format_func=lambda x: ['Low', 'Medium', 'High'][x])
teacher_quality = st.selectbox("Teacher Quality", options=[0, 1, 2], format_func=lambda x: ['Low', 'Medium', 'High'][x])

# Tombol prediksi
if st.button("Prediksi"):
    # DataFrame input
    new_data_df = pd.DataFrame(
        [[sleep_hours, motivation_level, teacher_quality]],
        columns=['Sleep_Hours', 'Motivation_Level', 'Teacher_Quality']
    )

    try:
        predicted_code = nb.predict(new_data_df)[0]
        label_mapping = {1: 'Lolos', 0: 'Tidak Lolos'}
        predicted_label = label_mapping.get(predicted_code, 'Tidak diketahui')

        st.subheader("Hasil Prediksi:")
        st.write(f"Sleep Hours: {sleep_hours}")
        st.write(f"Motivation Level: {['Low', 'Medium', 'High'][motivation_level]}")
        st.write(f"Teacher Quality: {['Low', 'Medium', 'High'][teacher_quality]}")
        st.success(f"**Prediksi: {predicted_label}**")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses prediksi: {e}")
