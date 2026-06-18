import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Config halaman web
st.set_page_config(page_title="Agrikultur Presisi - UNP", layout="centered")

# Definisikan nama kelas penyakit sesuai dengan urutan index dataset saat training
# Catatan: Sesuaikan isi list ini dengan nama kelas tanaman asli yang muncul di Colab Anda
CLASS_NAMES = [
   # Urutan nama kelas yang sudah diperbaiki sesuai log output training Google Colab
    'Pepper__bell___Bacterial_spot',                      # Kelas 0
    'Pepper__bell___healthy',                             # Kelas 1
    'Potato___Early_blight',                              # Kelas 2
    'Potato___Late_blight',                               # Kelas 3
    'Potato___healthy',                                   # Kelas 4
    'Tomato_Bacterial_spot',                              # Kelas 5
    'Tomato_Early_blight',                                # Kelas 6
    'Tomato_Late_blight',                                 # Kelas 7
    'Tomato_Leaf_Mold',                                   # Kelas 8
    'Tomato_Septoria_leaf_spot',                          # Kelas 9
    'Tomato_Spider_mites_Two_spotted_spider_mite',        # Kelas 10
    'Tomato__Target_Spot',                                # Kelas 11
    'Tomato__Tomato_YellowLeaf__Curl_Virus',              # Kelas 12
    'Tomato__Tomato_mosaic_virus',                        # Kelas 13
    'Tomato_healthy'                                      # Kelas 14 (Sekarang aman terbaca)
]


@st.cache_resource
def load_our_model():
    # Memuat file model AI yang diunduh dari Google Drive
    # Pastikan file model berada dalam folder yang sama dengan file app.py ini
    return tf.keras.models.load_model('model_agrikultur_presisi.keras')

try:
    model = load_our_model()
    st.sidebar.success("Model AI Berhasil Dimuat!")
except Exception as e:
    st.sidebar.error(f"Gagal memuat model. Pastikan file model ada di folder yang sama. Error: {e}")

# Header Aplikasi
st.title("🌱 Sistem Deteksi Penyakit Tanaman")
st.write("Proyek Agrikultur Presisi - Pengembangan Sistem Deteksi Berbasis Deep Learning CNN")
st.write("---")

# Komponen Upload Gambar
uploaded_file = st.file_uploader("Unggah foto daun tanaman yang ingin dianalisis...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Menampilkan gambar yang diunggah pengguna
    image = Image.open(uploaded_file)
    st.image(image, caption='Gambar Daun Terunggah', use_column_width=True)
    
    # Tombol Aksi Analisis
    if st.button("Mulai Diagnosis Penyakit"):
        with st.spinner('Sedang memindai tekstur dan warna daun...'):
            # Pra-pemrosesan gambar agar sesuai input training (150x150)
            img_resized = image.resize((150, 150))
            img_array = np.array(img_resized)
            
            # Memastikan gambar memiliki 3 channel warna (RGB)
            if img_array.shape[-1] == 4:
                img_array = img_array[..., :3]
                
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalisasi nilai piksel
            
            # Prediksi menggunakan model CNN
            predictions = model.predict(img_array)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100
            
            # Menangani penentuan nama kelas secara aman
            if predicted_class_idx < len(CLASS_NAMES):
                result_label = CLASS_NAMES[predicted_class_idx]
            else:
                result_label = f"Kelas Tidak Dikenal (Index {predicted_class_idx})"
            
            # Menampilkan hasil ke antarmuka pengguna
            st.write("---")
            st.subheader("📋 Hasil Analisis Sistem:")
            
            if "Healthy" in result_label:
                st.success(f"Kondisi: **{result_label}**")
            else:
                st.error(f"Terdeteksi Penyakit: **{result_label}**")
                
            st.info(f"Tingkat Keyakinan Prediksi AI (*Confidence Score*): **{confidence:.2f}%**")