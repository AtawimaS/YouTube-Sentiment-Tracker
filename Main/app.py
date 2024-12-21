import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import ngambil  # Import fungsi dari file ngambil.py

import os
import pickle
from ngambil import ngambil_youtube
from sentiment_analisis import SentimentAnalyzer

# Load Model Sentimen
os.chdir('../Saved_Model')
now = os.getcwd()

path = os.path.join(now, 'tfid_lr_model.pkl')
with open(path, 'rb') as file:
    tfid_lr = pickle.load(file)
path = os.path.join(now, 'tfid_svc_model.pkl')
with open(path, 'rb') as file:
    tfid_svc = pickle.load(file)
path = os.path.join(now, 'tfidf_model.pkl')
with open(path, 'rb') as file:
    tfid = pickle.load(file)

os.chdir('../Main')

# Inisialisasi SentimentAnalyzer
analyzer = SentimentAnalyzer(tfid, tfid_lr, tfid_svc)

# Fungsi untuk scraping komentar YouTube dan analisis sentimen otomatis
def scrape_and_analyze_youtube(url, num_comments=30):
    comments_data = ngambil_youtube(url=url, num_comments=num_comments)
    df = pd.DataFrame(comments_data, columns=['Comments'])
    results = analyzer.analyze(df['Comments'].tolist())
    return results, df

# Streamlit UI
st.title("YouTube Comments Sentiment Analyzer")

# Input YouTube URL
youtube_url = st.text_input("Masukkan URL Video YouTube:")

# Pilihan visualisasi
visualization_type = st.radio(
    "Pilih jenis visualisasi:",
    ('Bar Chart', 'Pie Chart')
)

if st.button("Ambil Komentar dan Analisis Sentimen"):
    if youtube_url:
        st.info("Mengambil komentar dari YouTube...")
        
        try:
            results, df = scrape_and_analyze_youtube(youtube_url, num_comments=30)
            st.success(f"Berhasil mengambil {len(df)} komentar.")

            # Visualisasi Sentimen
            st.subheader("Distribusi Sentimen Berdasarkan Model TFIDF:")
            sentiment_counts = pd.DataFrame(results['TFIDF_Results']).T
            sentiment_counts['Model'] = sentiment_counts.index
            sentiment_counts = sentiment_counts.set_index('Model')

            if visualization_type == 'Bar Chart':
                st.bar_chart(sentiment_counts)
            else:
                # Pie chart menggunakan matplotlib
                st.write("Pie Chart Visualisasi:")
                fig, ax = plt.subplots()
                sentiment_counts.sum().plot(kind='pie', autopct='%1.1f%%', ax=ax, startangle=90)
                ax.set_ylabel('')  # Remove default y-label
                st.pyplot(fig)

            # Visualisasi Lexicon Results
            st.subheader("Distribusi Sentimen Berdasarkan Lexicon:")
            lexicon_results = pd.DataFrame(results['Lexicon_Results']).T
            lexicon_results['Model'] = lexicon_results.index
            lexicon_results = lexicon_results.set_index('Model')

            if visualization_type == 'Bar Chart':
                st.bar_chart(lexicon_results)
            else:
                st.write("Pie Chart Visualisasi:")
                fig, ax = plt.subplots()
                lexicon_results.sum().plot(kind='pie', autopct='%1.1f%%', ax=ax, startangle=90)
                ax.set_ylabel('')  # Remove default y-label
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
    else:
        st.warning("Silakan masukkan URL YouTube terlebih dahulu.")
