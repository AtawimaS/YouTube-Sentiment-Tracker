import os
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from afinn import Afinn
from ngambil import ngambil_youtube
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class SentimentAnalyzer:
    def __init__(self, tfid_model, tfid_lr_model, tfid_svc_model):
        # Inisialisasi model
        self.tfid = tfid_model
        self.tfid_lr = tfid_lr_model
        self.tfid_svc = tfid_svc_model
        self.afinn = Afinn()
        self.vader = SentimentIntensityAnalyzer()

    def tfid_pipeline(self, comments):
        # Transformasi komentar dengan TF-IDF
        x_tfidf = self.tfid.transform(comments)
        
        # Prediksi Logistic Regression
        logistic_out = self.tfid_lr.predict(x_tfidf)
        logistic_pos_pct = (np.sum(logistic_out == 1) / len(logistic_out)) * 100
        logistic_neg_pct = (np.sum(logistic_out == 0) / len(logistic_out)) * 100

        # Prediksi SVC
        svc_out = self.tfid_svc.predict(x_tfidf)
        svc_pos_pct = (np.sum(svc_out == 1) / len(svc_out)) * 100
        svc_neg_pct = (np.sum(svc_out == 0) / len(svc_out)) * 100

        # Hasil
        return {
            "logistic": {"positive": logistic_pos_pct, "negative": logistic_neg_pct},
            "svc": {"positive": svc_pos_pct, "negative": svc_neg_pct},
        }

    def lexicon_pipeline(self, comments):
        afinn_scores = [self.afinn.score(comment) for comment in comments]
        vader_scores = [self.vader.polarity_scores(comment)['compound'] for comment in comments]
        
        # Menghitung persentase positif, negatif, dan netral
        afinn_pos_pct = (np.sum(np.array(afinn_scores) > 0) / len(afinn_scores)) * 100
        afinn_neg_pct = (np.sum(np.array(afinn_scores) < 0) / len(afinn_scores)) * 100
        afinn_neutral_pct = (np.sum(np.array(afinn_scores) == 0) / len(afinn_scores)) * 100

        vader_pos_pct = (np.sum(np.array(vader_scores) > 0.05) / len(vader_scores)) * 100
        vader_neg_pct = (np.sum(np.array(vader_scores) < -0.05) / len(vader_scores)) * 100
        vader_neutral_pct = (np.sum((np.array(vader_scores) >= -0.05) & (np.array(vader_scores) <= 0.05)) / len(vader_scores)) * 100

        # Hasil
        return {
            "afinn": {"positive": afinn_pos_pct, "negative": afinn_neg_pct, "neutral": afinn_neutral_pct},
            "vader": {"positive": vader_pos_pct, "negative": vader_neg_pct, "neutral": vader_neutral_pct},
        }

    def visualize_wordcloud(self, comments, title="WordCloud"):
        # Gabungkan semua komentar menjadi satu string
        text = " ".join(comments)
        
        # Generate WordCloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        return wordcloud
        

    def analyze(self, comments, output_path=None):
        # TF-IDF Model
        tfid_results = self.tfid_pipeline(comments)

        # Lexicon-based Model
        lexicon_results = self.lexicon_pipeline(comments)

        # Gabungkan hasil
        results = {
            "TFIDF_Results": tfid_results,
            "Lexicon_Results": lexicon_results
        }
        
        # Simpan hasil ke file JSON jika path diberikan
        if output_path:
            import json
            with open(output_path, 'w') as file:
                json.dump(results, file, indent=4)
        
        return results
