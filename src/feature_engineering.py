import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import entropy
import joblib
import os
import re

class FeatureEngineer:
    def __init__(self, vectorizer_path=None):
        if vectorizer_path and os.path.exists(vectorizer_path):
            self.vectorizer = joblib.load(vectorizer_path)
        else:
            self.vectorizer = TfidfVectorizer(
                analyzer='char',
                ngram_range=(2, 5),
                max_features=20000,
                lowercase=True
            )

    def get_statistical_features(self, text_series):
        features = pd.DataFrame()
        features['length'] = text_series.apply(len) / 1000.0
        
        def calc_entropy(text):
            if not text: return 0
            counts = pd.Series(list(text)).value_counts()
            return entropy(counts) / 10.0
        
        features['entropy'] = text_series.apply(calc_entropy)
        
        # Keyword frequencies
        keywords = [
            'select', 'union', 'insert', 'update', 'delete', 'drop', 'from', 'where',
            'script', 'alert', 'onerror', 'eval', '../', './',
            '${', '{{', '}}', '() {', ';', '|', '&',
            '$gt', '$ne', '$in', 'cat ', 'whoami'
        ]
        for kw in keywords:
            features[f'kw_{kw}'] = text_series.apply(lambda x: x.count(kw) / (len(x) + 1))
            
        return features

    def extract_text(self, row):
        """Standard bag-of-fragments extraction."""
        fields = ['path', 'query', 'headers', 'body']
        vals = []
        for f in fields:
            v = str(row.get(f, '')).strip()
            if v and v.lower() != 'nan':
                vals.append(v)
        
        return " ".join(vals)

    def fit(self, df):
        print("Standardizing text for training...")
        if isinstance(df, pd.Series):
            texts = df.astype(str)
        else:
            texts = df.apply(self.extract_text, axis=1)
            
        from preprocessing import clean_text
        cleaned_texts = texts.apply(clean_text)
        self.vectorizer.fit(cleaned_texts)
        return self

    def transform(self, df):
        if isinstance(df, pd.Series):
            texts = df.astype(str)
        else:
            texts = df.apply(self.extract_text, axis=1)
            
        from preprocessing import clean_text
        cleaned_texts = texts.apply(clean_text)
        tfidf_features = self.vectorizer.transform(cleaned_texts)
        
        # Use values to ensure Series for statistical features
        stat_features = self.get_statistical_features(pd.Series(texts.values))
        
        # Combine TF-IDF and statistical features
        from scipy.sparse import hstack
        combined = hstack([tfidf_features, stat_features.values])
        return combined

    def save(self, path):
        joblib.dump(self.vectorizer, path)

if __name__ == "__main__":
    pass
