import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import entropy
import joblib
import os

class FeatureEngineer:
    def __init__(self, vectorizer_path=None):
        if vectorizer_path and os.path.exists(vectorizer_path):
            self.vectorizer = joblib.load(vectorizer_path)
        else:
            self.vectorizer = TfidfVectorizer(
                analyzer='char',
                ngram_range=(2, 5),
                max_features=10000,
                lowercase=True
            )

    def get_statistical_features(self, text_series):
        features = pd.DataFrame()
        features['length'] = text_series.apply(len)
        features['special_chars_count'] = text_series.apply(lambda x: len([c for c in x if c in "'\";<>(){}[]/\\%"]))
        
        def calc_entropy(text):
            if not text: return 0
            counts = pd.Series(list(text)).value_counts()
            return entropy(counts)
        
        features['entropy'] = text_series.apply(calc_entropy)
        
        # Keyword frequencies (normalized)
        keywords = [
            'select', 'union', 'insert', 'update', 'delete', 'drop', 
            'script', 'alert', 'onerror', 'eval', '../', './',
            '${', '{{', '}}', '() {', ';', '||', '&&', '$(',
            'http://', 'https://', '169.254', '0x', 'entity',
            '__schema', '__proto__', 'javascript:', 'data:', 'vbscript:'
        ]
        for kw in keywords:
            features[f'kw_{kw}'] = text_series.apply(lambda x: x.count(kw) / (len(x) + 1))
            
        return features

    def extract_text(self, row):
        """Unified text extraction for consistency between training and prediction."""
        # Row is now expected to have the standard schema:
        # method, path, query, headers, body, ua
        method = str(row.get('method', ''))
        path = str(row.get('path', ''))
        query = str(row.get('query', ''))
        headers = str(row.get('headers', ''))
        body = str(row.get('body', ''))
        ua = str(row.get('ua', ''))
        
        return f"{path} {query} {headers} {body}"

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
    # Test feature engineering
    processed_dir = "/Users/dmac/Desktop/ml/data/processed"
    train_df = pd.read_csv(os.path.join(processed_dir, 'train.csv'))
    
    fe = FeatureEngineer()
    print("Fitting vectorizer...")
    fe.fit(train_df['cleaned_text'].fillna(""))
    
    print("Transforming training data...")
    X_train = fe.transform(train_df['cleaned_text'].fillna(""))
    print(f"Feature matrix shape: {X_train.shape}")
    
    os.makedirs("/Users/dmac/Desktop/ml/models", exist_ok=True)
    fe.save("/Users/dmac/Desktop/ml/models/vectorizer.joblib")
    print("Vectorizer saved.")
