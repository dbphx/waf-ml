import os

import joblib
import pandas as pd
from scipy.stats import entropy
from sklearn.feature_extraction.text import TfidfVectorizer

SUSPICIOUS_KEYWORDS = [
    'select', 'union', 'insert', 'update', 'delete', 'drop', 'from', 'where',
    'script', 'alert(', 'onerror', 'eval', '../', './',
    '${', '{{', '}}', '() {', ';', '|', '&',
    '$gt', '$ne', '$in', 'cat ', 'whoami'
]

REQUEST_FIELDS = ('path', 'query', 'headers', 'body')
FIELD_MAX_FEATURES = {
    'path': 5000,
    'query': 5000,
    'headers': 5000,
    'body': 5000,
}


class FeatureEngineer:
    def __init__(self, vectorizer_path=None):
        self.legacy_shared_vectorizer = None
        if vectorizer_path and os.path.exists(vectorizer_path):
            stored = joblib.load(vectorizer_path)
            if isinstance(stored, dict) and 'vectorizers' in stored:
                self.vectorizers = stored['vectorizers']
            elif isinstance(stored, TfidfVectorizer):
                self.legacy_shared_vectorizer = stored
                self.vectorizers = {field: stored for field in REQUEST_FIELDS}
            else:
                raise TypeError(f'Unsupported vectorizer payload type: {type(stored)!r}')
        else:
            self.vectorizers = {
                field: TfidfVectorizer(
                    analyzer='char',
                    ngram_range=(2, 5),
                    max_features=FIELD_MAX_FEATURES[field],
                    lowercase=True,
                )
                for field in REQUEST_FIELDS
            }

    def _extract_field_texts(self, df):
        if isinstance(df, pd.Series):
            series = df.fillna('').astype(str)
            return {field: series for field in REQUEST_FIELDS}

        field_texts = {}
        for field in REQUEST_FIELDS:
            if field in df:
                field_texts[field] = df[field].fillna('').astype(str)
            else:
                field_texts[field] = pd.Series('', index=df.index, dtype=str)
        return field_texts

    def get_statistical_features(self, text_series):
        features = pd.DataFrame(index=text_series.index)
        features['length'] = text_series.apply(len) / 1000.0

        def calc_entropy(text):
            if not text:
                return 0
            counts = pd.Series(list(text)).value_counts()
            return entropy(counts) / 10.0

        features['entropy'] = text_series.apply(calc_entropy)

        for kw in SUSPICIOUS_KEYWORDS:
            features[f'kw_{kw}'] = text_series.apply(lambda x: x.count(kw) / (len(x) + 1))

        return features

    def _combine_field_texts(self, field_texts):
        combined = pd.Series('', index=next(iter(field_texts.values())).index, dtype=str)
        for field in REQUEST_FIELDS:
            combined = combined + ' ' + field_texts[field]
        return combined.str.strip()

    def fit(self, df):
        print("Standardizing text for training...")
        from preprocessing import clean_text

        for field, texts in self._extract_field_texts(df).items():
            self.vectorizers[field].fit(texts.apply(clean_text))
        return self

    def transform(self, df):
        from preprocessing import clean_text
        from scipy.sparse import hstack

        field_texts = self._extract_field_texts(df)

        if self.legacy_shared_vectorizer is not None:
            combined_texts = self._combine_field_texts(field_texts)
            return hstack([
                self.legacy_shared_vectorizer.transform(combined_texts.apply(clean_text)),
                self.get_statistical_features(combined_texts).values,
            ])

        tfidf_blocks = []
        stat_blocks = []

        for field in REQUEST_FIELDS:
            texts = field_texts[field]
            tfidf_blocks.append(self.vectorizers[field].transform(texts.apply(clean_text)))
            stat_blocks.append(self.get_statistical_features(texts).values)

        return hstack([*tfidf_blocks, *stat_blocks])

    def save(self, path):
        joblib.dump({'vectorizers': self.vectorizers}, path)


if __name__ == "__main__":
    pass
