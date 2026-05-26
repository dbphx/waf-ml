import os
import re
from collections import Counter
from math import log

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

SUSPICIOUS_KEYWORDS = [
    'select', 'union', 'insert', 'update', 'delete', 'drop', 'from', 'where',
    'script', 'alert(', 'onerror', 'eval', '../', './',
    '${', '{{', '}}', '() {', ';', '|', '&',
    '$gt', '$ne', '$in', 'cat ', 'whoami'
]
DEFAULT_KEYWORD_MATCH_MODES = {
    'eval': 'token',
}

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
        self.keyword_match_modes = {}
        if vectorizer_path and os.path.exists(vectorizer_path):
            stored = joblib.load(vectorizer_path)
            if isinstance(stored, dict) and 'vectorizers' in stored:
                self.vectorizers = stored['vectorizers']
                self.keyword_match_modes = dict(stored.get('keyword_match_modes', {}))
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
                    dtype=np.float32,
                )
                for field in REQUEST_FIELDS
            }
            self.keyword_match_modes = dict(DEFAULT_KEYWORD_MATCH_MODES)

    def _count_keyword_occurrences(self, text, keyword):
        mode = self.keyword_match_modes.get(keyword, 'substring')
        if mode == 'token':
            pattern = rf'(?<![a-z]){re.escape(keyword)}(?![a-z])'
            return len(re.findall(pattern, text))
        return text.count(keyword)

    def _keyword_pattern(self, keyword):
        mode = self.keyword_match_modes.get(keyword, 'substring')
        if mode == 'token':
            return rf'(?<![a-z]){re.escape(keyword)}(?![a-z])'
        return re.escape(keyword)

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

    def _normalize_series(self, text_series):
        return text_series.fillna('').astype(str)

    def _clean_text_series(self, text_series):
        from preprocessing import clean_text

        normalized = self._normalize_series(text_series)
        codes, uniques = pd.factorize(normalized, sort=False)
        cleaned_uniques = pd.Index([clean_text(text) for text in uniques], dtype=object)
        return pd.Series(cleaned_uniques.take(codes), index=normalized.index, dtype=str)

    def _transform_unique_texts(self, vectorizer, cleaned_text_series):
        normalized = self._normalize_series(cleaned_text_series)
        codes, uniques = pd.factorize(normalized, sort=False)
        unique_matrix = vectorizer.transform(uniques.tolist())
        return unique_matrix[codes]

    def _calc_entropy(self, text):
        if not text:
            return 0.0

        total = len(text)
        ent = 0.0
        for count in Counter(text).values():
            p = count / total
            ent -= p * log(p)
        return ent / 10.0

    def prepare(self, df):
        field_texts = self._extract_field_texts(df)
        cleaned_field_texts = {}
        stat_blocks = {}

        for field, texts in field_texts.items():
            normalized = self._normalize_series(texts)
            cleaned_field_texts[field] = self._clean_text_series(normalized)
            stat_blocks[field] = self.get_statistical_features(normalized).values.astype(np.float32, copy=False)

        return {
            'field_texts': field_texts,
            'cleaned_field_texts': cleaned_field_texts,
            'stat_blocks': stat_blocks,
        }

    def _coerce_prepared(self, df_or_prepared):
        if isinstance(df_or_prepared, dict) and 'cleaned_field_texts' in df_or_prepared:
            return df_or_prepared
        return self.prepare(df_or_prepared)

    def get_statistical_features(self, text_series):
        text_series = self._normalize_series(text_series)
        features = pd.DataFrame(index=text_series.index)
        codes, uniques = pd.factorize(text_series, sort=False)
        lengths = np.fromiter((len(text) for text in uniques), dtype=np.float64, count=len(uniques))

        features['length'] = lengths.take(codes) / 1000.0
        entropies = np.fromiter((self._calc_entropy(text) for text in uniques), dtype=np.float64, count=len(uniques))
        features['entropy'] = entropies.take(codes)

        for kw in SUSPICIOUS_KEYWORDS:
            pattern = self._keyword_pattern(kw)
            keyword_counts = np.fromiter(
                (len(re.findall(pattern, text)) for text in uniques),
                dtype=np.float64,
                count=len(uniques),
            )
            features[f'kw_{kw}'] = keyword_counts.take(codes) / (lengths.take(codes) + 1.0)

        return features

    def _combine_field_texts(self, field_texts):
        combined = pd.Series('', index=next(iter(field_texts.values())).index, dtype=str)
        for field in REQUEST_FIELDS:
            combined = combined + ' ' + field_texts[field]
        return combined.str.strip()

    def fit(self, df_or_prepared):
        print("Standardizing text for training...")
        prepared = self._coerce_prepared(df_or_prepared)

        for field, texts in prepared['cleaned_field_texts'].items():
            self.vectorizers[field].fit(texts)
        return self

    def transform(self, df_or_prepared):
        from scipy.sparse import hstack

        prepared = self._coerce_prepared(df_or_prepared)
        field_texts = prepared['field_texts']
        cleaned_field_texts = prepared['cleaned_field_texts']
        stat_blocks = prepared['stat_blocks']

        if self.legacy_shared_vectorizer is not None:
            combined_texts = self._combine_field_texts(field_texts)
            cleaned_combined = self._clean_text_series(combined_texts)
            return hstack([
                self.legacy_shared_vectorizer.transform(cleaned_combined),
                self.get_statistical_features(combined_texts).values,
            ], format='csr', dtype=np.float32)

        tfidf_blocks = []

        for field in REQUEST_FIELDS:
            tfidf_blocks.append(self._transform_unique_texts(self.vectorizers[field], cleaned_field_texts[field]))

        return hstack(
            [*tfidf_blocks, *(stat_blocks[field] for field in REQUEST_FIELDS)],
            format='csr',
            dtype=np.float32,
        )

    def save(self, path):
        joblib.dump(
            {
                'vectorizers': self.vectorizers,
                'keyword_match_modes': self.keyword_match_modes,
            },
            path,
        )


if __name__ == "__main__":
    pass
