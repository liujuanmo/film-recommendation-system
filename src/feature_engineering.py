import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
# 可选: from sentence_transformers import SentenceTransformer

class TextFeatureExtractor:
    def __init__(self, method='tfidf', max_features=200):
        self.method = method
        self.max_features = max_features
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        # elif method == 'bert':
        #     self.model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            raise ValueError('Unknown method')

    def fit_transform(self, texts):
        if self.method == 'tfidf':
            return self.vectorizer.fit_transform(texts).toarray()
        # elif self.method == 'bert':
        #     return self.model.encode(texts, show_progress_bar=True)

    def transform(self, texts):
        if self.method == 'tfidf':
            return self.vectorizer.transform(texts).toarray()
        # elif self.method == 'bert':
        #     return self.model.encode(texts, show_progress_bar=False) 