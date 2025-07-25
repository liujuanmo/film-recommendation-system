import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class NeighborhoodRecommender:
    def __init__(self, movie_features, movie_index):
        self.movie_features = movie_features
        self.movie_index = movie_index

    def get_similar_movies(self, movie_idx, top_k=10):
        # 计算与指定电影的相似度
        target_vec = self.movie_features[movie_idx].reshape(1, -1)
        sim_scores = cosine_similarity(target_vec, self.movie_features)[0]
        # 排除自身
        sim_scores[movie_idx] = -1
        top_idx = np.argsort(sim_scores)[-top_k:][::-1]
        return self.movie_index.iloc[top_idx] 