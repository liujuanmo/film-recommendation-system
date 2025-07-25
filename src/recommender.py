import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from .feature_engineering import TextFeatureExtractor
from .embedding_utils import build_embedding_dict, get_mean_embedding
from .sqlite_vec_client import get_con, init_sqlite_vec, insert_embeddings, search as sqlite_vec_search
import sqlite3
import json
import pickle

DATA_DIR = 'imdb_data'
EMBED_DIM = 32
MOVIE_INDEX_PATH = "movie_index.pkl"

class MovieRecommender:
    def __init__(self):
        self.titles = None
        self.crew = None
        self.principals = None
        self.names = None
        self.genre_mlb = None
        self.movie_index = None
        self.text_extractor = TextFeatureExtractor(method='tfidf', max_features=200)
        self.text_features = None
        self.director_emb_dict = None
        self.cast_emb_dict = None
        self.movie_vectors = None
        self.sqlite_con = None
        self._load_data()
        self._init_or_load_embeddings()

    def _load_data(self):
        basics_path = os.path.join(DATA_DIR, 'title.basics.tsv')
        crew_path = os.path.join(DATA_DIR, 'title.crew.tsv')
        principals_path = os.path.join(DATA_DIR, 'title.principals.tsv')
        names_path = os.path.join(DATA_DIR, 'name.basics.tsv')
        self.titles = pd.read_csv(basics_path, sep='\t', na_values='\\N',
                                  usecols=['tconst', 'primaryTitle', 'startYear', 'genres', 'titleType'])
        self.crew = pd.read_csv(crew_path, sep='\t', na_values='\\N', usecols=['tconst', 'directors'])
        self.principals = pd.read_csv(principals_path, sep='\t', na_values='\\N', usecols=['tconst', 'nconst', 'category'])
        self.names = pd.read_csv(names_path, sep='\t', na_values='\\N', usecols=['nconst', 'primaryName'])
        if 'overview' not in self.titles.columns:
            self.titles['overview'] = self.titles['primaryTitle']
        if 'keywords' not in self.titles.columns:
            self.titles['keywords'] = self.titles['genres']

    def _save_movie_index(self):
        with open(MOVIE_INDEX_PATH, "wb") as f:
            pickle.dump(self.movie_index, f)

    def _load_movie_index_from_pickle(self):
        with open(MOVIE_INDEX_PATH, "rb") as f:
            self.movie_index = pickle.load(f)

    def _init_or_load_embeddings(self):
        emb_dim = self._get_embedding_dim()
        self.sqlite_con = get_con()
        try:
            cur = self.sqlite_con.execute("SELECT COUNT(*) FROM movie_embeddings")
            count = cur.fetchone()[0]
        except sqlite3.OperationalError:
            init_sqlite_vec(self.sqlite_con, emb_dim)
            count = 0
        if count == 0:
            self._build_features_and_sqlitevec()
            self._save_movie_index()  # 首次保存
        else:
            # 优先用pickle加载
            try:
                self._load_movie_index_from_pickle()
            except Exception:
                self._load_movie_index_only()
                self._save_movie_index()

    def _get_embedding_dim(self):
        # 预加载部分数据用于确定embedding维度
        basics_path = os.path.join(DATA_DIR, 'title.basics.tsv')
        basics = pd.read_csv(basics_path, sep='\t', na_values='\\N', usecols=['tconst', 'primaryTitle', 'startYear', 'genres', 'titleType'])
        basics = basics[basics['titleType'] == 'movie'].dropna(subset=['genres'])
        genre_dim = len(set(
            g for genres in basics['genres'].apply(lambda x: x.split(',')) for g in genres
        ))
        # 1 (year) + 32 (director) + 32 (cast) + 200 (text) + genre_dim
        return genre_dim + 1 + 32 + 32 + 200

    def _load_movie_index_only(self):
        # 只加载movie_index用于id->元数据映射
        basics_path = os.path.join(DATA_DIR, 'title.basics.tsv')
        crew_path = os.path.join(DATA_DIR, 'title.crew.tsv')
        principals_path = os.path.join(DATA_DIR, 'title.principals.tsv')
        names_path = os.path.join(DATA_DIR, 'name.basics.tsv')
        titles = pd.read_csv(basics_path, sep='\t', na_values='\\N', usecols=['tconst', 'primaryTitle', 'startYear', 'genres', 'titleType'])
        crew = pd.read_csv(crew_path, sep='\t', na_values='\\N', usecols=['tconst', 'directors'])
        principals = pd.read_csv(principals_path, sep='\t', na_values='\\N', usecols=['tconst', 'nconst', 'category'])
        names = pd.read_csv(names_path, sep='\t', na_values='\\N', usecols=['nconst', 'primaryName'])
        if 'overview' not in titles.columns:
            titles['overview'] = titles['primaryTitle']
        if 'keywords' not in titles.columns:
            titles['keywords'] = titles['genres']
        movies = titles[titles['titleType'] == 'movie'].copy()
        movies = movies.dropna(subset=['genres'])
        movies['genres'] = movies['genres'].apply(lambda x: x.split(','))
        movies = movies.merge(crew, on='tconst', how='left')
        movies['directors'] = movies['directors'].fillna('').apply(lambda x: x.split(',') if isinstance(x, str) else [])
        principals_actors = principals[principals['category'].isin(['actor', 'actress'])]
        actors_grouped = principals_actors.groupby('tconst')['nconst'].apply(lambda x: list(x)[:5]).reset_index()
        movies = movies.merge(actors_grouped, on='tconst', how='left')
        movies = movies.rename(columns={'nconst': 'cast'})
        movies['cast'] = movies['cast'].apply(lambda x: x if isinstance(x, list) else [])
        nconst_to_name = dict(zip(names['nconst'], names['primaryName']))
        movies['directors'] = movies['directors'].apply(lambda lst: [nconst_to_name.get(n, n) for n in lst if n])
        movies['cast'] = movies['cast'].apply(lambda lst: [nconst_to_name.get(n, n) for n in lst if n])
        self.movie_index = movies.reset_index(drop=True)

    def _build_features_and_sqlitevec(self):
        movies = self.titles[self.titles['titleType'] == 'movie'].copy()
        movies = movies.dropna(subset=['genres'])
        movies['genres'] = movies['genres'].apply(lambda x: x.split(','))
        movies = movies.merge(self.crew, on='tconst', how='left')
        movies['directors'] = movies['directors'].fillna('').apply(lambda x: x.split(',') if isinstance(x, str) else [])
        principals_actors = self.principals[self.principals['category'].isin(['actor', 'actress'])]
        actors_grouped = principals_actors.groupby('tconst')['nconst'].apply(lambda x: list(x)[:5]).reset_index()
        movies = movies.merge(actors_grouped, on='tconst', how='left')
        movies = movies.rename(columns={'nconst': 'cast'})
        movies['cast'] = movies['cast'].apply(lambda x: x if isinstance(x, list) else [])
        nconst_to_name = dict(zip(self.names['nconst'], self.names['primaryName']))
        movies['directors'] = movies['directors'].apply(lambda lst: [nconst_to_name.get(n, n) for n in lst if n])
        movies['cast'] = movies['cast'].apply(lambda lst: [nconst_to_name.get(n, n) for n in lst if n])
        self.genre_mlb = MultiLabelBinarizer()
        genre_features = self.genre_mlb.fit_transform(movies['genres'])
        movies['startYear'] = pd.to_numeric(movies['startYear'], errors='coerce')
        year_norm = (movies['startYear'] - movies['startYear'].min()) / (movies['startYear'].max() - movies['startYear'].min())
        year_features = year_norm.fillna(0).values.reshape(-1, 1)
        text_corpus = (movies['primaryTitle'].fillna('') + ' ' +
                       movies['overview'].fillna('') + ' ' +
                       movies['keywords'].fillna('')).tolist()
        self.text_features = self.text_extractor.fit_transform(text_corpus)
        all_directors = set([d for sublist in movies['directors'] for d in sublist])
        all_cast = set([c for sublist in movies['cast'] for c in sublist])
        self.director_emb_dict = build_embedding_dict(all_directors, dim=EMBED_DIM)
        self.cast_emb_dict = build_embedding_dict(all_cast, dim=EMBED_DIM)
        director_embs = np.array([get_mean_embedding(lst, self.director_emb_dict, dim=EMBED_DIM) for lst in movies['directors']])
        cast_embs = np.array([get_mean_embedding(lst, self.cast_emb_dict, dim=EMBED_DIM) for lst in movies['cast']])
        self.movie_vectors = np.hstack([
            genre_features,
            year_features,
            director_embs,
            cast_embs,
            self.text_features
        ]).astype(np.float32)
        self.movie_index = movies.reset_index(drop=True)
        self.sqlite_con = init_sqlite_vec(self.movie_vectors.shape[1])
        ids = list(range(len(self.movie_index)))
        insert_embeddings(self.sqlite_con, ids, self.movie_vectors.tolist())

    def recommend(self, genres=None, year=None, directors=None, cast=None, keywords=None, overview=None, title=None, top_n=10):
        user_genres = genres if genres else []
        user_genre_vec = self.genre_mlb.transform([user_genres]) if user_genres else np.zeros((1, self.genre_mlb.classes_.shape[0]), dtype=np.float32)
        if year:
            year_val = float(year)
            year_norm = (year_val - self.movie_index['startYear'].min()) / (self.movie_index['startYear'].max() - self.movie_index['startYear'].min())
            user_year_vec = np.array([[year_norm]], dtype=np.float32)
        else:
            user_year_vec = np.zeros((1, 1), dtype=np.float32)
        user_directors = directors if directors else []
        user_director_vec = np.array([get_mean_embedding(user_directors, self.director_emb_dict, dim=EMBED_DIM)], dtype=np.float32)
        user_cast = cast if cast else []
        user_cast_vec = np.array([get_mean_embedding(user_cast, self.cast_emb_dict, dim=EMBED_DIM)], dtype=np.float32)
        user_text = ''
        if title:
            user_text += title + ' '
        if overview:
            user_text += overview + ' '
        if keywords:
            user_text += ' '.join(keywords) + ' '
        if user_text.strip():
            user_text_vec = self.text_extractor.transform([user_text]).astype(np.float32)
        else:
            user_text_vec = np.zeros((1, self.text_features.shape[1]), dtype=np.float32)
        user_vec = np.hstack([
            user_genre_vec,
            user_year_vec,
            user_director_vec,
            user_cast_vec,
            user_text_vec
        ]).astype(np.float32)[0]
        results = sqlite_vec_search(self.sqlite_con, user_vec, top_n=top_n)
        out = []
        for movie_id, distance in results:
            if movie_id < 0 or movie_id >= len(self.movie_index):
                continue  # 跳过无效索引
            row = self.movie_index.iloc[movie_id]
            out.append({
                'title': row['primaryTitle'],
                'year': int(row['startYear']) if not pd.isna(row['startYear']) else 'N/A',
                'genres': ','.join(row['genres']),
                'directors': ','.join(row['directors']),
                'cast': ','.join(row['cast'])
            })
        return out 