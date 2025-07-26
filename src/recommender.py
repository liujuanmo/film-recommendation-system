import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from .feature_engineering import TextFeatureExtractor
from .embedding_utils import build_embedding_dict, get_mean_embedding
from .postgresql_vec_client import (
    get_connection, init_embeddings_table, insert_embeddings, 
    search as postgresql_vec_search, get_embedding_count, table_exists,
    get_movies_with_details, get_movie_count
)
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
        self.pg_con = None
        
        # Establish PostgreSQL connection first
        try:
            self.pg_con = get_connection()
        except Exception as e:
            raise Exception(
                f"Failed to connect to PostgreSQL: {e}\n"
                "Please ensure:\n"
                "1. PostgreSQL is running\n"
                "2. Database 'movie_recommendations' exists\n"
                "3. Connection settings are correct (check environment variables)"
            )
        
        self._load_data()
        self._init_or_load_embeddings()

    def _load_data(self):
        """Load movie data from PostgreSQL instead of CSV files."""
        print("Loading movie data from PostgreSQL...")
        
        # Check if we have movie data in PostgreSQL
        if not table_exists(self.pg_con, 'movies'):
            raise Exception(
                "No movie data found in PostgreSQL!\n"
                "Please run 'python load_data.py' first to load IMDB data into the database."
            )
        
        movie_count = get_movie_count(self.pg_con)
        if movie_count == 0:
            raise Exception(
                "Movie table exists but is empty!\n"
                "Please run 'python load_data.py' first to load IMDB data into the database."
            )
        
        print(f"Found {movie_count} movies in database.")
        
        # We'll load data when needed for embeddings computation
        # No need to preload all CSV data since it's now in PostgreSQL

    def _save_movie_index(self):
        with open(MOVIE_INDEX_PATH, "wb") as f:
            pickle.dump(self.movie_index, f)

    def _load_movie_index_from_pickle(self):
        with open(MOVIE_INDEX_PATH, "rb") as f:
            self.movie_index = pickle.load(f)

    def _init_or_load_embeddings(self):
        # Check if embeddings table exists and has data
        if not table_exists(self.pg_con, 'movie_embeddings'):
            raise Exception(
                "Movie embeddings table not found!\n"
                "Please run 'python load_embeddings.py' to compute and store embeddings first."
            )
        
        count = get_embedding_count(self.pg_con)
        if count == 0:
            raise Exception(
                "Movie embeddings table exists but is empty!\n"
                "Please run 'python load_embeddings.py' to compute and store embeddings first."
            )
        
        print(f"Found {count} movie embeddings in database.")
                
        # Load movie index for fast lookups
        try:
            self._load_movie_index_from_pickle()
            print("Loaded movie index from cache.")
        except Exception:
            print("Loading movie index from database...")
            self._load_movie_index_from_db()
            self._save_movie_index()
        
        # Initialize transformers for query processing
        print("Initializing feature transformers...")
        self._init_transformers()

    def _get_embedding_dim(self, movies_data):
        """Calculate embedding dimension based on actual movie data."""
        # Extract all genres from the movies
        all_genres = set()
        for movie in movies_data:
            if movie['genres']:
                all_genres.update(movie['genres'])
        
        genre_dim = len(all_genres)
        # 1 (year) + 32 (director) + 32 (cast) + 200 (text) + genre_dim
        return genre_dim + 1 + 32 + 32 + 200

    def _load_movie_index_from_db(self):
        """Load movie index from PostgreSQL database."""
        movies_data = get_movies_with_details(self.pg_con)
        
        # Convert to pandas DataFrame for compatibility
        movies_list = []
        for movie in movies_data:
            movies_list.append({
                'id': movie['id'],
                'tconst': movie['tconst'],
                'primaryTitle': movie['primary_title'],
                'startYear': movie['start_year'],
                'genres': movie['genres'],
                'titleType': movie['title_type'],
                'overview': movie['overview'] or movie['primary_title'],
                'keywords': movie['keywords'] or movie['genres'],
                'directors': movie['directors'],
                'cast': movie['cast']
            })
        
        self.movie_index = pd.DataFrame(movies_list)

    def _init_transformers(self):
        """Initialize feature transformers needed for query processing."""
        print("  → Loading movie data for transformer fitting...")
        movies_data = get_movies_with_details(self.pg_con)
        
        if not movies_data:
            raise Exception("No movie data found for transformer initialization!")
        
        # Extract data for fitting transformers
        genres_list = [movie['genres'] for movie in movies_data]
        directors_list = [movie['directors'] for movie in movies_data]
        cast_list = [movie['cast'] for movie in movies_data]
        years_list = [movie['start_year'] for movie in movies_data]
        
        # 1. Fit genre MultiLabelBinarizer
        print("  → Fitting genre transformer...")
        self.genre_mlb = MultiLabelBinarizer()
        self.genre_mlb.fit(genres_list)
        
        # 2. Store year normalization parameters
        print("  → Computing year normalization parameters...")
        years_array = np.array([y if y is not None else 0 for y in years_list])
        self.year_min, self.year_max = years_array.min(), years_array.max()
        
        # 3. Fit text feature extractor
        print("  → Fitting text feature extractor...")
        text_corpus = []
        for movie in movies_data:
            title = movie['primary_title'] or ''
            overview = movie['overview'] or ''
            keywords = ' '.join(movie['keywords']) if movie['keywords'] else ''
            text_corpus.append(f"{title} {overview} {keywords}")
        
        self.text_features = self.text_extractor.fit_transform(text_corpus)
        
        # 4. Build director embeddings dictionary
        print("  → Building director embeddings...")
        all_directors = set()
        for directors in directors_list:
            if directors:
                all_directors.update(directors)
        self.director_emb_dict = build_embedding_dict(all_directors, dim=EMBED_DIM)
        
        # 5. Build cast embeddings dictionary
        print("  → Building cast embeddings...")
        all_cast = set()
        for cast in cast_list:
            if cast:
                all_cast.update(cast)
        self.cast_emb_dict = build_embedding_dict(all_cast, dim=EMBED_DIM)
        
        print("  → Feature transformers initialized successfully!")

    def recommend(self, genres=None, year=None, directors=None, cast=None, keywords=None, overview=None, title=None, top_n=10):
        user_genres = genres if genres else []
        user_genre_vec = self.genre_mlb.transform([user_genres]) if user_genres else np.zeros((1, self.genre_mlb.classes_.shape[0]), dtype=np.float32)
        if year:
            year_val = float(year)
            if self.year_max > self.year_min:
                year_norm = (year_val - self.year_min) / (self.year_max - self.year_min)
            else:
                year_norm = 0.0
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
        results = postgresql_vec_search(self.pg_con, user_vec, top_n=top_n)
        out = []
        for movie_id, distance in results:
            # movie_id is now the database ID, find corresponding row in movie_index
            row = self.movie_index[self.movie_index['id'] == movie_id]
            if row.empty:
                continue  # Skip if movie not found in index
            row = row.iloc[0]
            
            out.append({
                'title': row['primaryTitle'],
                'year': str(int(row['startYear'])) if pd.notna(row['startYear']) else 'N/A',
                'genres': ','.join(row['genres']) if row['genres'] else 'N/A',
                'directors': ','.join(row['directors']) if row['directors'] else 'N/A',
                'cast': ','.join(row['cast']) if row['cast'] else 'N/A'
            })
        return out 