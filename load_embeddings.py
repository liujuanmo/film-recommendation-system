#!/usr/bin/env python3
"""
Embeddings Loader for Movie Recommendation System

This script computes movie embeddings from PostgreSQL data and stores them back.
Run this script after loading movie data with load_data.py.

Usage:
    python load_embeddings.py [--force]

Options:
    --force    Force recomputation even if embeddings already exist

Environment Variables:
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
"""

import argparse
import sys
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

from src.postgresql_vec_client import (
    get_engine, init_embeddings_table, insert_embeddings, 
    get_movies_with_details, table_exists, get_embedding_count,
    store_transformer_metadata, store_person_embeddings, init_postgresql_tables
)
from src.feature_engineering import TextFeatureExtractor
from src.embedding_utils import build_embedding_dict, get_mean_embedding

EMBED_DIM = 32  # Dimension for director and cast embeddings

def check_prerequisites():
    """Check if movie data is loaded in PostgreSQL."""
    if not table_exists('movies'):
        print("❌ Error: Movie data not found in PostgreSQL!")
        print("Please run 'python load_data.py' first to load IMDB data.")
        return False
    
    # Check if we have movies - use SQLAlchemy
    from src.postgresql_vec_client import get_session, Movie
    session = get_session()
    try:
        movie_count = session.query(Movie).filter(
            Movie.title_type == 'movie',
            Movie.genres != None
        ).count()
    finally:
        session.close()
    
    if movie_count == 0:
        print("❌ Error: No valid movies found in database!")
        print("Please run 'python load_data.py' first to load IMDB data.")
        return False
    
    print(f"✅ Found {movie_count} movies in database")
    return True

def check_existing_embeddings():
    """Check if embeddings already exist."""
    if not table_exists('movie_embeddings'):
        return 0
    
    count = get_embedding_count()
    return count

def ensure_metadata_tables():
    """Ensure transformer metadata and person embedding tables exist."""
    print("🔧 Ensuring metadata tables exist...")
    try:
        # This will create all tables including the new metadata tables
        init_postgresql_tables()
        print("✅ Metadata tables verified/created")
    except Exception as e:
        print(f"❌ Error creating metadata tables: {e}")
        raise

def compute_and_store_embeddings(force=False):
    """Compute movie embeddings and store them in PostgreSQL."""
    
    # Check existing embeddings
    existing_count = check_existing_embeddings()
    if existing_count > 0 and not force:
        print(f"⚠️  Found {existing_count} existing embeddings in database.")
        response = input("Do you want to recompute embeddings? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Skipping embedding computation. Use --force to override.")
            return False
    
    print("🔄 Loading movie data from PostgreSQL...")
    movies_data = get_movies_with_details()
    
    if not movies_data:
        print("❌ No movie data found!")
        return False
    
    print(f"📊 Processing {len(movies_data)} movies for embedding computation...")
    
    # Convert data for processing
    print("  → Extracting features...")
    genres_list = [movie['genres'] for movie in movies_data]
    directors_list = [movie['directors'] for movie in movies_data]
    cast_list = [movie['cast'] for movie in movies_data]
    years_list = [movie['start_year'] for movie in movies_data]
    
    # 1. Genre features
    print("  → Computing genre embeddings...")
    genre_mlb = MultiLabelBinarizer()
    genre_features = genre_mlb.fit_transform(genres_list)
    print(f"     Found {len(genre_mlb.classes_)} unique genres")
    
    # 2. Year features (normalized)
    print("  → Computing year features...")
    years_array = np.array([y if y is not None else 0 for y in years_list])
    year_min, year_max = years_array.min(), years_array.max()
    if year_max > year_min:
        year_norm = (years_array - year_min) / (year_max - year_min)
    else:
        year_norm = np.zeros(len(years_array))
    year_features = year_norm.reshape(-1, 1)
    print(f"     Year range: {year_min} - {year_max}")
    
    # 3. Text features
    print("  → Computing text features...")
    text_corpus = []
    for movie in movies_data:
        title = movie['primary_title'] or ''
        overview = movie['overview'] or ''
        keywords = ' '.join(movie['keywords']) if movie['keywords'] else ''
        text_corpus.append(f"{title} {overview} {keywords}")
    
    text_extractor = TextFeatureExtractor(method='tfidf', max_features=200)
    text_features = text_extractor.fit_transform(text_corpus)
    print(f"     Text feature dimension: {text_features.shape[1]}")
    
    # 4. Director embeddings
    print("  → Computing director embeddings...")
    all_directors = set()
    for directors in directors_list:
        if directors:
            all_directors.update(directors)
    
    director_emb_dict = build_embedding_dict(all_directors, dim=EMBED_DIM)
    director_embs = np.array([
        get_mean_embedding(directors, director_emb_dict, dim=EMBED_DIM) 
        for directors in directors_list
    ])
    print(f"     Processed {len(all_directors)} unique directors")
    
    # 5. Cast embeddings
    print("  → Computing cast embeddings...")
    all_cast = set()
    for cast in cast_list:
        if cast:
            all_cast.update(cast)
    
    cast_emb_dict = build_embedding_dict(all_cast, dim=EMBED_DIM)
    cast_embs = np.array([
        get_mean_embedding(cast, cast_emb_dict, dim=EMBED_DIM) 
        for cast in cast_list
    ])
    print(f"     Processed {len(all_cast)} unique actors")
    
    # 6. Combine all features
    print("  → Combining all features...")
    movie_vectors = np.hstack([
        genre_features,      # Genre one-hot vectors
        year_features,       # Normalized year
        director_embs,       # Director embeddings
        cast_embs,          # Cast embeddings  
        text_features       # Text features (TF-IDF)
    ]).astype(np.float32)
    
    total_dim = movie_vectors.shape[1]
    print(f"     Final embedding dimension: {total_dim}")
    print(f"       - Genres: {genre_features.shape[1]}")
    print(f"       - Year: {year_features.shape[1]}")
    print(f"       - Directors: {director_embs.shape[1]}")
    print(f"       - Cast: {cast_embs.shape[1]}")
    print(f"       - Text: {text_features.shape[1]}")
    
    # 7. Initialize embeddings table
    print("🗄️  Initializing embeddings table...")
    init_embeddings_table(total_dim)
    
    # 8. Store embeddings
    print("💾 Storing embeddings in PostgreSQL...")
    movie_ids = [movie['id'] for movie in movies_data]
    
    # Store in batches for better performance
    batch_size = 1000
    total_batches = (len(movie_ids) + batch_size - 1) // batch_size
    
    for i in range(0, len(movie_ids), batch_size):
        batch_ids = movie_ids[i:i+batch_size]
        batch_vectors = movie_vectors[i:i+batch_size]
        
        insert_embeddings(batch_ids, batch_vectors.tolist())
        
        batch_num = (i // batch_size) + 1
        print(f"   Batch {batch_num}/{total_batches} completed ({len(batch_ids)} embeddings)")
    
    # 9. Store transformer metadata for database-first operations
    print("🗄️  Storing transformer metadata...")
    
    # Store genre classes
    store_transformer_metadata('genre_classes', list(genre_mlb.classes_))
    print("   ✓ Stored genre classes")
    
    # Store year statistics
    store_transformer_metadata('year_stats', {
        'min': float(year_min),
        'max': float(year_max)
    })
    print("   ✓ Stored year normalization parameters")
    
    # Store TF-IDF metadata
    tfidf_metadata = {
        'vocabulary': text_extractor.vectorizer.get_feature_names_out().tolist(),
        'idf_weights': text_extractor.vectorizer.idf_.tolist()
    }
    store_transformer_metadata('tfidf_metadata', tfidf_metadata)
    print("   ✓ Stored TF-IDF metadata")
    
    # Store person embeddings in database
    store_person_embeddings(director_emb_dict, 'director')
    print(f"   ✓ Stored {len(director_emb_dict)} director embeddings")
    
    store_person_embeddings(cast_emb_dict, 'actor')
    print(f"   ✓ Stored {len(cast_emb_dict)} actor embeddings")
    
    print("✅ Embedding computation and storage completed!")
    print(f"📊 Total embeddings stored: {len(movie_ids)}")
    print(f"🎯 Embedding dimension: {total_dim}")
    
    return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Compute and load movie embeddings into PostgreSQL")
    parser.add_argument('--force', action='store_true', 
                       help='Force recomputation even if embeddings already exist')
    args = parser.parse_args()
    
    print("Movie Embeddings Loader")
    print("=" * 50)
    
    try:
        # Connect to PostgreSQL
        print("🔌 Connecting to PostgreSQL...")
        engine = get_engine()
        print("✅ Connected to PostgreSQL")
        
        # Check prerequisites
        if not check_prerequisites():
            return 1
        
        # Ensure metadata tables exist
        ensure_metadata_tables()

        # Compute and store embeddings
        success = compute_and_store_embeddings(force=args.force)
        
        if success:
            print("\n🎉 Embeddings loading completed successfully!")
            print("\nNext steps:")
            print("1. Start the FastAPI server: python main.py")
            print("2. Visit http://localhost:8000/docs for the API documentation")
        else:
            print("\n❌ Embeddings loading was cancelled or failed.")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease check:")
        print("1. PostgreSQL is running")
        print("2. Database 'movie_recommendations' exists")
        print("3. Movie data has been loaded (run 'python load_data.py')")
        print("4. Database connection settings are correct")
        return 1

if __name__ == '__main__':
    sys.exit(main()) 