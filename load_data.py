#!/usr/bin/env python3
"""
Data Loader for Movie Recommendation System

This script loads IMDB data into PostgreSQL tables.
Run this script first before using the recommendation system.

Usage:
    python load_data.py

Environment Variables:
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
"""

import os
import pandas as pd
from src.postgresql_vec_client import (
    get_engine, init_postgresql_tables, 
    load_movies_data, load_directors_data, load_actors_data,
    load_movie_directors, load_movie_actors,
    table_exists, get_movie_count
)

DATA_DIR = 'imdb_data'

def load_imdb_data():
    """Load all IMDB data into PostgreSQL."""
    print("Connecting to PostgreSQL...")
    engine = get_engine()
    
    print("Initializing database tables...")
    init_postgresql_tables()
    
    # Check if data is already loaded
    if table_exists('movies'):
        movie_count = get_movie_count()
        if movie_count > 0:
            print(f"Found {movie_count} movies already loaded in database.")
            response = input("Do you want to reload all data? (y/N): ").strip().lower()
            if response != 'y':
                print("Skipping data load. Use existing data.")
                return
    
    print("Loading IMDB data files...")
    
    # Load basic movie information
    print("  → Loading title.basics.tsv...")
    basics_path = os.path.join(DATA_DIR, 'title.basics.tsv')
    if not os.path.exists(basics_path):
        print(f"Error: {basics_path} not found!")
        print("Please ensure IMDB data files are in the imdb_data/ directory.")
        return
    
    titles = pd.read_csv(basics_path, sep='\t', na_values='\\N',
                        usecols=['tconst', 'primaryTitle', 'startYear', 'genres', 'titleType'])
    
    # Filter for movies only and clean data
    movies = titles[titles['titleType'] == 'movie'].copy()
    movies = movies.dropna(subset=['genres'])
    movies['genres'] = movies['genres'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
    
    # Add placeholder columns for overview and keywords if not present
    if 'overview' not in movies.columns:
        movies['overview'] = movies['primaryTitle']
    if 'keywords' not in movies.columns:
        movies['keywords'] = movies['genres']  # Use genres as keywords placeholder
    
    print(f"  → Found {len(movies)} movies to load")
    
    # Load crew information
    print("  → Loading title.crew.tsv...")
    crew_path = os.path.join(DATA_DIR, 'title.crew.tsv')
    if not os.path.exists(crew_path):
        print(f"Warning: {crew_path} not found! Directors will not be available.")
        crew = pd.DataFrame(columns=['tconst', 'directors'])
    else:
        crew = pd.read_csv(crew_path, sep='\t', na_values='\\N', usecols=['tconst', 'directors'])
    
    # Load principals (actors/actresses)
    print("  → Loading title.principals.tsv...")
    principals_path = os.path.join(DATA_DIR, 'title.principals.tsv')
    if not os.path.exists(principals_path):
        print(f"Warning: {principals_path} not found! Cast information will not be available.")
        principals = pd.DataFrame(columns=['tconst', 'nconst', 'category'])
    else:
        principals = pd.read_csv(principals_path, sep='\t', na_values='\\N', 
                               usecols=['tconst', 'nconst', 'category'])
    
    # Load names
    print("  → Loading name.basics.tsv...")
    names_path = os.path.join(DATA_DIR, 'name.basics.tsv')
    if not os.path.exists(names_path):
        print(f"Warning: {names_path} not found! Names will not be resolved.")
        names = pd.DataFrame(columns=['nconst', 'primaryName'])
    else:
        names = pd.read_csv(names_path, sep='\t', na_values='\\N', 
                          usecols=['nconst', 'primaryName'])
    
    print("\nLoading data into PostgreSQL...")
    
    # Load movies
    print("  → Inserting movies...")
    load_movies_data(movies)
    
    # Process and load directors
    if not crew.empty and not names.empty:
        print("  → Processing directors...")
        
        # Get all director nconsts
        director_nconsts = set()
        for _, row in crew.iterrows():
            if pd.notna(row['directors']) and row['directors']:
                directors_list = row['directors'].split(',')
                director_nconsts.update(directors_list)
        
        # Get director names
        director_names = names[names['nconst'].isin(director_nconsts)].copy()
        if not director_names.empty:
            print(f"  → Inserting {len(director_names)} directors...")
            load_directors_data(director_names)
            
            # Create movie-director relationships
            print("  → Creating movie-director relationships...")
            movie_directors_data = []
            for _, row in crew.iterrows():
                if pd.notna(row['directors']) and row['directors']:
                    directors_list = row['directors'].split(',')
                    for director_nconst in directors_list:
                        if director_nconst.strip():
                            movie_directors_data.append((row['tconst'], director_nconst.strip()))
            
            if movie_directors_data:
                load_movie_directors(movie_directors_data)
    
    # Process and load actors
    if not principals.empty and not names.empty:
        print("  → Processing actors...")
        
        # Filter for actors and actresses
        actors_principals = principals[principals['category'].isin(['actor', 'actress'])].copy()
        
        # Get unique actor nconsts
        actor_nconsts = set(actors_principals['nconst'].dropna())
        
        # Get actor names
        actor_names = names[names['nconst'].isin(actor_nconsts)].copy()
        if not actor_names.empty:
            print(f"  → Inserting {len(actor_names)} actors...")
            load_actors_data(actor_names)
            
            # Create movie-actor relationships (limit to top 5 per movie)
            print("  → Creating movie-actor relationships...")
            movie_actors_data = []
            actors_grouped = actors_principals.groupby('tconst')['nconst'].apply(
                lambda x: list(x)[:5]  # Limit to first 5 actors per movie
            ).reset_index()
            
            for _, row in actors_grouped.iterrows():
                for actor_nconst in row['nconst']:
                    if actor_nconst and pd.notna(actor_nconst):
                        movie_actors_data.append((row['tconst'], actor_nconst))
            
            if movie_actors_data:
                load_movie_actors(movie_actors_data)
    
    print("\nData loading completed!")
    
    # Show summary
    final_count = get_movie_count()
    print(f"Total movies loaded: {final_count}")
    
    print("Database operations completed.")
    print("\nNext step: Run 'python main.py' to start the recommendation system.")

if __name__ == '__main__':
    try:
        load_imdb_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        print("\nPlease check:")
        print("1. PostgreSQL is running")
        print("2. Database 'movie_recommendations' exists")
        print("3. IMDB data files are in imdb_data/ directory")
        print("4. Database connection settings are correct") 