import argparse
import sys
import os
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

from src.db_client import (
    get_engine,
    init_embeddings_table,
    insert_embeddings,
    get_session,
    Movie,
    get_movies_with_details,
    table_exists,
    get_embedding_count,
    store_transformer_metadata,
    store_person_embeddings,
    init_postgresql_tables,
    init_person_embeddings_table,
)
from src.feature_engineering import (
    TextFeatureExtractor,
    build_embedding_dict,
    get_mean_embedding,
)
from src.constants import DEFAULT_MODEL


def ensure_metadata_tables():
    """Ensure transformer metadata and person embedding tables exist."""
    print("🔧 Ensuring metadata tables exist...")
    try:
        # This will create all tables including the new metadata tables
        init_postgresql_tables()
        print("✅ Metadata tables verified/created")

        # Ensure person embeddings table has correct dimensions for sentence transformers
        init_person_embeddings_table()

    except Exception as e:
        print(f"❌ Error creating metadata tables: {e}")
        raise


def extract_movie_features(movies_data):
    """Extract and prepare movie features for embedding computation."""
    print("  → Extracting features...")

    genres_list = [movie["genres"] for movie in movies_data]
    directors_list = [movie["directors"] for movie in movies_data]
    cast_list = [movie["cast"] for movie in movies_data]
    years_list = [movie["start_year"] for movie in movies_data]

    return genres_list, directors_list, cast_list, years_list


def compute_genre_features(genres_list):
    """Compute genre one-hot encoding features."""
    print("  → Computing genre embeddings...")
    genre_mlb = MultiLabelBinarizer()
    genre_features = genre_mlb.fit_transform(genres_list)
    print(f"     Found {len(genre_mlb.classes_)} unique genres")
    return genre_features, genre_mlb


def compute_year_features(years_list):
    """Compute normalized year features."""
    print("  → Computing year features...")
    years_array = np.array([y if y is not None else 0 for y in years_list])
    year_min, year_max = years_array.min(), years_array.max()

    if year_max > year_min:
        year_norm = (years_array - year_min) / (year_max - year_min)
    else:
        year_norm = np.zeros(len(years_array))

    year_features = year_norm.reshape(-1, 1)
    print(f"     Year range: {year_min} - {year_max}")

    return year_features, year_min, year_max


def compute_text_features(movies_data):
    """Compute text features using sentence transformers."""
    print(f"  → Computing text features...")

    text_corpus = []
    for movie in movies_data:
        title = movie["primary_title"] or ""
        overview = movie["overview"] or ""
        keywords = " ".join(movie["keywords"]) if movie["keywords"] else ""
        text_corpus.append(f"{title} {overview} {keywords}")

    text_extractor = TextFeatureExtractor()
    text_features = text_extractor.fit_transform(text_corpus)
    print(
        f"     Text feature dimension: {text_features.shape[1]} (sentence transformer)"
    )

    return text_features


def compute_person_embeddings(directors_list, cast_list):
    """Compute semantic embeddings for directors and cast."""
    # Director embeddings
    print("  → Computing semantic director embeddings...")
    all_directors = set()
    for directors in directors_list:
        if directors:
            all_directors.update(directors)

    director_emb_dict = build_embedding_dict(all_directors)
    director_embs = np.array(
        [
            get_mean_embedding(directors, director_emb_dict)
            for directors in directors_list
        ]
    )
    print(
        f"     Processed {len(all_directors)} unique directors with semantic embeddings"
    )

    # Cast embeddings
    print("  → Computing semantic cast embeddings...")
    all_cast = set()
    for cast in cast_list:
        if cast:
            all_cast.update(cast)

    cast_emb_dict = build_embedding_dict(all_cast)
    cast_embs = np.array(
        [get_mean_embedding(cast, cast_emb_dict) for cast in cast_list]
    )
    print(f"     Processed {len(all_cast)} unique actors with semantic embeddings")

    return director_embs, cast_embs, director_emb_dict, cast_emb_dict


def combine_features(
    genre_features, year_features, director_embs, cast_embs, text_features
):
    """Combine all features into final movie vectors."""
    print("  → Combining all features...")

    movie_vectors = np.hstack(
        [
            genre_features,  # Genre one-hot vectors
            year_features,  # Normalized year
            director_embs,  # Director embeddings
            cast_embs,  # Cast embeddings
            text_features,  # Text features (sentence transformer)
        ]
    ).astype(np.float32)

    total_dim = movie_vectors.shape[1]
    print(f"     Final embedding dimension: {total_dim}")
    print(f"       - Genres: {genre_features.shape[1]}")
    print(f"       - Year: {year_features.shape[1]}")
    print(f"       - Directors: {director_embs.shape[1]}")
    print(f"       - Cast: {cast_embs.shape[1]}")
    print(f"       - Text: {text_features.shape[1]}")

    return movie_vectors, total_dim


def store_embeddings_in_batches(movie_ids, movie_vectors):
    """Store embeddings in PostgreSQL in batches."""
    print("💾 Storing embeddings in PostgreSQL...")

    # Store in batches for better performance
    batch_size = 1000
    total_batches = (len(movie_ids) + batch_size - 1) // batch_size

    for i in range(0, len(movie_ids), batch_size):
        batch_ids = movie_ids[i : i + batch_size]
        batch_vectors = movie_vectors[i : i + batch_size]

        insert_embeddings(batch_ids, batch_vectors.tolist())

        batch_num = (i // batch_size) + 1
        print(
            f"   Batch {batch_num}/{total_batches} completed ({len(batch_ids)} embeddings)"
        )


def store_transformer_metadata_and_person_embeddings(
    genre_mlb,
    year_min,
    year_max,
    text_features,
    director_embs,
    director_emb_dict,
    cast_emb_dict,
):
    """Store transformer metadata and person embeddings in database."""
    print("🗄️  Storing transformer metadata...")

    # Store genre classes
    store_transformer_metadata("genre_classes", list(genre_mlb.classes_))
    print("   ✓ Stored genre classes")

    # Store year statistics
    store_transformer_metadata(
        "year_stats", {"min": float(year_min), "max": float(year_max)}
    )
    print("   ✓ Stored year normalization parameters")

    # Store text embedding metadata (Sentence Transformer)
    text_metadata = {
        "method": "sentence_transformer",
        "model_name": DEFAULT_MODEL,
        "embedding_dim": text_features.shape[1],
    }
    store_transformer_metadata("text_metadata", text_metadata)
    print("   ✓ Stored text embedding metadata (sentence transformer)")

    # Store person embedding metadata
    person_metadata = {
        "method": "sentence_transformer",
        "model_name": DEFAULT_MODEL,
        "embedding_dim": director_embs.shape[1],
    }
    store_transformer_metadata("person_metadata", person_metadata)
    print("   ✓ Stored person embedding metadata (sentence transformer)")

    # Store person embeddings in database
    store_person_embeddings(director_emb_dict, "director")
    print(f"   ✓ Stored {len(director_emb_dict)} director embeddings")

    store_person_embeddings(cast_emb_dict, "actor")
    print(f"   ✓ Stored {len(cast_emb_dict)} actor embeddings")


def compute_and_store_embeddings():
    """Main function to compute and store movie embeddings."""
    print("🔄 Loading movie data from PostgreSQL...")
    movies_data = get_movies_with_details()

    if not movies_data:
        print("❌ No movie data found!")
        return False

    print(f"📊 Processing {len(movies_data)} movies for embedding computation...")

    # Extract features
    genres_list, directors_list, cast_list, years_list = extract_movie_features(
        movies_data
    )

    # Compute individual feature types
    genre_features, genre_mlb = compute_genre_features(genres_list)
    year_features, year_min, year_max = compute_year_features(years_list)
    text_features = compute_text_features(movies_data)
    director_embs, cast_embs, director_emb_dict, cast_emb_dict = (
        compute_person_embeddings(directors_list, cast_list)
    )

    # Combine all features
    movie_vectors, total_dim = combine_features(
        genre_features, year_features, director_embs, cast_embs, text_features
    )

    # Initialize embeddings table
    print("🗄️  Initializing embeddings table...")
    init_embeddings_table(total_dim)

    # Store embeddings
    movie_ids = [movie["id"] for movie in movies_data]
    store_embeddings_in_batches(movie_ids, movie_vectors)

    # Store metadata and person embeddings
    store_transformer_metadata_and_person_embeddings(
        genre_mlb,
        year_min,
        year_max,
        text_features,
        director_embs,
        director_emb_dict,
        cast_emb_dict,
    )

    print("✅ Embedding computation and storage completed!")
    print(f"📊 Total embeddings stored: {len(movie_ids)}")
    print(f"🎯 Embedding dimension: {total_dim}")

    return True


def main():
    print("Loading movie embeddings...")

    ensure_metadata_tables()
    compute_and_store_embeddings()

    print("✅ Embedding loading completed!")


if __name__ == "__main__":
    main()
