import argparse
import sys
import os
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

from src.db_functions import (
    enable_pgvector_and_create_tables,
    init_embeddings_table,
    insert_embeddings,
    get_session,
    get_movies_with_details,
    table_exists,
    get_embedding_count,
    store_transformer_metadata,
    store_person_embeddings,
    stream_movie_features,
    get_year_min_max,
    get_all_unique_genres,
)
from src.feature_engineering import (
    TextFeatureExtractor,
    build_embedding_dict,
    get_mean_embedding,
)
from src.constants import DEFAULT_MODEL


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
        keywords = " ".join(movie["keywords"]) if movie["keywords"] else ""
        text_corpus.append(f"{title} {keywords}")

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


def compute_and_store_embeddings(batch_size=1000):
    """Main function to compute and store movie embeddings in batches, optimized."""
    print("🔄 Streaming movie data from PostgreSQL in batches...")
    batch_num = 0
    total_embeddings = 0
    director_emb_dict = {}
    cast_emb_dict = {}

    # Get normalization and one-hot info in advance
    print("🔍 Querying year min/max and all unique genres...")
    year_min, year_max = get_year_min_max()
    all_genres = get_all_unique_genres()
    genre_mlb = MultiLabelBinarizer(classes=all_genres)
    genre_mlb.fit([])  # fit with empty to set classes

    text_extractor = TextFeatureExtractor()

    for batch in stream_movie_features(batch_size):
        batch_num += 1
        genres_list = [row["genres"] for row in batch]
        directors_list = [row["directors"] for row in batch]
        cast_list = [row["cast"] for row in batch]
        years_list = [row["start_year"] for row in batch]

        # One-hot encode genres using precomputed classes
        genre_batch_features = genre_mlb.transform(genres_list)
        # Normalize years using precomputed min/max
        years_array = np.array([y if y is not None else 0 for y in years_list])
        if year_max > year_min:
            year_norm = (years_array - year_min) / (year_max - year_min)
        else:
            year_norm = np.zeros(len(years_array))
        year_batch_features = year_norm.reshape(-1, 1)

        # Batch text features (using title as text)
        text_corpus = [row.get("primary_title", "") for row in batch]
        text_features = text_extractor.fit_transform(text_corpus)

        # Person embeddings: only compute for new names
        new_directors = set(n for directors in directors_list for n in directors) - set(
            director_emb_dict
        )
        if new_directors:
            director_emb_dict.update(build_embedding_dict(new_directors))
        new_cast = set(n for cast in cast_list for n in cast) - set(cast_emb_dict)
        if new_cast:
            cast_emb_dict.update(build_embedding_dict(new_cast))
        director_embs = np.array(
            [get_mean_embedding(d, director_emb_dict) for d in directors_list]
        )
        cast_embs = np.array([get_mean_embedding(c, cast_emb_dict) for c in cast_list])

        # Combine features
        movie_vectors, total_dim = combine_features(
            genre_batch_features,
            year_batch_features,
            director_embs,
            cast_embs,
            text_features,
        )

        # Initialize embeddings table if first batch
        if batch_num == 1:
            print("🗄️  Initializing embeddings table...")
            init_embeddings_table(total_dim)

        # Store embeddings
        movie_ids = [row.get("id") for row in batch]
        store_embeddings_in_batches(movie_ids, movie_vectors)
        total_embeddings += len(movie_ids)
        print(f"   Batch {batch_num} completed ({len(movie_ids)} embeddings)")

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
    print(f"📊 Total embeddings stored: {total_embeddings}")
    print(f"🎯 Embedding dimension: {total_dim}")
    return True


def main():
    print("Loading movie embeddings...")

    enable_pgvector_and_create_tables()
    compute_and_store_embeddings()

    print("✅ Embedding loading completed!")


if __name__ == "__main__":
    main()
