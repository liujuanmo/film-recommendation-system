import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sentence_transformers import SentenceTransformer
from src.constants import DEFAULT_MODEL, get_text_model
from src.db_functions import (
    create_tables,
    init_embeddings_table,
    insert_embeddings,
    store_transformer_metadata,
    store_person_embeddings,
    get_movies_with_details,
    bulk_insert_movies,
    bulk_insert_directors,
    bulk_insert_actors,
    bulk_insert_movie_directors,
    bulk_insert_movie_actors,
)


def check_database_connection():
    """Check if database connection is available."""
    try:
        from src.db_connect import get_engine

        engine = get_engine()
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception as e:
        print(f"⚠️  Database connection failed: {e}")
        print("💡 To use database storage, please:")
        print("   1. Install PostgreSQL: brew install postgresql")
        print("   2. Start PostgreSQL: brew services start postgresql")
        print("   3. Create database and update connection settings")
        return False


def save_embeddings_to_file(
    movie_ids, movie_vectors, metadata, output_file="movie_embeddings.npz"
):
    """Save embeddings and metadata to NPZ file."""
    print(f"💾 Saving embeddings to {output_file}...")

    np.savez_compressed(
        output_file, movie_ids=movie_ids, embeddings=movie_vectors, metadata=metadata
    )
    print(f"    ✅ Saved {len(movie_ids)} embeddings to {output_file}")


def save_metadata_to_json(metadata, output_file="embedding_metadata.json"):
    """Save metadata to JSON file."""
    import json

    print(f"💾 Saving metadata to {output_file}...")

    # Convert numpy arrays to lists for JSON serialization
    json_metadata = {}
    for key, value in metadata.items():
        if isinstance(value, np.ndarray):
            json_metadata[key] = value.tolist()
        elif isinstance(value, np.integer):
            json_metadata[key] = int(value)
        elif isinstance(value, np.floating):
            json_metadata[key] = float(value)
        else:
            json_metadata[key] = value

    with open(output_file, "w") as f:
        json.dump(json_metadata, f, indent=2)

    print(f"    ✅ Saved metadata to {output_file}")


class TextFeatureExtractor:
    """Extract text features using sentence transformers."""

    def __init__(self):
        self.model = get_text_model()

    def fit_transform(self, text_corpus):
        """Extract text embeddings from corpus."""
        if not text_corpus:
            return np.array([])

        # Clean and prepare text
        cleaned_texts = []
        for text in text_corpus:
            if pd.isna(text) or text == "":
                cleaned_texts.append("")
            else:
                cleaned_texts.append(str(text).strip())

        # Generate embeddings
        embeddings = self.model.encode(cleaned_texts, show_progress_bar=False)
        return embeddings


def build_embedding_dict(names_set):
    """Build embedding dictionary for a set of names."""
    if not names_set:
        return {}

    model = get_text_model()
    names_list = list(names_set)
    embeddings = model.encode(names_list, show_progress_bar=False)

    return dict(zip(names_list, embeddings))


def get_mean_embedding(names_list, embedding_dict):
    """Get mean embedding for a list of names."""
    if not names_list or not embedding_dict:
        # Return zero vector with same dimension as other embeddings
        model = get_text_model()
        return np.zeros(model.get_sentence_embedding_dimension())

    valid_embeddings = []
    for name in names_list:
        if name in embedding_dict:
            valid_embeddings.append(embedding_dict[name])

    if not valid_embeddings:
        model = get_text_model()
        return np.zeros(model.get_sentence_embedding_dimension())

    return np.mean(valid_embeddings, axis=0)


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
    for _, movie in movies_data.iterrows():
        title = movie["primaryTitle"] or ""
        text_corpus.append(title)

    text_extractor = TextFeatureExtractor()
    text_features = text_extractor.fit_transform(text_corpus)
    print(
        f"     Text feature dimension: {text_features.shape[1]} (sentence transformer)"
    )

    return text_features


def compute_person_embeddings(directors_list, actors_list):
    """Compute semantic embeddings for directors and actors."""
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

    # Actor embeddings
    print("  → Computing semantic actor embeddings...")
    all_actors = set()
    for actors in actors_list:
        if actors:
            all_actors.update(actors)

    actor_emb_dict = build_embedding_dict(all_actors)
    actor_embs = np.array(
        [get_mean_embedding(actors, actor_emb_dict) for actors in actors_list]
    )
    print(f"     Processed {len(all_actors)} unique actors with semantic embeddings")

    return director_embs, actor_embs, director_emb_dict, actor_emb_dict


def combine_features(
    genre_features, year_features, director_embs, actor_embs, text_features
):
    """Combine all features into final movie vectors."""
    print("  → Combining all features...")

    movie_vectors = np.hstack(
        [
            genre_features,  # Genre one-hot vectors
            year_features,  # Normalized year
            director_embs,  # Director embeddings
            actor_embs,  # Actor embeddings
            text_features,  # Text features (sentence transformer)
        ]
    ).astype(np.float32)

    total_dim = movie_vectors.shape[1]
    print(f"     Final embedding dimension: {total_dim}")
    print(f"       - Genres: {genre_features.shape[1]}")
    print(f"       - Year: {year_features.shape[1]}")
    print(f"       - Directors: {director_embs.shape[1]}")
    print(f"       - Actors: {actor_embs.shape[1]}")
    print(f"       - Text: {text_features.shape[1]}")

    return movie_vectors, total_dim


def load_and_preprocess_data(csv_file):
    """Load and preprocess the merged CSV data."""
    print(f"🔄 Loading data from {csv_file}...")

    # Load CSV
    df = pd.read_csv(csv_file)
    print(f"    ✅ Loaded {len(df)} movies")

    # Convert pipe-separated strings back to lists
    df["genres"] = df["genres"].apply(
        lambda x: x.split("|") if pd.notna(x) and x != "" else []
    )
    df["directors"] = df["directors"].apply(
        lambda x: x.split("|") if pd.notna(x) and x != "" else []
    )
    df["actors"] = df["actors"].apply(
        lambda x: x.split("|") if pd.notna(x) and x != "" else []
    )

    # Clean up data types
    df["startYear"] = (
        pd.to_numeric(df["startYear"], errors="coerce").fillna(0).astype(int)
    )

    print(f"    ✅ Preprocessed data")
    return df


def save_embeddings_to_database(movie_ids, movie_vectors, total_dim):
    """Save embeddings to PostgreSQL database."""
    print("💾 Saving embeddings to PostgreSQL...")

    # Initialize embeddings table
    init_embeddings_table(total_dim)

    # Store embeddings in batches
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

    print(f"    ✅ Saved {len(movie_ids)} embeddings to database")


def save_metadata_to_database(metadata, director_emb_dict, actor_emb_dict):
    """Save metadata and person embeddings to PostgreSQL database."""
    print("💾 Saving metadata to PostgreSQL...")

    # Store genre classes
    store_transformer_metadata("genre_classes", metadata["genre_classes"])
    print("   ✓ Stored genre classes")

    # Store year statistics
    store_transformer_metadata(
        "year_stats", {"min": metadata["year_min"], "max": metadata["year_max"]}
    )
    print("   ✓ Stored year normalization parameters")

    # Store text embedding metadata
    text_metadata = {
        "method": "sentence_transformer",
        "model_name": metadata["model_name"],
        "embedding_dim": metadata["feature_breakdown"]["text"],
    }
    store_transformer_metadata("text_metadata", text_metadata)
    print("   ✓ Stored text embedding metadata")

    # Store person embedding metadata
    person_metadata = {
        "method": "sentence_transformer",
        "model_name": metadata["model_name"],
        "embedding_dim": metadata["feature_breakdown"]["directors"],
    }
    store_transformer_metadata("person_metadata", person_metadata)
    print("   ✓ Stored person embedding metadata")

    # Store person embeddings
    store_person_embeddings(director_emb_dict, "director")
    print(f"   ✓ Stored {len(director_emb_dict)} director embeddings")

    store_person_embeddings(actor_emb_dict, "actor")
    print(f"   ✓ Stored {len(actor_emb_dict)} actor embeddings")


def compute_and_save_embeddings(csv_file):
    """Main function to compute and save movie embeddings."""
    print("🎬 Movie Embedding Generation Pipeline")
    print("=" * 50)

    try:
        # Check database connection
        use_database = check_database_connection()

        if use_database:
            # Create database tables
            print("🗄️  Setting up database...")
            create_tables()
            print("   ✅ Database setup completed")

        # Load and preprocess data
        movies_data = load_and_preprocess_data(csv_file)

        # Extract features
        genres_list = movies_data["genres"].tolist()
        directors_list = movies_data["directors"].tolist()
        actors_list = movies_data["actors"].tolist()
        years_list = movies_data["startYear"].tolist()

        # Compute features
        genre_features, genre_mlb = compute_genre_features(genres_list)
        year_features, year_min, year_max = compute_year_features(years_list)
        text_features = compute_text_features(movies_data)
        director_embs, actor_embs, director_emb_dict, actor_emb_dict = (
            compute_person_embeddings(directors_list, actors_list)
        )

        # Combine features
        movie_vectors, total_dim = combine_features(
            genre_features,
            year_features,
            director_embs,
            actor_embs,
            text_features,
        )

        # Prepare metadata
        metadata = {
            "total_movies": len(movies_data),
            "embedding_dimension": total_dim,
            "genre_classes": genre_mlb.classes_.tolist(),
            "year_min": year_min,
            "year_max": year_max,
            "model_name": DEFAULT_MODEL,
            "feature_breakdown": {
                "genres": genre_features.shape[1],
                "year": year_features.shape[1],
                "directors": director_embs.shape[1],
                "actors": actor_embs.shape[1],
                "text": text_features.shape[1],
            },
        }

        # Save embeddings and metadata
        movie_ids = movies_data["tconst"].tolist()

        if use_database:
            save_embeddings_to_database(movie_ids, movie_vectors, total_dim)
            save_metadata_to_database(metadata, director_emb_dict, actor_emb_dict)
            print(f"🗄️  Data saved to PostgreSQL database")
        else:
            save_embeddings_to_file(
                movie_ids, movie_vectors, metadata, "movie_embeddings.npz"
            )
            save_metadata_to_json(metadata, "embedding_metadata.json")
            print(f"📁 Data saved to files")

        print("\n🎉 Embedding generation completed successfully!")
        print(f"📊 Total movies: {len(movies_data)}")
        print(f"🎯 Embedding dimension: {total_dim}")

        return True

    except Exception as e:
        print(f"❌ Error in embedding generation: {e}")
        raise


def main():
    """Main function."""
    compute_and_save_embeddings("work_data/merged.csv")


if __name__ == "__main__":
    main()
