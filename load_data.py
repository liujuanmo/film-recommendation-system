import os
import pandas as pd
from src.db_functions import (
    create_tables,
    bulk_insert_movies,
    bulk_insert_directors,
    bulk_insert_actors,
    bulk_insert_movie_directors,
    bulk_insert_movie_actors,
)
from src.constants import DATA_DIR


def load_titles():
    titles_file = os.path.join(DATA_DIR, "title.basics.tsv")
    movies = pd.read_csv(titles_file, sep="\t", na_values="\\N", low_memory=False)

    movies = movies[movies["titleType"] == "movie"]
    movies = movies.dropna(subset=["genres"])
    movies["genres"] = movies["genres"].apply(
        lambda x: x.split(",") if isinstance(x, str) else []
    )

    return movies


def load_crews():
    crew_path = os.path.join(DATA_DIR, "title.crew.tsv")
    crews = pd.read_csv(
        crew_path, sep="\t", na_values="\\N", usecols=["tconst", "directors"]
    )
    return crews[crews["directors"].notna()].copy()


def load_principals():
    principals_path = os.path.join(DATA_DIR, "title.principals.tsv")
    principals = pd.read_csv(
        principals_path,
        sep="\t",
        na_values="\\N",
        usecols=["tconst", "nconst", "category"],
    )
    return principals[principals["category"].isin(["actor", "actress"])].copy()


def load_names():
    names_path = os.path.join(DATA_DIR, "name.basics.tsv")
    names = pd.read_csv(
        names_path, sep="\t", na_values="\\N", usecols=["nconst", "primaryName"]
    )
    return names[names["nconst"].notna()].copy()


def load_directors(crews, names):
    # Get all director nconsts from crews
    director_nconsts = set()
    for _, row in crews.iterrows():
        directors_list = row["directors"].split(",")
        director_nconsts.update([d.strip() for d in directors_list if d.strip()])

    # Filter names to get only directors
    directors_df = names[names["nconst"].isin(director_nconsts)].copy()

    if not directors_df.empty:
        bulk_insert_directors(directors_df)

        # Create movie-director relationships in batches
        batch_size = 5000
        movie_directors_data = []

        for i in range(0, len(crews), batch_size):
            batch_crews = crews.iloc[i : i + batch_size]

            for _, row in batch_crews.iterrows():
                tconst = row["tconst"]
                directors_list = row["directors"].split(",")
                for director_nconst in directors_list:
                    director_nconst = director_nconst.strip()
                    if director_nconst:
                        movie_directors_data.append((tconst, director_nconst))

            # Process batch when we reach batch_size or at the end
            if len(movie_directors_data) >= batch_size or i + batch_size >= len(crews):
                if movie_directors_data:
                    inserted = bulk_insert_movie_directors(movie_directors_data)
                    print(
                        f"✅ Inserted movie-director relationships: {inserted}, {inserted/len(movie_directors_data)*100:.0f}%"
                    )
                    movie_directors_data = []  # Reset for next batch


def load_actors(principals, names):
    # Get all actor nconsts from principals
    actor_nconsts = set(principals["nconst"].dropna())

    # Filter names to get only actors
    actors_df = names[names["nconst"].isin(actor_nconsts)].copy()

    if not actors_df.empty:
        bulk_insert_actors(actors_df)

        # Create movie-actor relationships in batches
        batch_size = 5000
        movie_actors_data = []

        # Group by movie first (limit to first 5 actors per movie)
        actors_grouped = (
            principals.groupby("tconst")["nconst"]
            .apply(lambda x: list(x)[:5])  # Limit to first 5 actors per movie
            .reset_index()
        )

        for i in range(0, len(actors_grouped), batch_size):
            batch_grouped = actors_grouped.iloc[i : i + batch_size]

            for _, row in batch_grouped.iterrows():
                tconst = row["tconst"]
                for actor_nconst in row["nconst"]:
                    if actor_nconst and pd.notna(actor_nconst):
                        movie_actors_data.append((tconst, actor_nconst))

                if movie_actors_data:
                    bulk_insert_movie_actors(movie_actors_data)
                    movie_actors_data = []
            print(
                f"✅ Inserted movie-actor relationships, {i} - {i+len(batch_grouped)}/{len(actors_grouped)}"
            )


if __name__ == "__main__":
    # create_tables()
    # print("Database tables initialized")

    # print("  → Loading movies...")
    # movies = load_titles()
    # print("  → Loaded movies")
    # bulk_insert_movies(movies)
    # print("  → Inserted all movies")

    print("  → Loading names...")
    names = load_names()
    print("  → Loaded names")

    # print("  → Loading crews...")
    # crews = load_crews()
    # print("  → Loaded crews")

    # print("  → Loading directors...")
    # load_directors(crews, names)
    # print("  → Loaded directors")

    print("  → Loading actors...")
    principals = load_principals()
    print("  → Loaded principals")

    print("  → Loading actors...")
    load_actors(principals, names)
    print("  → Loaded actors")

    print("  → All data loaded.")
