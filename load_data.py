import os
import pandas as pd
from src.db_client import (
    enable_pgvector_and_create_tables,
    get_movie_count,
    insert_movies,
    insert_directors,
    insert_actors,
    insert_movie_directors,
    insert_movie_actors,
)
from src.constants import DATA_DIR


def load_titles():
    basics_path = os.path.join(DATA_DIR, "title.basics.tsv")
    titles = pd.read_csv(
        basics_path,
        sep="\t",
        na_values="\\N",
        usecols=["tconst", "primaryTitle", "startYear", "genres", "titleType"],
    )
    movies = titles[titles["titleType"] == "movie"].copy()
    movies = movies.dropna(subset=["genres"])
    movies["genres"] = movies["genres"].apply(
        lambda x: x.split(",") if isinstance(x, str) else []
    )
    if "overview" not in movies.columns:
        movies["overview"] = movies["primaryTitle"]

    return movies


def load_crew():
    """Load and process title.crew.tsv."""
    crew_path = os.path.join(DATA_DIR, "title.crew.tsv")
    if not os.path.exists(crew_path):
        print(f"Warning: {crew_path} not found! Directors will not be available.")
        return pd.DataFrame(columns=["tconst", "directors"])
    return pd.read_csv(
        crew_path, sep="\t", na_values="\\N", usecols=["tconst", "directors"]
    )


def load_principals():
    principals_path = os.path.join(DATA_DIR, "title.principals.tsv")
    if not os.path.exists(principals_path):
        return pd.DataFrame(columns=["tconst", "nconst", "category"])
    return pd.read_csv(
        principals_path,
        sep="\t",
        na_values="\\N",
        usecols=["tconst", "nconst", "category"],
    )


def load_names():
    """Load and process name.basics.tsv."""
    names_path = os.path.join(DATA_DIR, "name.basics.tsv")
    if not os.path.exists(names_path):
        return pd.DataFrame(columns=["nconst", "primaryName"])
    return pd.read_csv(
        names_path, sep="\t", na_values="\\N", usecols=["nconst", "primaryName"]
    )


def load_directors(crew, names):
    if crew.empty or names.empty:
        return

    director_nconsts = set()
    for _, row in crew.iterrows():
        if pd.notna(row["directors"]) and row["directors"]:
            directors_list = row["directors"].split(",")
            director_nconsts.update(directors_list)

    director_names = names[names["nconst"].isin(director_nconsts)].copy()
    if not director_names.empty:
        insert_directors(director_names)
        movie_directors_data = []
        for _, row in crew.iterrows():
            if pd.notna(row["directors"]) and row["directors"]:
                directors_list = row["directors"].split(",")
                for director_nconst in directors_list:
                    if director_nconst.strip():
                        movie_directors_data.append(
                            (row["tconst"], director_nconst.strip())
                        )

        if movie_directors_data:
            insert_movie_directors(movie_directors_data)


def load_actors(principals, names):
    if principals.empty or names.empty:
        return
    actors_principals = principals[
        principals["category"].isin(["actor", "actress"])
    ].copy()
    actor_nconsts = set(actors_principals["nconst"].dropna())
    actor_names = names[names["nconst"].isin(actor_nconsts)].copy()

    if not actor_names.empty:
        insert_actors(actor_names)
        movie_actors_data = []
        actors_grouped = (
            actors_principals.groupby("tconst")["nconst"]
            .apply(lambda x: list(x)[:5])  # Limit to first 5 actors per movie
            .reset_index()
        )
        for _, row in actors_grouped.iterrows():
            for actor_nconst in row["nconst"]:
                if actor_nconst and pd.notna(actor_nconst):
                    movie_actors_data.append((row["tconst"], actor_nconst))

        if movie_actors_data:
            insert_movie_actors(movie_actors_data)


def main_load_imdb_data():
    enable_pgvector_and_create_tables()
    print("Database tables initialized")

    print("  → Processing movies...")
    movies = load_titles()
    insert_movies(movies)
    print("  → Processed movies.")

    print("  → Processing names...")
    names = load_names()
    print("  → Processed names")

    print("  → Processing directors...")
    crew = load_crew()
    load_directors(crew, names)
    print("  → Processed directors")

    print("  → Processing actors...")
    principals = load_principals()
    load_actors(principals, names)
    print("  → Processed actors")

    final_count = get_movie_count()
    print(f"  → Completed. Total movies: {final_count}")


if __name__ == "__main__":
    main_load_imdb_data()
