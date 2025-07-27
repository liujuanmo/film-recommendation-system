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


def load_crew():
    crew_path = os.path.join(DATA_DIR, "title.crew.tsv")
    return pd.read_csv(
        crew_path, sep="\t", na_values="\\N", usecols=["tconst", "directors"]
    )


def load_principals():
    principals_path = os.path.join(DATA_DIR, "title.principals.tsv")
    return pd.read_csv(
        principals_path,
        sep="\t",
        na_values="\\N",
        usecols=["tconst", "nconst", "category"],
    )


def load_names():
    names_path = os.path.join(DATA_DIR, "name.basics.tsv")
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
        bulk_insert_directors(director_names)
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
            bulk_insert_movie_directors(movie_directors_data)


def load_actors(principals, names):
    if principals.empty or names.empty:
        return
    actors_principals = principals[
        principals["category"].isin(["actor", "actress"])
    ].copy()
    actor_nconsts = set(actors_principals["nconst"].dropna())
    actor_names = names[names["nconst"].isin(actor_nconsts)].copy()

    if not actor_names.empty:
        bulk_insert_actors(actor_names)
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
            bulk_insert_movie_actors(movie_actors_data)


def main_load_imdb_data():
    create_tables()
    print("Database tables initialized")

    # run `truncate table movies cascade;` if you want to start fresh
    print("  → Loading movies...")
    movies = load_titles()
    bulk_insert_movies(movies)
    print("  → Loaded movies.")

    print("  → Loading names...")
    names = load_names()
    print("  → Loaded names")

    print("  → Loading directors...")
    crew = load_crew()
    load_directors(crew, names)
    print("  → Loaded directors")

    print("  → Loading actors...")
    principals = load_principals()
    load_actors(principals, names)
    print("  → Loaded actors")

    print("  → All data loaded.")


if __name__ == "__main__":
    print("  → Loading movies...")
    movies = load_titles()
    print("  → Loaded movies")
    bulk_insert_movies(movies)
    print("  → Inserted all movies")
