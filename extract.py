import os
import pandas as pd
from src.constants import DATA_DIR


def load_and_merge_data():
    """
    Load and merge all IMDB data files to create a single dataframe with required columns:
    - tconst: str
    - primaryTitle: str
    - startYear: int
    - genres: list[str]
    - directors: list[str]
    - actors: list[str]
    """
    print("🔄 Loading IMDB data files...")

    # Load title basics (movies only)
    print("  → Loading title basics...")
    titles_file = os.path.join(DATA_DIR, "title.basics.tsv")
    movies = pd.read_csv(titles_file, sep="\t", na_values="\\N", low_memory=False)
    movies = movies[movies["titleType"] == "movie"]
    movies = movies.dropna(subset=["genres"])
    movies["genres"] = movies["genres"].apply(
        lambda x: x.split(",") if isinstance(x, str) else []
    )
    # Keep only required columns - keep original column names
    movies = movies[["tconst", "primaryTitle", "startYear", "genres"]].copy()
    print(f"    ✅ Loaded {len(movies)} movies")

    # Load crew data (directors)
    print("  → Loading crew data...")
    crew_file = os.path.join(DATA_DIR, "title.crew.tsv")
    crews = pd.read_csv(
        crew_file, sep="\t", na_values="\\N", usecols=["tconst", "directors"]
    )
    crews = crews[crews["directors"].notna()].copy()
    print(f"    ✅ Loaded crew data for {len(crews)} movies")

    # Load principals (actors/actresses)
    print("  → Loading principals...")
    principals_file = os.path.join(DATA_DIR, "title.principals.tsv")
    principals = pd.read_csv(
        principals_file,
        sep="\t",
        na_values="\\N",
        usecols=["tconst", "nconst", "category"],
    )
    principals = principals[principals["category"].isin(["actor", "actress"])].copy()
    print(f"    ✅ Loaded {len(principals)} actor/actress records")

    # Load names
    print("  → Loading names...")
    names_file = os.path.join(DATA_DIR, "name.basics.tsv")
    names = pd.read_csv(
        names_file, sep="\t", na_values="\\N", usecols=["nconst", "primaryName"]
    )
    names = names[names["nconst"].notna()].copy()
    print(f"    ✅ Loaded {len(names)} names")

    print("🔄 Processing and merging data...")

    # Process directors: convert nconst IDs to names and create list per movie
    print("  → Processing directors...")
    crews["directors_list"] = crews["directors"].apply(
        lambda x: [d.strip() for d in x.split(",") if d.strip()] if pd.notna(x) else []
    )

    # Create director mapping
    director_nconsts = set()
    for directors_list in crews["directors_list"]:
        director_nconsts.update(directors_list)

    director_names = names[names["nconst"].isin(director_nconsts)].copy()
    director_mapping = dict(
        zip(director_names["nconst"], director_names["primaryName"])
    )

    # Convert director nconsts to names
    crews["directors"] = crews["directors_list"].apply(
        lambda directors_list: [
            director_mapping.get(nconst, nconst)
            for nconst in directors_list
            if nconst in director_mapping
        ]
    )
    crews = crews[["tconst", "directors"]].copy()
    print(f"    ✅ Processed directors for {len(crews)} movies")

    # Process actors: group by movie, limit to first 5, convert to names
    print("  → Processing actors...")
    # Group actors by movie and limit to first 5
    actors_grouped = (
        principals.groupby("tconst")["nconst"]
        .apply(lambda x: list(x)[:5])  # Limit to first 5 actors per movie
        .reset_index()
    )
    actors_grouped.columns = ["tconst", "actor_nconsts"]

    # Create actor mapping
    actor_nconsts = set()
    for actor_list in actors_grouped["actor_nconsts"]:
        actor_nconsts.update(actor_list)

    actor_names = names[names["nconst"].isin(actor_nconsts)].copy()
    actor_mapping = dict(zip(actor_names["nconst"], actor_names["primaryName"]))

    # Convert actor nconsts to names
    actors_grouped["actors"] = actors_grouped["actor_nconsts"].apply(
        lambda actor_list: [
            actor_mapping.get(nconst, nconst)
            for nconst in actor_list
            if nconst in actor_mapping
        ]
    )
    actors_grouped = actors_grouped[["tconst", "actors"]].copy()
    print(f"    ✅ Processed actors for {len(actors_grouped)} movies")

    # Merge all data
    print("  → Merging all data...")
    merged_df = movies.copy()

    # Merge directors
    merged_df = merged_df.merge(crews, on="tconst", how="left")
    merged_df["directors"] = merged_df["directors"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    # Merge actors
    merged_df = merged_df.merge(actors_grouped, on="tconst", how="left")
    merged_df["actors"] = merged_df["actors"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    # Clean up data types
    merged_df["startYear"] = (
        pd.to_numeric(merged_df["startYear"], errors="coerce").fillna(0).astype(int)
    )

    # Filter out movies with no directors or actors (optional, but keeps quality high)
    print(f"  → Before filtering: {len(merged_df)} movies")
    merged_df = merged_df[
        (merged_df["directors"].apply(len) > 0) | (merged_df["actors"].apply(len) > 0)
    ].copy()
    print(f"  → After filtering: {len(merged_df)} movies")

    print(f"✅ Successfully merged data for {len(merged_df)} movies")
    return merged_df


def save_to_csv(df, filename="merged.csv"):
    """Save the merged dataframe to CSV file."""
    print(f"🔄 Saving data to {filename}...")

    # Convert list columns to string representation for CSV
    df_csv = df.copy()

    def safe_join(x):
        """Safely join list items, filtering out NaN/float values."""
        if isinstance(x, list):
            # Filter out NaN and non-string values, convert to string
            clean_items = [
                str(item) for item in x if pd.notna(item) and str(item) != "nan"
            ]
            return "|".join(clean_items)
        return ""

    df_csv["genres"] = df_csv["genres"].apply(safe_join)
    df_csv["directors"] = df_csv["directors"].apply(safe_join)
    df_csv["actors"] = df_csv["actors"].apply(safe_join)

    df_csv.to_csv(filename, index=False)
    print(f"✅ Saved {len(df_csv)} movies to {filename}")


def main():
    """Main function to extract, merge, and save movie data to CSV."""
    print("🎬 Movie Data Extraction and Merging Pipeline")
    print("=" * 50)

    try:
        # Load and merge data
        merged_df = load_and_merge_data()

        # Save to CSV
        save_to_csv(merged_df, "work_data/merged.csv")

        print("\n🎉 Pipeline completed successfully!")
        print(f"📊 Final dataset: {len(merged_df)} movies")
        print("📁 Output: merged.csv")

    except Exception as e:
        print(f"❌ Error in pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
