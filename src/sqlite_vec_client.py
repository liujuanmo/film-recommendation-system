import sqlite3
import sqlite_vec
import json

DB_PATH = "movies_vec.db"

def get_con():
    con = sqlite3.connect(DB_PATH)
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    return con

def init_sqlite_vec(con, emb_dim):
    con.execute(f"""
    CREATE VIRTUAL TABLE IF NOT EXISTS movie_embeddings USING vec0(
        movie_id INTEGER PRIMARY KEY,
        embedding FLOAT[{emb_dim}]
    )
    """)
    con.commit()

def insert_embeddings(con, ids, embeddings):
    for movie_id, emb in zip(ids, embeddings):
        if hasattr(emb, 'tolist'):
            emb = emb.tolist()
        emb_str = json.dumps(emb)
        con.execute(
            "INSERT OR REPLACE INTO movie_embeddings (movie_id, embedding) VALUES (?, ?)",
            (movie_id, emb_str)
        )
    con.commit()

def search(con, query_vec, top_n=10):
    if hasattr(query_vec, 'tolist'):
        query_vec = query_vec.tolist()
    query_str = json.dumps(query_vec)
    cur = con.execute(
        "SELECT movie_id, distance FROM movie_embeddings WHERE embedding match ? ORDER BY distance LIMIT ?",
        (query_str, top_n)
    )
    return cur.fetchall() 