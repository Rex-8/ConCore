import pandas as pd
import sqlite3

def parse_sqlite(filepath: str):
    conn = sqlite3.connect(filepath)
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)

    previews = {}
    for t in tables["name"]:
        df = pd.read_sql(f"SELECT * FROM {t} LIMIT 5;", conn)
        previews[t] = df.to_dict()

    conn.close()
    return {"tables": tables["name"].tolist(), "preview": previews}
