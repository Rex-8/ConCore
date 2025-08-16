import pandas as pd

def parse_json(filepath: str):
    df = pd.read_json(filepath)
    return {"columns": df.columns.tolist(), "preview": df.head().to_dict()}
