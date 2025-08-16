import pandas as pd

def parse_csv(filepath: str):
    df = pd.read_csv(filepath)
    return {"columns": df.columns.tolist(), "preview": df.head().to_dict()}
