import pandas as pd

def parse_excel(filepath: str):
    df = pd.read_excel(filepath)
    return {"columns": df.columns.tolist(), "preview": df.head().to_dict()}
