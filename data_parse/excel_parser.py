import pandas as pd
from .utils import dataframe_to_json_dict

def parse_excel(filepath: str):
    """
    Parse Excel file and return standardized format with:
    - features: column names
    - population: number of rows
    - preview: sample data
    """
    df = pd.read_excel(filepath)
    
    return {
        "features": df.columns.tolist(),  # Column names/features
        "population": len(df),
        "preview": dataframe_to_json_dict(df.head()),
        "file_type": "excel"
    }