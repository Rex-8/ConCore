import pandas as pd
import sqlite3
from .utils import dataframe_to_json_dict

def parse_sqlite(filepath: str):
    """
    Parse SQLite file and return standardized format with:
    - features: dictionary mapping table names to their column lists
    - population: total rows across all tables
    - tables: list of table names
    - preview: sample data from each table
    """
    conn = sqlite3.connect(filepath)
    
    # Get all table names
    tables_df = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
    table_names = tables_df["name"].tolist()
    
    # Get previews and analyze structure
    previews = {}
    features_by_table = {}  # Dictionary mapping table -> [columns]
    total_population = 0
    
    for table_name in table_names:
        try:
            # Get table info
            df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 5;", conn)
            
            # Convert DataFrame to JSON-safe dict
            previews[table_name] = dataframe_to_json_dict(df)
            
            # Store columns for this specific table
            features_by_table[table_name] = df.columns.tolist()
            
            # Count total rows in this table
            count_df = pd.read_sql(f"SELECT COUNT(*) as count FROM {table_name};", conn)
            total_population += int(count_df["count"].iloc[0])  # Convert to Python int
            
        except Exception as e:
            previews[table_name] = {"error": str(e)}
            features_by_table[table_name] = []
    
    conn.close()
    
    return {
        "features": features_by_table,  # Dict: {table1: [col1, col2], table2: [col3, col4]}
        "population": total_population, # Total rows across all tables
        "tables": table_names,          # List of table names
        "preview": previews,            # Data preview for each table
        "file_type": "sqlite"
    }