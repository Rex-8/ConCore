import os
from .csv_parser import parse_csv
from .excel_parser import parse_excel
from .json_parser import parse_json
from .sqlite_parser import parse_sqlite

def parse_file(filepath: str):
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".csv":
        return parse_csv(filepath)
    elif ext in [".xls", ".xlsx"]:
        return parse_excel(filepath)
    elif ext == ".json":
        return parse_json(filepath)
    elif ext in [".db", ".sqlite"]:
        return parse_sqlite(filepath)
    else:
        return {"error": f"Unsupported file type: {ext}"}
