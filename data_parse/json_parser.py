import json

def parse_json(filepath: str):
    """
    Just load and return raw JSON.
    Can be dict, list, or primitives depending on file content.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {"raw_json": data}
