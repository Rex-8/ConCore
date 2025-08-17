import json

def parse_json(filepath: str):
    """
    Parse JSON file and return standardized format.
    Handles different JSON structures (dict, list, primitives).
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Analyze the JSON structure
    features = []
    population = 0
    
    if isinstance(data, dict):
        # If it's a dictionary, use keys as features
        features = list(data.keys())
        population = 1  # Single record
        
        # If any values are lists, count those instead
        for key, value in data.items():
            if isinstance(value, list):
                population = max(population, len(value))
                
    elif isinstance(data, list):
        population = len(data)
        
        # If list contains dictionaries, use keys from first dict as features
        if data and isinstance(data[0], dict):
            features = list(data[0].keys())
        else:
            features = ["value"]  # For list of primitives
    else:
        # Primitive value
        features = ["value"]
        population = 1

    return {
        "features": features,  # Keys/features
        "population": population,
        "preview": data if population <= 5 else (data[:5] if isinstance(data, list) else data),
        "raw_json": data,
        "file_type": "json"
    }