import json

def load_config(path=r"utils\config.json"):
    with open(path, "r") as f:
        return json.load(f)
    
