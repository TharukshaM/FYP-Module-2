import pandas as pd

def get_question_by_level(level: str, path='model/softskill_dataset.csv') -> str:
    df = pd.read_csv(path)
    level_map = {
    "intern": "basic",
    "associate": "medium",
    "software engineer": "hard"
    }
    mapped_level = level_map.get(level, "basic")
    filtered = df[df['Level'].str.lower() == mapped_level.lower()]
    if filtered.empty:
        return "No question available for this level."
    return filtered.sample(1).iloc[0]['Question']
