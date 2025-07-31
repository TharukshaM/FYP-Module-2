import pickle
import sys
from utils.enums import ExperienceLevel

# Ensure global registration
sys.modules['__main__'].ExperienceLevel = ExperienceLevel

# def load_model():
#     with open('model/trained_communication_evaluator.pkl', 'rb') as f:
#         return pickle.load(f)

import pickle

def load_model():
    with open("model/trained_communication_evaluator.pkl", "rb") as f:
        bundle = pickle.load(f)
    print("🔍 Loaded model bundle type:", type(bundle))
    print("📦 Bundle content:", bundle)
    return bundle
