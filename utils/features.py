import re
import numpy as np
from typing import Dict
from textstat import flesch_reading_ease

def extract_statistical_features(text: str) -> Dict[str, float]:
    if not text or len(text.strip()) == 0:
        return {f'stat_{i}': 0.0 for i in range(15)}
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    features = {}
    features['stat_0'] = len(words)
    features['stat_1'] = len(sentences)
    features['stat_2'] = np.mean([len(word) for word in words])
    features['stat_3'] = np.mean([len(s.split()) for s in sentences])
    try:
        features['stat_4'] = flesch_reading_ease(text)
    except:
        features['stat_4'] = 50.0
    unique_words = len(set(word.lower() for word in words))
    features['stat_5'] = unique_words / len(words)
    technical_terms = ['api', 'system', 'database', 'server', 'code', 'algorithm', 'architecture', 'performance', 'security', 'debugging', 'compiler','syntax', 'spaced repetition', 'flashcards', 'mnemonic', 'cheat sheet', 'annotations','ide', 'vscode', 'autocomplete', 'leetcode', 'hackerrank', 'codewars', 'git','implement', 'optimize', 'deploy', 'integrate', 'refactor', 'simulate', 'unit test', 'benchmark']
    features['stat_6'] = sum(1 for term in technical_terms if term in text.lower()) / len(words)
    quality_words = ['clear', 'efficient', 'scalable', 'maintain', 'optimize', 'collaborate']
    features['stat_7'] = sum(1 for word in quality_words if word in text.lower()) / len(words)
    action_words = ['implemented', 'created', 'built', 'designed', 'led', 'managed', 'developed']
    features['stat_8'] = sum(1 for word in action_words if word in text.lower()) / len(words)
    negative_phrases = ["don't know", 'no experience', 'not familiar', 'never used', "can't"]
    features['stat_9'] = sum(1 for phrase in negative_phrases if phrase in text.lower())
    features['stat_10'] = sum(1 for word in words if len(word) > 8) / len(words)
    features['stat_11'] = text.count('?') / len(words)
    professional_words = ['strategy', 'process', 'methodology', 'framework', 'solution']
    features['stat_12'] = sum(1 for word in professional_words if word in text.lower()) / len(words)
    example_indicators = ['example', 'instance', 'such as', 'like', 'including']
    features['stat_13'] = sum(1 for indicator in example_indicators if indicator in text.lower())
    confidence_words = ['will', 'would', 'can', 'able', 'ensure', 'achieve']
    features['stat_14'] = sum(1 for word in confidence_words if word in text.lower()) / len(words)
    return features
