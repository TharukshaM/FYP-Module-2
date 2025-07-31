import numpy as np
from utils.features import extract_statistical_features

def evaluate_answer(models, vectorizer, scalers, experience_indicators, answer, experience_level):
    # Extract statistical features
    stats = extract_statistical_features(answer)
    stat_values = np.array(list(stats.values())).reshape(1, -1)

    # TF-IDF features
    tfidf_vector = vectorizer.transform([answer])

    # Combine features
    input_vector = np.hstack((stat_values, tfidf_vector.toarray()))

    # Predictions
    predictions = {}
    for key in models:
        model = models[key]
        scaler = scalers[key]
        scaled_input = scaler.transform(input_vector)
        pred = model.predict(scaled_input)[0]
        predictions[key] = round(float(pred), 2)

    # Experience-level based score adjustment (optional but recommended)
    level_indicators = experience_indicators.get(experience_level, {})
    positive_words = level_indicators.get("positive", [])
    negative_words = level_indicators.get("negative", [])

    lower_answer = answer.lower()
    bonus = sum(1 for word in positive_words if word in lower_answer)
    penalty = sum(1 for word in negative_words if word in lower_answer)

    predictions["adjusted_score"] = round(
        (predictions["technical_accuracy"] + predictions["communication_effectiveness"] + predictions["competency_demonstration"]) / 3
        + bonus * 0.5
        - penalty * 0.5,
        2
    )

    return predictions
