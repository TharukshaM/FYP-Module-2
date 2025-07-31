from flask import Flask, render_template, request, jsonify
from utils.model_loader import load_model
from utils.question_selector import get_question_by_level
from utils.evaluation_logic import evaluate_answer
from utils.enums import ExperienceLevel

app = Flask(__name__)
model_bundle = load_model()
model_bundle = load_model()
models = model_bundle["models"]
vectorizer = model_bundle["tfidf_vectorizer"]
scalers = model_bundle["scalers"]
experience_indicators = model_bundle["experience_indicators"]


@app.route("/evaluation", methods=["POST"])
def index():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    # Optional: map UI levels to model levels
    level_map = {
        "intern": "basic",
        "associate": "medium",
        "software engineer": "hard"
    }
    experience_level_str = data.get("level", "").lower()
    mapped_level = level_map.get(experience_level_str, "basic")
    question = data.get("question", "")
    answer = data.get("answer", "")

    result = evaluate_answer(models, vectorizer, scalers, experience_indicators, answer, mapped_level)



    return jsonify({
        "evaluation": result,
        "question": question,
        "level": mapped_level
    })

@app.route("/question", methods=["POST"])
def get_question():
    data = request.get_json()
    level = data.get("level", "").lower()
    question = get_question_by_level(level)
    return jsonify({"question": question})

if __name__ == "__main__":
    app.run(debug=True)
