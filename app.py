from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from utils.model_loader import load_model
from utils.question_selector import get_question_by_level
from utils.evaluation_logic import evaluate_answer
from utils.enums import ExperienceLevel
from utils.technical_evaluator import technical_evaluator
from utils.json_encoder import convert_numpy_types, CustomJSONEncoder
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Set custom JSON encoder
app.json = CustomJSONEncoder(app)

# Load communication skills model
model_bundle = load_model()
models = model_bundle["models"]
vectorizer = model_bundle["tfidf_vectorizer"]
scalers = model_bundle["scalers"]
experience_indicators = model_bundle["experience_indicators"]

@app.route("/")
def index():
    return render_template("index.html")

# === COMMUNICATION SKILLS ROUTES ===

@app.route("/communication/question", methods=["POST"])
def get_communication_question():
    """Get a communication skills question"""
    data = request.get_json()
    level = data.get("level", "").lower()
    question = get_question_by_level(level)
    return jsonify({"question": question, "type": "communication"})

@app.route("/communication/evaluation", methods=["POST"])
def evaluate_communication():
    """Evaluate communication skills answer"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    # Map UI levels to model levels
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
        "level": mapped_level,
        "type": "communication"
    })

# === TECHNICAL SKILLS ROUTES ===

@app.route("/technical/question", methods=["POST"])
def get_technical_question():
    """Get a technical question based on level and skills"""
    try:
        data = request.get_json()
        level = data.get("level", "intern").lower()
        skills = data.get("skills", ["java", "react"])  # Default skills
        current_complexity = data.get("current_complexity", None)
        
        question_data = technical_evaluator.get_technical_question(
            experience_level=level,
            skills=skills,
            current_complexity=current_complexity
        )
        
        # Convert numpy types to native Python types
        response_data = convert_numpy_types({
            "question": question_data["question"],
             "expected_answer": question_data["expected_answer"],
            "complexity_score": question_data["complexity_score"],
            "technology": question_data["technology"],
            "bloom_label": question_data["bloom_label"],
            "question_id": question_data["question_id"],
            "type": "technical"
        })
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in get_technical_question: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route("/technical/evaluation", methods=["POST"])
def evaluate_technical():
    """Evaluate technical skills answer"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing JSON body"}), 400

        level = data.get("level", "intern").lower()
        question = data.get("question", "")
        answer = data.get("answer", "")
        expected_answer = data.get("expected_answer", "")
        complexity_score = data.get("complexity_score", 2.0)
        technology = data.get("technology", "general")
        bloom_label = data.get("bloom_label", "")

        # Create question data structure
        question_data = {
            "question": question,
            "expected_answer": expected_answer,
            "complexity_score": complexity_score,
            "technology": technology,
            "bloom_label": bloom_label
        }

        # Get comprehensive evaluation
        result = technical_evaluator.get_comprehensive_evaluation(
            question_data=question_data,
            candidate_answer=answer,
            experience_level=level
        )

        # Convert numpy types to native Python types
        response_data = convert_numpy_types({
            "evaluation": result,
            "question": question,
            "level": level,
            "type": "technical"
        })

        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in evaluate_technical: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# === COMBINED ASSESSMENT ROUTES ===

@app.route("/assessment/start", methods=["POST"])
def start_assessment():
    """Start a new assessment session"""
    data = request.get_json()
    level = data.get("level", "intern").lower()
    skills = data.get("skills", ["java", "react"])
    assessment_type = data.get("type", "both")  # "communication", "technical", or "both"
    
    session_data = {
        "level": level,
        "skills": skills,
        "type": assessment_type,
        "current_complexity": technical_evaluator.experience_starting_score.get(level, 2.0),
        "questions_answered": 0,
        "total_score": 0
    }
    
    return jsonify({
        "session": session_data,
        "message": f"Assessment started for {level} level"
    })

@app.route("/assessment/next-question", methods=["POST"])
def get_next_question():
    """Get the next question in an adaptive assessment"""
    data = request.get_json()
    session = data.get("session", {})
    last_performance = data.get("last_performance", None)
    
    level = session.get("level", "intern")
    skills = session.get("skills", ["java", "react"])
    assessment_type = session.get("type", "both")
    current_complexity = session.get("current_complexity", 2.0)
    questions_answered = session.get("questions_answered", 0)
    
    # Determine question type (alternate between communication and technical if "both")
    if assessment_type == "both":
        question_type = "technical" if questions_answered % 2 == 0 else "communication"
    else:
        question_type = assessment_type
    
    if question_type == "technical":
        question_data = technical_evaluator.get_technical_question(
            experience_level=level,
            skills=skills,
            current_complexity=current_complexity
        )
        question_data["type"] = "technical"
    else:
        question = get_question_by_level(level)
        question_data = {
            "question": question,
            "type": "communication"
        }
    
    return jsonify(question_data)

# === UTILITY ROUTES ===

@app.route("/skills/list", methods=["GET"])
def get_available_skills():
    """Get list of available technical skills"""
    skills = technical_evaluator.question_bank["technology"].unique().tolist()
    return jsonify({"skills": skills})

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "communication_model": "loaded",
        "technical_model": "loaded",
        "question_bank_size": len(technical_evaluator.question_bank)
    })

if __name__ == "__main__":
    app.run(debug=True)