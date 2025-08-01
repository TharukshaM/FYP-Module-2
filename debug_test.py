#!/usr/bin/env python3
"""
Debug test script to check technical evaluator functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.technical_evaluator import technical_evaluator
from utils.json_encoder import convert_numpy_types
import json

def test_technical_evaluator():
    """Test the technical evaluator functions"""
    
    print("🔍 Testing Technical Evaluator...")
    
    try:
        # Test 1: Get technical question
        print("\n1. Testing get_technical_question:")
        question_data = technical_evaluator.get_technical_question(
            experience_level="associate",
            skills=["java", "react"],
            current_complexity=3.0
        )
        
        print(f"Question: {question_data['question'][:100]}...")
        print(f"Complexity: {question_data['complexity_score']} (type: {type(question_data['complexity_score'])})")
        print(f"Technology: {question_data['technology']} (type: {type(question_data['technology'])})")
        print(f"Question ID: {question_data['question_id']} (type: {type(question_data['question_id'])})")
        
        # Test 2: Convert to JSON-safe format
        print("\n2. Testing JSON conversion:")
        safe_data = convert_numpy_types(question_data)
        json_string = json.dumps(safe_data, indent=2)
        print("✅ JSON serialization successful!")
        print(f"JSON length: {len(json_string)} characters")
        
        # Test 3: Evaluate answer
        print("\n3. Testing answer evaluation:")
        eval_result = technical_evaluator.evaluate_technical_answer(
            question=question_data["question"],
            expected_answer=question_data["expected_answer"],
            candidate_answer="This is a test answer about technical concepts."
        )
        
        print(f"Correctness: {eval_result['correctness']} (type: {type(eval_result['correctness'])})")
        print(f"Similarity: {eval_result.get('semantic_similarity', 'N/A')}")
        
        # Test 4: Full evaluation
        print("\n4. Testing comprehensive evaluation:")
        comp_eval = technical_evaluator.get_comprehensive_evaluation(
            question_data=question_data,
            candidate_answer="This is a comprehensive test answer with technical details.",
            experience_level="associate"
        )
        
        safe_eval = convert_numpy_types(comp_eval)
        eval_json = json.dumps(safe_eval, indent=2)
        print("✅ Comprehensive evaluation JSON serialization successful!")
        
        print("\n🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading():
    """Test if models load correctly"""
    
    print("\n🔍 Testing Model Loading...")
    
    try:
        # Check if question bank loads
        print(f"Question bank size: {len(technical_evaluator.question_bank)}")
        print(f"Available technologies: {technical_evaluator.question_bank['technology'].unique()[:5]}")
        
        # Check if models are loaded
        print(f"Complexity model type: {type(technical_evaluator.complexity_model)}")
        print(f"Semantic model type: {type(technical_evaluator.semantic_model)}")
        
        print("✅ Model loading successful!")
        return True
        
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting Debug Tests...\n")
    
    # Test 1: Model Loading
    model_ok = test_model_loading()
    
    if model_ok:
        # Test 2: Technical Evaluator
        eval_ok = test_technical_evaluator()
        
        if eval_ok:
            print("\n✅ All systems operational! Flask app should work correctly.")
        else:
            print("\n❌ Technical evaluator issues found.")
    else:
        print("\n❌ Model loading issues found.")
    
    print("\n🏁 Debug tests complete.")