import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer, util
import os

class TechnicalEvaluator:
    def __init__(self):
        # Load models
        self.complexity_model = joblib.load("model/next_complexity_model.pkl")
        self.semantic_model = SentenceTransformer("all-mpnet-base-v2")
        
        # Load question bank
        self.question_bank = pd.read_csv("model/model_b_full_labels.csv")
        
        # Experience mappings
        self.experience_mapping = {
            "intern": 0,
            "associate": 1,
            "software engineer": 2
        }
        
        self.experience_starting_score = {
            "intern": 2.0,
            "associate": 2.7,
            "software engineer": 3.2
        }
    
    def get_technical_question(self, experience_level, skills=None, current_complexity=None):
        """Get a technical question based on experience level and skills"""
        
        # Set default skills if none provided
        if skills is None:
            skills = ["java", "react", "javascript", "python"]
        
        skill_set = [skill.lower() for skill in skills]
        
        # Set starting complexity if not provided
        if current_complexity is None:
            current_complexity = self.experience_starting_score.get(experience_level.lower(), 2.0)
        
        # Filter questions by skill set
        subset = self.question_bank[
            self.question_bank["technology"].str.lower().isin(skill_set)
        ].copy()
        
        if subset.empty:
            # Fallback to any question if no skills match
            subset = self.question_bank.copy()
        
        # Find question closest to current complexity
        subset["score_diff"] = abs(subset["complexity_score"] - current_complexity)
        question_row = subset.sort_values("score_diff").iloc[0]
        
        return {
            "question": question_row["question_text"],
            "expected_answer": question_row["expected_answer"],
            "complexity_score": question_row["complexity_score"],
            "technology": question_row["technology"],
            "bloom_label": question_row["bloom_label"],
            "question_id": question_row.name
        }
    
    def evaluate_technical_answer(self, question, expected_answer, candidate_answer):
        """Evaluate technical correctness using semantic similarity"""
        
        if not candidate_answer or not candidate_answer.strip():
            return {"correctness": 0.0, "semantic_similarity": 0.0}
        
        try:
            # Calculate semantic similarity
            emb_expected = self.semantic_model.encode(expected_answer, convert_to_tensor=True)
            emb_candidate = self.semantic_model.encode(candidate_answer, convert_to_tensor=True)
            similarity = util.cos_sim(emb_expected, emb_candidate).item()
            
            # Scale similarity to 0-10 range
            correctness = round(similarity * 10, 2)
            
            return {
                "correctness": correctness,
                "semantic_similarity": round(similarity, 3),
                "technical_accuracy": correctness,
                "answer_length": len(candidate_answer.split()),
                "has_technical_terms": self._count_technical_terms(candidate_answer)
            }
            
        except Exception as e:
            print(f"Error in technical evaluation: {e}")
            return {"correctness": 0.0, "semantic_similarity": 0.0}
    
    def predict_next_complexity(self, question_text, expected_answer, answer_quality_score, 
                              current_complexity, experience_level):
        """Predict the next question complexity using the trained model"""
        
        try:
            qa_text = question_text + " " + expected_answer
            experience_encoded = self.experience_mapping.get(experience_level.lower(), 0)
            
            sample = pd.DataFrame([{
                "qa_text": qa_text,
                "complexity_score": current_complexity,
                "answer_quality_score": answer_quality_score / 10.0,  # Normalize to 0-1
                "experience_encoded": experience_encoded
            }])
            
            next_complexity = self.complexity_model.predict(sample)[0]
            return round(next_complexity, 2)
            
        except Exception as e:
            print(f"Error predicting next complexity: {e}")
            # Fallback: simple adjustment based on performance
            if answer_quality_score >= 7:
                return min(current_complexity + 0.3, 5.0)
            elif answer_quality_score >= 4:
                return current_complexity
            else:
                return max(current_complexity - 0.3, 1.0)
    
    def get_comprehensive_evaluation(self, question_data, candidate_answer, experience_level):
        """Get comprehensive evaluation including next complexity prediction"""
        
        # Basic technical evaluation
        tech_eval = self.evaluate_technical_answer(
            question_data["question"],
            question_data["expected_answer"],
            candidate_answer
        )
        
        # Predict next complexity
        next_complexity = self.predict_next_complexity(
            question_data["question"],
            question_data["expected_answer"],
            tech_eval["correctness"],
            question_data["complexity_score"],
            experience_level
        )
        
        # Combine results
        result = {
            "technical_accuracy": tech_eval["correctness"],
            "semantic_similarity": tech_eval.get("semantic_similarity", 0.0),
            "current_complexity": question_data["complexity_score"],
            "next_complexity": next_complexity,
            "technology": question_data["technology"],
            "bloom_level": question_data["bloom_label"],
            "answer_analysis": {
                "word_count": tech_eval.get("answer_length", 0),
                "technical_terms": tech_eval.get("has_technical_terms", 0),
                "completeness": self._assess_completeness(candidate_answer, question_data["expected_answer"])
            }
        }
        
        return result
    
    def _count_technical_terms(self, text):
        """Count technical terms in the answer"""
        technical_terms = [
            'api', 'database', 'algorithm', 'function', 'method', 'class', 'object',
            'array', 'string', 'boolean', 'integer', 'async', 'await', 'promise',
            'callback', 'closure', 'prototype', 'inheritance', 'polymorphism',
            'encapsulation', 'abstraction', 'framework', 'library', 'module',
            'component', 'service', 'controller', 'model', 'view', 'rest', 'json',
            'xml', 'http', 'https', 'sql', 'nosql', 'crud', 'mvc', 'mvvm'
        ]
        
        text_lower = text.lower()
        return sum(1 for term in technical_terms if term in text_lower)
    
    def _assess_completeness(self, candidate_answer, expected_answer):
        """Assess how complete the answer is compared to expected"""
        if not candidate_answer or not expected_answer:
            return 0.0
        
        candidate_words = set(candidate_answer.lower().split())
        expected_words = set(expected_answer.lower().split())
        
        if not expected_words:
            return 0.0
        
        overlap = len(candidate_words.intersection(expected_words))
        return round(overlap / len(expected_words), 2)

# Singleton instance
technical_evaluator = TechnicalEvaluator()