from flask import Flask, render_template, request, redirect, url_for
import joblib
import os
import numpy as np

# Import our custom modules
from data_processing import preprocess_input
from model import get_risk_level_and_suggestions
from retention_backend import (process_upload, get_materials_list, get_quizzes_list, 
                               get_quiz_data, save_quiz_score, get_document_preview,
                               generate_study_plan, get_study_plans_for_material,
                               get_all_materials_with_plans, mark_study_plan_completed,
                               DB_PATH)

app = Flask(__name__)

# Ensure directories exist
os.makedirs('static/images', exist_ok=True)
os.makedirs('dataset', exist_ok=True)

# Try to load model
MODEL_PATH = 'rf_model.pkl'

@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')

@app.route('/input')
def input_data():
    """Page to enter student data."""
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle form submission and show results."""
    if not os.path.exists(MODEL_PATH):
        return "Model not trained yet. Please run model.py first.", 500
        
    model = joblib.load(MODEL_PATH)
    
    try:
        # Get data from form
        input_dict = {
            'Previous_Score': float(request.form['previous_score']),
            'Study_Time_Hours': float(request.form['study_time']),
            'Num_Revisions': int(request.form['revisions']),
            'Time_Gap_Days': int(request.form['time_gap']),
            'Quiz_Score': float(request.form['quiz_score'])
        }
        
        # Preprocess input (Scale using saved scaler)
        X_scaled = preprocess_input(input_dict)
        
        # Predict
        predicted_score = model.predict(X_scaled)[0]
        predicted_score = round(min(max(predicted_score, 0), 100), 2)  # Clamp between 0-100
        
        # Get risk level and suggestions
        risk_level, suggestion = get_risk_level_and_suggestions(predicted_score)
        
        return render_template('results.html', 
                               score=predicted_score, 
                               risk=risk_level, 
                               suggestion=suggestion,
                               inputs=input_dict)
                               
    except Exception as e:
        return f"Error during prediction: {e}", 400

@app.route('/retention')
def retention():
    """Knowledge Retention Page - Phase 1 UI + Phase 3 Integration"""
    materials = get_materials_list()
    quizzes = get_quizzes_list()
    
    return render_template('retention_module.html', 
                            materials=materials,
                            quizzes=quizzes,
                            materials_count=len(materials),
                            quizzes_pending=len(quizzes),
                            score=85)

@app.route('/upload_material', methods=['POST'])
def upload_material():
    """Endpoint to handle Phase 3 file uploads."""
    if 'file' not in request.files:
        return "No file part", 400
        
    files = request.files.getlist('file')
    for file in files:
        if file and file.filename != '':
            process_upload(file)
            
    return redirect(url_for('retention'))

@app.route('/quiz/<int:quiz_id>')
def take_quiz(quiz_id):
    """Render a generated quiz for the student to take."""
    questions, _ = get_quiz_data(quiz_id)
    if not questions:
        return "Quiz not found", 404
        
    return render_template('quiz.html', quiz_id=quiz_id, questions=questions)

@app.route('/submit_quiz/<int:quiz_id>', methods=['POST'])
def submit_quiz(quiz_id):
    """Grade the academic assessment, capture analytical metrics, and map performance heuristics."""
    questions, _ = get_quiz_data(quiz_id)
    if not questions:
        return "Quiz not found", 404
        
    correct_count = 0
    total_questions = len(questions)
    marks_obtained = 0
    total_marks = 0
    wrong_topics = set()
    incorrect_breakdown = []
    
    for i, q in enumerate(questions):
        user_answer = request.form.get(f"q_{i}")
        q_marks = q.get('marks', 1)
        total_marks += q_marks
        
        if user_answer == q['answer']:
            correct_count += 1
            marks_obtained += q_marks
        else:
            topic = q.get('topic', 'General Analysis')
            wrong_topics.add(topic)
            incorrect_breakdown.append({
                'q_num': i+1,
                'question': q['question'],
                'user_ans': user_answer if user_answer else 'Skipped',
                'correct_ans': q['answer'],
                'topic': topic
            })
            
    score_percentage = (marks_obtained / total_marks) * 100 if total_marks > 0 else 0
    save_quiz_score(quiz_id, score_percentage)
    
    return render_template('quiz_result.html', 
                            score=score_percentage, 
                            correct=correct_count, 
                            total=total_questions,
                            marks_obtained=marks_obtained,
                            total_marks=total_marks,
                            wrong_topics=list(wrong_topics),
                            incorrect_breakdown=incorrect_breakdown)

@app.route('/dashboard')
def dashboard():
    """Dashboard to view overall analytics and generated visualizations."""
    
    # Check if images exist
    images = {
        'feature_importance': os.path.exists('static/images/feature_importance.png'),
        'actual_vs_pred': os.path.exists('static/images/actual_vs_pred.png'),
        'study_vs_retention': os.path.exists('static/images/study_vs_retention.png'),
        'retention_dist': os.path.exists('static/images/retention_dist.png'),
    }
    
    return render_template('dashboard.html', images=images)

# --- Phase 5 Future Scaffolding Routes ---
@app.route('/preview/<int:material_id>')
def preview_material(material_id):
    meta, summary = get_document_preview(material_id)
    if not meta:
        return "Material not found", 404
    return render_template('preview.html', meta=meta, summary=summary, material_id=material_id)

@app.route('/study_planner', methods=['GET'])
def study_planner():
    material_id = request.args.get('material_id', type=int)
    materials = get_all_materials_with_plans()
    plan_data = None
    if material_id:
        plan_data = get_study_plans_for_material(material_id)
    return render_template('study_planner.html', materials=materials, current_material_id=material_id, plan=plan_data)

@app.route('/create_study_plan/<int:material_id>', methods=['POST'])
def create_study_plan(material_id):
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT extracted_text FROM StudyMaterial WHERE id = ?", (material_id,))
    row = cursor.fetchone()
    conn.close()
    if row and row[0]:
        generate_study_plan(material_id, row[0], score=85)
    return redirect(url_for('study_planner', material_id=material_id))

@app.route('/update_plan_status/<int:plan_id>', methods=['POST'])
def update_plan_status(plan_id):
    status = request.form.get('status', 'Completed')
    mark_study_plan_completed(plan_id, status)
    material_id = request.form.get('material_id')
    return redirect(url_for('study_planner', material_id=material_id))

@app.route('/api/generate_summary/<int:material_id>', methods=['POST'])
def api_generate_summary(material_id):
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT extracted_text FROM StudyMaterial WHERE id = ?", (material_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row and row[0]:
        generate_document_summary(material_id, row[0])
        # Fetch the newly generated summary
        meta, summary = get_document_preview(material_id)
        if summary:
            return {"status": "success", "summary": summary}
            
    return {"status": "error", "message": "Failed to generate summary or document lacks text."}, 404

@app.route('/dev/chatbot')
def chatbot_stub():
    return "Chatbot Tutor is currently in development (Phase 5).", 200

@app.route('/dev/flashcards')
def flashcards_stub():
    return "Flashcards learning module is currently in development (Phase 5).", 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
