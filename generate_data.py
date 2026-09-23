import pandas as pd
import numpy as np
import os

def generate_student_data(num_samples=1000, output_file='dataset/student_data.csv'):
    np.random.seed(42)  # For reproducibility

    # Generate synthetic features
    student_ids = np.arange(1, num_samples + 1)
    
    # Quiz scores (0-100)
    quiz_scores = np.random.normal(75, 12, num_samples).clip(0, 100)
    
    # Previous test scores (0-100)
    previous_scores = np.random.normal(70, 15, num_samples).clip(0, 100)
    
    # Time spent studying (hours, 0-50)
    study_time = np.random.normal(25, 10, num_samples).clip(0, 50)
    
    # Number of revisions (0-10)
    num_revisions = np.random.randint(0, 11, num_samples)
    
    # Time gap between revisions (days, 1-30) - larger gap means lower retention
    time_gap = np.random.randint(1, 31, num_samples)
    
    # Formulate "Knowledge Retention Score" (target variable, 0-100)
    # Weights for the features impacting retention
    base_retention = (
        0.3 * quiz_scores +
        0.2 * previous_scores +
        (study_time / 50.0) * 20.0 +  # Max +20 from study time
        (num_revisions / 10.0) * 15.0 - # Max +15 from revisions
        (time_gap / 30.0) * 15.0      # Max -15 from time gap
    )
    
    # Add some random noise
    noise = np.random.normal(0, 5, num_samples)
    knowledge_retention_score = (base_retention + noise).clip(0, 100)
    
    # Final performance score (correlates heavily with retention)
    final_score = (0.8 * knowledge_retention_score + 0.2 * np.random.normal(80, 10, num_samples)).clip(0, 100)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Student_ID': student_ids,
        'Previous_Score': np.round(previous_scores, 2),
        'Study_Time_Hours': np.round(study_time, 2),
        'Num_Revisions': num_revisions,
        'Time_Gap_Days': time_gap,
        'Quiz_Score': np.round(quiz_scores, 2),
        'Final_Score': np.round(final_score, 2),
        'Knowledge_Retention_Score': np.round(knowledge_retention_score, 2)
    })
    
    # Calculate Risk Level based on retention score
    conditions = [
        (data['Knowledge_Retention_Score'] >= 75),
        (data['Knowledge_Retention_Score'] >= 50) & (data['Knowledge_Retention_Score'] < 75),
        (data['Knowledge_Retention_Score'] < 50)
    ]
    choices = ['Low Risk', 'Medium Risk', 'High Risk']
    data['Risk_Level'] = np.select(conditions, choices, default='Unknown')

    # Create directory if not exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to CSV
    data.to_csv(output_file, index=False)
    print(f"Generated synthetic dataset with {num_samples} records and saved to '{output_file}'.")

if __name__ == "__main__":
    generate_student_data()
