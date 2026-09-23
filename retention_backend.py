import sqlite3
import os
import random
import json
import re
from werkzeug.utils import secure_filename
import PyPDF2

try:
    import docx
except ImportError:
    docx = None

DB_PATH = 'dataset/retention.db'
UPLOAD_FOLDER = 'dataset/uploads'

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Document / Materials Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS StudyMaterial (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            filepath TEXT,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            extracted_text TEXT
        )
    ''')
    
    # Quiz Metadata Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Quiz (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            material_id INTEGER,
            questions_json TEXT,
            next_revision_date TIMESTAMP,
            FOREIGN KEY (material_id) REFERENCES StudyMaterial (id)
        )
    ''')
    
    # User Attempt / Progress Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS UserProgress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            quiz_id INTEGER,
            score REAL,
            attempt_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (quiz_id) REFERENCES Quiz (id)
        )
    ''')
    
    # Document Summary Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS DocumentSummary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            material_id INTEGER,
            summary_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (material_id) REFERENCES StudyMaterial (id)
        )
    ''')
    
    # Study Plan Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS StudyPlan (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            material_id INTEGER,
            day_no INTEGER,
            chapter_name TEXT,
            task_type TEXT,
            duration INTEGER,
            status TEXT DEFAULT 'Pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (material_id) REFERENCES StudyMaterial (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def extract_text(filepath):
    """Extract text from supported file formats robustly."""
    ext = filepath.split('.')[-1].lower()
    text = ""
    try:
        if ext == 'pdf':
            with open(filepath, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                limit = min(20, len(reader.pages))
                for i in range(limit):
                    extracted = reader.pages[i].extract_text()
                    if extracted:
                        text += extracted + "\n"
        elif ext in ['doc', 'docx'] and docx:
            doc = docx.Document(filepath)
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif ext in ['ppt', 'pptx']:
            try:
                import pptx
                prs = pptx.Presentation(filepath)
                for slide in prs.slides:
                    for shape in slide.shapes:
                        if hasattr(shape, "text"):
                            text += shape.text + "\n"
            except ImportError:
                print("python-pptx not installed for PPT extraction")
            except Exception as e:
                print(f"PPTx parsing soft error: {e}")
        elif ext == 'txt':
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
    except Exception as e:
        print(f"Extraction error: {e}")
        
    if text:
        # Purge non-ascii binary artifacts and excessive spacing
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
    return text

def process_upload(file):
    """Save file, extract text, and insert into DB."""
    init_db()
    
    if not file:
        return None, "No file provided"
        
    filename = secure_filename(file.filename)
    
    # VALIDATION: Check extension
    ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt', 'ppt', 'pptx'}
    if '.' not in filename or filename.split('.')[-1].lower() not in ALLOWED_EXTENSIONS:
        return None, "Invalid file format."
        
    # VALIDATION: Prevent Duplicate Uploads
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM StudyMaterial WHERE filename = ?", (filename,))
        existing = cursor.fetchone()
        conn.close()
        if existing:
            return existing[0], "File already uploaded, skipping."
    except Exception as e:
        print("Duplicate DB check error:", e)

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    # Extract text from document
    text = extract_text(filepath)
    
    # Store Document in Database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO StudyMaterial (filename, filepath, extracted_text) VALUES (?, ?, ?)",
        (filename, filepath, text)
    )
    material_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    # Automatically generate a quiz based on text
    generate_quiz_for_material(material_id, text)
    
    # Generate document summary for preview
    generate_document_summary(material_id, text)
    
    return material_id, "Success"

def generate_quiz_for_material(material_id, text):
    """Generate automated quizzes mimicking academic assessments (Option A Algorithm).
       Designed cleanly so an LLM generator can easily swap in later."""
    if not text:
        return
        
    sentences = text.split('.')
    # Culling extremely short fragments or massively run-on OCR errors
    sentences = [s.strip() for s in sentences if 40 < len(s.strip()) < 350]
    
    if len(sentences) < 5:
        print(f"Skipped quiz generation for {material_id}: Insufficient logic matched.")
        return
        
    questions = []
    # Identify strong vocabulary words (proxies for 'concepts')
    all_long_words = [w for w in text.split() if w.isalpha() and len(w) > 6]
    if not all_long_words:
        print(f"Skipped quiz generation: Document lacks strong vocabulary.")
        return
    
    # Process up to 10 meaningful assessment nodes
    sample_size = min(10, len(sentences))
    valid_sentences = random.sample(sentences, sample_size)
    
    for i, s in enumerate(valid_sentences):
        words = s.split()
        target_words = [w for w in words if w.isalpha() and len(w) > 5]
        if not target_words:
            continue
            
        answer = random.choice(target_words)
        
        # Difficulty Engine (Heuristic simulation)
        if len(s) > 180 and len(answer) > 8:
            difficulty = "Hard"
            marks = 3
        elif len(s) > 100:
            difficulty = "Medium"
            marks = 2
        else:
            difficulty = "Easy"
            marks = 1
            
        # Topic Identifier (Longest word acts as simulated concept hub)
        topic = sorted(target_words, key=len, reverse=True)[0] if target_words else "General Analysis"

        # Determine Structural Question Format
        q_type = random.choice(["mcq", "tf", "mcq"]) 
        
        if q_type == "mcq":
            question_text = s.replace(answer, "__________", 1) + "?"
            other_words = [w for w in all_long_words if w.lower() != answer.lower()]
            # Distractor options
            options = random.sample(other_words, min(3, len(other_words))) if len(other_words) >= 3 else ['Analysis', 'Metric', 'Variant']
            options.append(answer)
            random.shuffle(options)
            
            questions.append({
                'question': f"Identify the missing academic concept: {question_text}",
                'options': options,
                'answer': answer,
                'type': 'mcq',
                'difficulty': difficulty,
                'marks': marks,
                'topic': topic.capitalize()
            })
        else:
            # True / False Generator
            is_true = random.choice([True, False])
            if is_true:
                question_text = f"Evaluate whether the following conceptual statement is universally true:\n\"{s}.\""
                answer_flag = "True"
            else:
                # Malform the statement by injecting a random foreign concept
                fake_word = random.choice(all_long_words)
                fake_statement = s.replace(answer, fake_word.upper(), 1)
                question_text = f"Analyze the validity of this inference:\n\"{fake_statement}.\""
                answer_flag = "False"
                
            questions.append({
                'question': question_text,
                'options': ["True", "False"],
                'answer': answer_flag,
                'type': 'tf',
                'difficulty': difficulty,
                'marks': marks,
                'topic': topic.capitalize()
            })
            
    if questions:
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            questions_json = json.dumps(questions)
            cursor.execute(
                "INSERT INTO Quiz (material_id, questions_json, next_revision_date) VALUES (?, ?, datetime('now', '+1 day'))",
                (material_id, questions_json)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print("Quiz DB append error:", e)

def get_materials_list():
    """Retrieve all study documents, ordered by latest."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, filename, upload_date FROM StudyMaterial ORDER BY upload_date DESC")
    rows = cursor.fetchall()
    conn.close()
    
    return [{"id": r[0], "filename": r[1], "upload_date": r[2]} for r in rows]

def get_quizzes_list():
    """Retrieve quizzes along with material names."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT q.id, m.filename, q.next_revision_date 
        FROM Quiz q 
        JOIN StudyMaterial m ON q.material_id = m.id
        ORDER BY q.next_revision_date ASC
    """)
    rows = cursor.fetchall()
    conn.close()
    
    return [{"quiz_id": r[0], "filename": r[1], "next_revision": r[2]} for r in rows]

def get_quiz_data(quiz_id):
    """Fetch structured quiz json data by ID."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT questions_json, material_id FROM Quiz WHERE id = ?", (quiz_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return json.loads(row[0]), row[1]
    return [], None

def save_quiz_score(quiz_id, score):
    """Save user performance on a quiz to tracking tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO UserProgress (quiz_id, score) VALUES (?, ?)", (quiz_id, score))
    conn.commit()
    conn.close()

def generate_document_summary(material_id, text):
    """Heuristic summary generator for the uploaded document."""
    if not text:
        return None
        
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM DocumentSummary WHERE material_id = ?", (material_id,))
        existing = cursor.fetchone()
        conn.close()
        if existing:
            return
    except Exception as e:
        print("Error checking DB:", e)
        
    # Force clean for legacy text strings containing corrupted data
    clean_text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    if len(clean_text) < 30:
        return None # Document was mostly meaningless bytes
        
    sentences = [s.strip() for s in clean_text.split('.') if len(s.strip()) > 30]
    words = clean_text.split()
    
    # Estimated wait time -> ~200 words per minute
    reading_time = max(1, len(words) // 200)
    
    topics = []
    chunk_size = max(1, len(sentences) // 4)
    for i in range(4):
        if i * chunk_size < len(sentences):
            chunk = sentences[i*chunk_size : (i+1)*chunk_size]
            long_words = [w for w in " ".join(chunk).split() if len(w) > 7 and w.isalpha()]
            if long_words:
                topics.append((random.choice(long_words)).capitalize() + " Overview")
                
    if not topics:
        topics = ["Introduction", "Core Concepts", "Analysis", "Summary"]
        
    # Overview (first few sentences)
    overview = " ".join(sentences[:min(3, len(sentences))]) + "." if sentences else "No overview available."
    
    # Key Concepts (long words)
    long_words_all = list(set([w for w in words if len(w) > 8 and w.isalpha()]))
    concepts = random.sample(long_words_all, min(5, len(long_words_all))) if long_words_all else ["Data", "Analysis"]
    
    # Revision Notes (random sentences)
    revision_notes = random.sample(sentences, min(4, len(sentences))) if len(sentences) >= 4 else sentences

    summary_data = {
        'reading_time': f"{reading_time}",
        'topics': topics,
        'overview': overview,
        'concepts': concepts,
        'revision_notes': revision_notes
    }
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO DocumentSummary (material_id, summary_json) VALUES (?, ?)", 
                       (material_id, json.dumps(summary_data)))
        conn.commit()
        conn.close()
    except Exception as e:
        print("Summary DB insert error:", e)

def get_document_preview(material_id):
    """Retrieve document metadata and summary for preview."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT filename, upload_date FROM StudyMaterial WHERE id = ?", (material_id,))
    material_info = cursor.fetchone()
    
    cursor.execute("SELECT summary_json FROM DocumentSummary WHERE material_id = ?", (material_id,))
    summary_row = cursor.fetchone()
    conn.close()
    
    if not material_info:
        return None, None
        
    meta = {
        'filename': material_info[0],
        'upload_date': material_info[1]
    }
    
    summary = json.loads(summary_row[0]) if summary_row else None
    return meta, summary

def generate_study_plan(material_id, text, score=85):
    """Generates an intelligent study plan based on extracted text and past performance."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id FROM StudyPlan WHERE material_id = ?", (material_id,))
    if cursor.fetchone():
        conn.close()
        return  # already generated

    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 30]
    
    total_chapters = min(5, max(1, len(sentences) // 10))
    if total_chapters == 0:
        total_chapters = 1
        
    topics = []
    for i in range(total_chapters):
        topics.append(f"Chapter {i+1}: Focus on " + random.choice(['Analysis', 'Data', 'Metrics', 'Concepts', 'Logic']))

    plan_entries = []
    day = 1
    
    for topic in topics:
        plan_entries.append((material_id, day, topic, 'Reading', random.randint(30, 60), 'Pending'))
        day += 1
        
        if score < 60:
            plan_entries.append((material_id, day, f"Deep Revision: {topic}", 'Revision', 45, 'Pending'))
            day += 1
            
        plan_entries.append((material_id, day, f"Assessment: {topic}", 'Quiz', 20, 'Pending'))
        day += 1
        
    cursor.executemany('''
        INSERT INTO StudyPlan (material_id, day_no, chapter_name, task_type, duration, status)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', plan_entries)
    
    conn.commit()
    conn.close()

def get_study_plans_for_material(material_id):
    """Retrieves the study plan entries for a given material."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, day_no, chapter_name, task_type, duration, status 
        FROM StudyPlan 
        WHERE material_id = ? 
        ORDER BY day_no ASC
    ''', (material_id,))
    rows = cursor.fetchall()
    conn.close()
    
    plan = []
    for r in rows:
        plan.append({
            'plan_id': r[0],
            'day': r[1],
            'topic': r[2],
            'type': r[3],
            'duration': r[4],
            'status': r[5]
        })
    return plan

def get_all_materials_with_plans():
    """Returns materials and whether they have a study plan."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT m.id, m.filename, 
               (SELECT COUNT(*) FROM StudyPlan s WHERE s.material_id = m.id) as has_plan
        FROM StudyMaterial m
        ORDER BY m.upload_date DESC
    ''')
    rows = cursor.fetchall()
    conn.close()
    
    return [{'id': r[0], 'filename': r[1], 'has_plan': r[2] > 0} for r in rows]

def mark_study_plan_completed(plan_id, status):
    """Updates a single study plan task status."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE StudyPlan SET status = ? WHERE id = ?", (status, plan_id))
    conn.commit()
    conn.close()
