import os
import json
import csv
import re
from datetime import timedelta, datetime
from io import StringIO, BytesIO
import pandas as pd
import numpy as np
import joblib
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from flask import Flask, render_template, request, redirect, session, url_for, flash, send_file
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'your_super_super_secret_key_here')

# Ensure NLTK data is available
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    print("NLTK data downloaded successfully")
except Exception as e:
    print(f"Failed to download NLTK data: {e}")
    exit(1)

# Flask app setup
app = Flask(__name__)
app.secret_key = SECRET_KEY
app.permanent_session_lifetime = timedelta(minutes=30)

# Configuration
UPLOAD_FOLDER = 'static/resumes'
PARSED_RESUMES_JSON = 'parsed_resumes.json'
USERS_CSV = 'users.csv'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and vectorizer once
try:
    model = joblib.load('rf_resume_model.pkl')
    vectorizer = joblib.load('rf_vectorizer.pkl')
    print("Model and vectorizer loaded successfully")
except FileNotFoundError as e:
    print(f"Model or vectorizer file not found: {e}")
    exit(1)
except Exception as e:
    print(f"Error loading model or vectorizer: {e}")
    exit(1)

# NLTK setup
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Skills list (unchanged)
SKILLS = [
    "Python", "R", "SQL", "Machine Learning", "Deep Learning", "Pandas", "NumPy", "Scikit-learn", "TensorFlow",
    "Power BI", "Tableau", "Data Visualization", "Data Wrangling", "Statistics", "Big Data", "NLP",
    "Recruitment", "Employee Relations", "Talent Acquisition", "Performance Management", "HRIS", "Payroll",
    "Organizational Development", "Communication Skills", "Conflict Resolution", "Training & Development",
    "Legal Research", "Drafting", "Litigation", "Legal Compliance", "Case Analysis", "Negotiation", "Civil Law",
    "Criminal Law", "Contract Law", "Creativity", "Graphic Design", "Adobe Photoshop", "Illustrator", "Sketching",
    "Visual Storytelling", "Fine Arts", "Color Theory", "Art History", "Portfolio Management", "HTML", "CSS",
    "JavaScript", "UI/UX Design", "Adobe XD", "Figma", "Responsive Design", "Bootstrap", "Web Accessibility",
    "AutoCAD", "SolidWorks", "Ansys", "Mechanical Design", "Thermodynamics", "Fluid Mechanics", "MATLAB",
    "CNC Programming", "Problem-Solving", "Technical Drawing", "CRM Tools", "Customer Service", "Lead Generation",
    "Sales Strategy", "Market Analysis", "Product Knowledge", "Target Achievement", "Personal Training",
    "Nutrition", "Fitness Assessment", "Exercise Planning", "CPR & First Aid", "Coaching", "Wellness Programs",
    "Strength Training", "Flexibility Training", "STAAD Pro", "Project Management", "Structural Analysis",
    "Construction Management", "Surveying", "Estimation", "Site Supervision", "Safety Regulations", "Spring Boot",
    "Hibernate", "REST APIs", "OOP", "Maven", "Jenkins", "JUnit", "Git", "Microservices", "Multithreading",
    "Eclipse", "IntelliJ IDEA", "Requirement Gathering", "Documentation", "Agile", "JIRA", "Use Case Modeling",
    "BPMN", "Stakeholder Management", "MS Excel", "SAP ABAP", "SAP FICO", "SAP HANA", "SAP MM", "SAP SD", "BAPI",
    "BADI", "Data Migration", "SAP UI5", "SAP BASIS", "Functional Specifications", "Selenium", "TestNG",
    "Automation Frameworks", "API Testing", "Postman", "Cucumber", "Unit Testing", "CI/CD", "Circuit Design",
    "Power Systems", "PCB Design", "Electrical Maintenance", "SCADA", "PLC", "Control Systems", "Load Flow Analysis",
    "Wiring Diagrams", "Operations Management", "Supply Chain", "Strategic Planning", "Team Leadership",
    "Logistics", "Budgeting", "ERP Systems", "Lean Manufacturing", "Flask", "Web Scraping", "SQLite", "PostgreSQL",
    "Data Structures", "Docker", "Kubernetes", "Ansible", "Terraform", "AWS", "Azure", "Monitoring Tools",
    "Bash Scripting", "Shell Scripting", "Network Protocols", "Firewalls", "IDS", "IPS", "VPNs", "Cybersecurity",
    "Ethical Hacking", "Wireshark", "Network Monitoring", "ISO Standards", "Risk Assessment", "Project Planning",
    "MS Project", "Risk Management", "Reporting", "Project Lifecycle", "PMBOK", "Oracle", "MySQL", "MongoDB",
    "PL/SQL", "Data Modeling", "Backup & Recovery", "Database Administration", "Indexing", "Query Optimization",
    "HDFS", "MapReduce", "Hive", "Pig", "HBase", "Spark", "Sqoop", "Kafka", "YARN", "Cloudera", "Data Lake",
    "Informatica", "Talend", "Data Warehousing", "Data Mapping", "SSIS", "Data Quality", "Data Transformation",
    "DBMS", "C#", ".NET Framework", "ASP.NET", "MVC", "Entity Framework", "LINQ", "SQL Server", "Web APIs",
    "Visual Studio", "Azure DevOps", "Ethereum", "Solidity", "Smart Contracts", "Web3.js", "Hyperledger",
    "Cryptography", "DApps", "Consensus Algorithms", "Manual Testing", "Test Cases", "Bug Tracking", "SDLC",
    "STLC", "Regression Testing", "Performance Testing", "QA Processes"
]

job_categories = [
    'Data Science',
    'HR',
    'Advocate',
    'Arts',
    'Web Designing',
    'Mechanical Engineer',
    'Sales',
    'Health and fitness',
    'Civil Engineer',
    'Java Developer',
    'Business Analyst',
    'SAP Developer',
    'Automation Testing',
    'Electrical Engineering',
    'Operations Manager',
    'Python Developer',
    'DevOps Engineer',
    'Network Security Engineer',
    'PMO',
    'Database',
    'Hadoop',
    'ETL Developer',
    'DotNet Developer',
    'Blockchain',
    'Testing'
]

# Precompute category_skills_dict at startup
try:
    df = pd.read_csv("UpdatedResumeDataSet.csv")
    df['Resume'] = df['Resume'].astype(str)
    job_skill_map = df.groupby('Category')['Resume'].apply(lambda x: ' '.join(x)).reset_index()
    vectorized_resume = vectorizer.transform(job_skill_map['Resume'])
    feature_names = vectorizer.get_feature_names_out()
    print("Dataset loaded and processed successfully")
except FileNotFoundError as e:
    print(f"Dataset file not found: {e}")
    exit(1)
except Exception as e:
    print(f"Error processing dataset: {e}")
    exit(1)

def get_top_skills(vector, top_n=10):
    top_indices = np.array(vector.toarray())[0].argsort()[-top_n:][::-1]
    return [feature_names[i] for i in top_indices]

category_skills_dict = {
    row['Category']: get_top_skills(vectorized_resume[i])
    for i, row in job_skill_map.iterrows()
}

def load_users_from_csv(file_path=USERS_CSV):
    """Load users from CSV, creating the file with a default admin if it doesn't exist."""
    if not os.path.exists(file_path):
        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['username', 'password', 'role'])
                writer.writerow(['admin@gmail.com', 'admin123', 'admin'])
                print("Default admin user created in users.csv")
        except IOError as e:
            print(f"Failed to create users CSV: {e}")
            return {}
    users = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                users[row['username']] = {'password': row['password'], 'role': row['role']}
                print(f"Loaded user: {row['username']}")
    except (IOError, csv.Error) as e:
        print(f"Failed to read users CSV: {e}")
        return {}
    except Exception as e:
        print(f"Unexpected error reading users CSV: {e}")
        return {}
    return users

def add_user(email, password, role='user'):
    """Add a new user with a plaintext password."""
    try:
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            return False, "Invalid email format"
        if len(password) < 8:
            return False, "Password must be at least 8 characters"
        users = load_users_from_csv()
        if email in users:
            return False, "Email already registered"
        with open(USERS_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([email, password, role])
        return True, "User registered successfully"
    except Exception as e:
        print(f"Error in add_user: {e}")
        return False, "Failed to register user"

def cleaned_text(text):
    """Clean resume text for processing, preserving numbers for skills like 'Python 3'."""
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text).lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file with error handling."""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = " ".join(page.extract_text() or "" for page in reader.pages)
        return text
    except Exception as e:
        print(f"Failed to extract text from PDF: {e}")
        return ""

def extract_skills(text):
    """Extract skills from text based on the predefined SKILLS list."""
    text_words = set(re.findall(r'\b[\w\s&]+(?=\b|$)', text.lower()))
    return [skill for skill in SKILLS if skill.lower() in text_words]

def save_resume_data(email, filename, skills, category, upload_date):
    """Save resume data to JSON with error handling."""
    new_entry = {"email": email, "filename": filename, "skills": skills, "category": category, "upload_date": upload_date}
    data = []
    try:
        if os.path.exists(PARSED_RESUMES_JSON):
            with open(PARSED_RESUMES_JSON, 'r') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    print(f"PARSED_RESUMES_JSON content is not a list, resetting to empty list: {data}")
                    data = []
    except (IOError, json.JSONDecodeError) as e:
        print(f"Failed to read parsed resumes: {e}")
        data = []
    data.append(new_entry)
    try:
        with open(PARSED_RESUMES_JSON, 'w') as f:
            json.dump(data, f, indent=4)
    except IOError as e:
        print(f"Failed to write parsed resumes: {e}")

def process_resume_upload(file):
    """Process an uploaded resume file (PDF only)."""
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
            with open(filepath, 'rb') as f:
                text = extract_text_from_pdf(f)
            cleaned = cleaned_text(text)
            vect_text = vectorizer.transform([cleaned]).toarray()
            prediction = model.predict(vect_text)[0]
            skills = extract_skills(text)
            upload_date = datetime.utcnow().isoformat()
            return prediction, skills, filename, upload_date
        except Exception as e:
            print(f"Failed to process resume: {e}")
            return None, [], None, None
    return None, [], None, None

@app.route('/')
def home():
    """Render the home page."""
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"Error rendering home page: {e}")
        return "Error rendering home page. Check if index.html exists in the templates folder.", 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login."""
    try:
        if request.method == 'POST':
            username = request.form.get('username', '').strip()
            password = request.form.get('password', '').strip()
            role = request.form.get('role', '').strip()
            print(f"Login attempt: username={username}, role={role}")
            if not all([username, password, role]):
                flash("All fields are required!", 'error')
                return render_template('login.html')
            users = load_users_from_csv()
            print(f"Loaded users: {users}")
            user = users.get(username)
            if user is None:
                flash("User not found!", 'error')
                print(f"User {username} not found in users.csv")
                return render_template('login.html')
            if user['password'] == password and user['role'] == role:
                session.permanent = True
                session['username'] = username
                session['role'] = role
                flash("Login successful!", 'success')
                print(f"Login successful for {username}")
                return redirect('/admin' if role == 'admin' else '/user')
            else:
                flash("Invalid credentials or role!", 'error')
                print(f"Login failed: password match={user['password'] == password}, role match={user['role'] == role}")
                return render_template('login.html')
        return render_template('login.html')
    except Exception as e:
        print(f"Error in login route: {e}")
        return "Error in login route. Check server logs for details.", 500

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handle user registration."""
    try:
        if request.method == 'POST':
            email = request.form.get('email')
            password = request.form.get('password')
            role = request.form.get('role')
            success, message = add_user(email, password, role)
            if success:
                flash(message, 'success')
                session['username'] = email
                session['role'] = role
                return redirect('/user' if role == 'user' else '/admin')
            else:
                flash(message, 'error')
        return render_template('register.html')
    except Exception as e:
        print(f"Error in register route: {e}")
        return "Internal Server Error", 500

@app.route('/logout')
def logout():
    """Handle user logout."""
    try:
        session.clear()
        flash("Logged out successfully!", 'success')
        return redirect('/')
    except Exception as e:
        print(f"Error in logout route: {e}")
        return "Internal Server Error", 500

@app.route('/user', methods=['GET', 'POST'])
def user_dashboard():
    """Render the user dashboard and handle resume uploads."""
    try:
        if session.get('role') != 'user':
            flash("Access denied!", 'error')
            return redirect('/')
        prediction = None
        skills = []
        user_resumes = []
        try:
            with open(PARSED_RESUMES_JSON, 'r') as f:
                all_resumes = json.load(f)
                if not isinstance(all_resumes, list):
                    print(f"PARSED_RESUMES_JSON content is not a list in user_dashboard: {all_resumes}")
                    all_resumes = []
                user_resumes = [r for r in all_resumes if r.get('email') == session['username']]
        except (FileNotFoundError, json.JSONDecodeError):
            print("Failed to load resumes in user_dashboard, using empty list")
        except Exception as e:
            print(f"Unexpected error loading resumes in user_dashboard: {e}")
        if request.method == 'POST':
            uploaded_file = request.files.get('resume_file')
            prediction, skills, filename, upload_date = process_resume_upload(uploaded_file)
            if filename:
                save_resume_data(session['username'], filename, skills, prediction, upload_date)
                flash("Resume uploaded successfully!", 'success')
                return redirect('/user')
            else:
                flash("Invalid file format. Please upload a PDF file.", 'error')
        return render_template('dashboard_user.html', username=session['username'], prediction=prediction, skills=skills, user_resumes=user_resumes)
    except Exception as e:
        print(f"Error in user_dashboard route: {e}")
        return "Internal Server Error", 500

@app.route('/upload', methods=['GET', 'POST'])
def upload_resume():
    """Handle resume uploads and redirect to user dashboard."""
    try:
        if session.get('role') != 'user':
            flash("Access denied!", 'error')
            return redirect('/')
        if request.method == 'POST':
            uploaded_file = request.files.get('resume_file')
            prediction, skills, filename, upload_date = process_resume_upload(uploaded_file)
            if filename:
                save_resume_data(session['username'], filename, skills, prediction, upload_date)
                flash("Resume uploaded successfully!", 'success')
                return redirect('/user')
            else:
                flash("Invalid file format. Please upload a PDF file.", 'error')
                return redirect('/upload')
        return render_template('upload_resume.html')
    except Exception as e:
        print(f"Error in upload_resume route: {e}")
        return "Internal Server Error", 500

import logging
# ... (other imports remain unchanged)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ... (previous code up to SKILLS list remains unchanged)

# Precompute category_skills_dict at startup
try:
    df = pd.read_csv("UpdatedResumeDataSet.csv")
    df['Resume'] = df['Resume'].astype(str)
    job_skill_map = df.groupby('Category')['Resume'].apply(lambda x: ' '.join(x)).reset_index()
    vectorized_resume = vectorizer.transform(job_skill_map['Resume'])
    feature_names = vectorizer.get_feature_names_out()
    print("Dataset loaded and processed successfully")
except FileNotFoundError as e:
    print(f"Dataset file not found: {e}")
    exit(1)
except Exception as e:
    print(f"Error processing dataset: {e}")
    exit(1)

def get_top_skills(vector, top_n=10):
    """Get top skills from vector, filtered by SKILLS list."""
    top_indices = np.array(vector.toarray())[0].argsort()[-top_n:][::-1]
    # Get all feature names for top indices
    all_top_skills = [feature_names[i] for i in top_indices]
    # Filter to include only skills present in SKILLS (case-insensitive)
    filtered_skills = [
        skill for skill in all_top_skills
        if any(skill.lower() == s.lower() for s in SKILLS)
    ]
    # Preserve original case from SKILLS list
    final_skills = [
        next(s for s in SKILLS if s.lower() == skill.lower())
        for skill in filtered_skills
    ]
    logger.debug(f"Top skills filtered: {final_skills}")
    return final_skills[:top_n]

# Create category_skills_dict with filtered skills
category_skills_dict = {
    row['Category']: get_top_skills(vectorized_resume[i])
    for i, row in job_skill_map.iterrows()
}

# Remove categories with no valid skills
category_skills_dict = {
    k: v for k, v in category_skills_dict.items() if v
}
logger.info(f"Categories with valid skills: {list(category_skills_dict.keys())}")

# ... (other functions like load_users_from_csv, add_user, etc., remain unchanged)

@app.route('/recommendations', methods=['GET', 'POST'])
def recommendations():
    """Render job recommendations and suggest skills from SKILLS list based on selected category."""
    try:
        if session.get('role') != 'user':
            logger.warning("Access denied for non-user attempting /recommendations")
            flash("Access denied! Please log in as a user.", 'error')
            return redirect('/')
        
        recommended_skills = []
        selected_category = None

        if request.method == 'POST':
            selected_category = request.form.get('category', '').strip()
            logger.debug(f"Selected category: {selected_category}")
            
            if not selected_category:
                logger.warning("No category selected in POST request")
                flash("Please select a job category.", 'error')
            elif selected_category not in category_skills_dict:
                logger.warning(f"Invalid category selected: {selected_category}")
                flash(f"Invalid job category: {selected_category}.", 'error')
            else:
                recommended_skills = category_skills_dict.get(selected_category, [])
                logger.info(f"Recommended skills for {selected_category}: {recommended_skills}")
                if not recommended_skills:
                    flash(f"No relevant skills found for category '{selected_category}'. Try another category.", 'info')
                else:
                    flash(f"Recommended skills for {selected_category} loaded successfully!", 'success')
        
        job_categories = sorted(category_skills_dict.keys())
        if not job_categories:
            logger.error("No job categories available")
            flash("No job categories available. Please contact the administrator.", 'error')

        return render_template(
            'recommendations.html',
            job_categories=job_categories,
            recommended_skills=recommended_skills,
            selected_category=selected_category
        )
    
    except Exception as e:
        logger.error(f"Error in recommendations route: {str(e)}", exc_info=True)
        flash("An error occurred while loading recommendations. Please try again later.", 'error')
        return redirect('/user')

# ... (rest of the code remains unchanged)


@app.route('/admin')
def admin_dashboard():
    """Render the admin dashboard with statistics and charts."""
    try:
        print("Entering admin dashboard route")
        if session.get('role') != 'admin':
            flash("Access denied!", 'error')
            return redirect('/')
        resumes = []
        try:
            print("Attempting to read PARSED_RESUMES_JSON")
            with open(PARSED_RESUMES_JSON, 'r') as f:
                resumes = json.load(f)
                if not isinstance(resumes, list):
                    print(f"PARSED_RESUMES_JSON content is not a list in admin_dashboard: {resumes}")
                    resumes = []
                print(f"Successfully loaded {len(resumes)} resumes")
        except FileNotFoundError:
            print("PARSED_RESUMES_JSON not found, using empty list")
        except json.JSONDecodeError as e:
            print(f"Failed to parse PARSED_RESUMES_JSON: {e}, using empty list")
        except Exception as e:
            print(f"Unexpected error reading resumes: {e}, using empty list")
            resumes = []
        users = load_users_from_csv()
        print(f"Loaded users: {users}")
        total_resumes = len(resumes)
        total_users = len(users)
        recent_uploads = 0
        try:
            recent_uploads = len([r for r in resumes if datetime.fromisoformat(r.get('upload_date', '1970-01-01T00:00:00')) > datetime.utcnow() - timedelta(days=7)])
            print(f"Calculated {recent_uploads} recent uploads")
        except ValueError as e:
            print(f"Invalid date format in resumes: {e}, setting recent_uploads to 0")
        category_counts = {}
        for resume in resumes:
            category = resume.get('category', 'Unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
        chart_labels = json.dumps(list(category_counts.keys()))
        chart_data = json.dumps(list(category_counts.values()))
        print(f"Chart data: labels={chart_labels}, data={chart_data}")
        return render_template('dashboard_admin.html',
                             admin_email=session['username'],
                             total_resumes=total_resumes,
                             total_users=total_users,
                             recent_uploads=recent_uploads,
                             chart_labels=chart_labels,
                             chart_data=chart_data)
    except Exception as e:
        print(f"Error in admin_dashboard route: {e}")
        return "Error in admin dashboard. Check server logs for details.", 500


from urllib.parse import quote

@app.route('/resumes')
def view_resumes():
    # Absolute path to the resumes folder
    resumes_folder = r'D:\OneDrive\Desktop\PAI Project\static\resumes'
    
    # Get all filenames in the folder
    try:
        resume_files = os.listdir(resumes_folder)
        print(f"Debug: Files found - {resume_files}")  # Debug: Print the list of files
    except Exception as e:
        print(f"Error: Unable to read directory {resumes_folder}. Exception: {e}")
        resume_files = []
    
    # Make sure only valid PDF files are added
    resumes = []
    for filename in resume_files:
        if filename.lower().endswith('.pdf'):  # Case insensitive check for PDFs
            resumes.append({
                'filename': filename,
                'path': f'/static/resumes/{filename}',  # Flask will serve this path
            })
    
    # Debug: Print resumes list
    print(f"Debug: Resumes list - {resumes}")
    
    # Pass the resume data to the template
    return render_template('view_resumes.html', resumes=resumes)


from urllib.parse import unquote
@app.route('/download_resume/<path:filename>')
def download_resume(filename):
    decoded_filename = unquote(filename)
    file_path = os.path.join('static', 'resumes', decoded_filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        flash('File not found.', 'error')
        return redirect(url_for('view_resumes'))

@app.route('/manage_users', methods=['GET', 'POST'])
def manage_users():
    """Allow admins to manage users (delete functionality)."""
    try:
        if session.get('role') != 'admin':
            flash("Access denied!", 'error')
            return redirect('/')
        users = load_users_from_csv()
        if request.method == 'POST':
            action = request.form.get('action')
            username = request.form.get('username')
            if action == 'delete' and username in users:
                try:
                    with open(USERS_CSV, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(['username', 'password', 'role'])
                        for u, data in users.items():
                            if u != username:
                                writer.writerow([u, data['password'], data['role']])
                    flash(f"User {username} deleted successfully!", 'success')
                except IOError as e:
                    print(f"Failed to update users CSV: {e}")
                    flash("Failed to delete user!", 'error')
            else:
                flash("Invalid action or user not found!", 'error')
            return redirect('/manage_users')
        return render_template('manage_users.html', users=users)
    except Exception as e:
        print(f"Error in manage_users route: {e}")
        return "Internal Server Error", 500

@app.route('/filter_by_skills', methods=['GET', 'POST'])
def filter_by_skills():
    """Allow admins to filter resumes by skills based on job category."""
    try:
        # Check if the user is an admin
        if session.get('role') != 'admin':
            flash("Access denied! Please log in as admin.", 'error')
            return redirect('/login')

        selected_category = request.form.get('category') or request.args.get('category')
        print(f"Selected category: {selected_category}")

        matched_resumes = []
        resumes = []

        # Load resumes if the file exists and is not empty
        with open(PARSED_RESUMES_JSON, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content:
                resumes = json.loads(content)
                print(f"Loaded {len(resumes)} resumes")

        # Filter resumes by selected category
        if selected_category:
            for resume in resumes:
                if resume.get('category') == selected_category:
                    matched_resumes.append(resume)
                    print(f"Matched resume for {selected_category}: {resume.get('email')}")
        else:
            flash("No category selected!", 'error')

        return render_template('filter_by_skills.html',
                               job_categories=job_categories,
                               matched_resumes=matched_resumes,
                               selected_category=selected_category)
    except Exception as e:
        print(f"Error in filter_by_skills route: {e}")
        return "Error in filter by skills. Check server logs for details.", 500


@app.route('/export_filtered_resumes')
def export_filtered_resumes():
    """Export filtered resumes as a CSV file."""
    try:
        if session.get('role') != 'admin':
            flash("Access denied!", 'error')
            return redirect('/')
        category = request.args.get('category')
        if not category:
            flash("No category selected for export!", 'error')
            return redirect('/filter_by_skills')
        resumes = []
        try:
            with open(PARSED_RESUMES_JSON, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    resumes = []
                else:
                    resumes = json.load(f)
                    if not isinstance(resumes, list):
                        print(f"PARSED_RESUMES_JSON content is not a list in export_filtered_resumes: {resumes}")
                        resumes = []
        except (FileNotFoundError, json.JSONDecodeError):
            resumes = []
        except Exception as e:
            print(f"Unexpected error reading resumes in export_filtered_resumes: {e}")
            resumes = []
        matched = [r for r in resumes if set(category_skills_dict.get(category, [])).issubset(set(map(str.lower, r.get('skills', []))))]
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(['Email', 'Filename', 'Skills'])
        for r in matched:
            writer.writerow([r.get('email', ''), r.get('filename', ''), ', '.join(r.get('skills', []))])
        output.seek(0)
        return send_file(
            BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'{category}_resumes.csv'
        )
    except Exception as e:
        print(f"Error in export_filtered_resumes route: {e}")
        return "Internal Server Error", 500

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True)

# import os
# import json
# import csv
# import re
# from datetime import timedelta, datetime
# from io import StringIO, BytesIO
# import pandas as pd
# import numpy as np
# import joblib
# import PyPDF2
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from flask import Flask, render_template, request, redirect, session, url_for, flash, send_file
# from werkzeug.utils import secure_filename
# from dotenv import load_dotenv
# import logging

# # Load environment variables
# load_dotenv()
# SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'your_super_super_secret_key_here')

# # Configure logging
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Ensure NLTK data is available
# try:
#     nltk.download('stopwords', quiet=True)
#     nltk.download('wordnet', quiet=True)
#     logger.info("NLTK data downloaded successfully")
# except Exception as e:
#     logger.error(f"Failed to download NLTK data: {e}")
#     exit(1)

# # Flask app setup
# app = Flask(__name__)
# app.secret_key = SECRET_KEY
# app.permanent_session_lifetime = timedelta(minutes=30)

# # Configuration
# UPLOAD_FOLDER = 'static/resumes'
# PARSED_RESUMES_JSON = 'parsed_resumes.json'
# USERS_CSV = 'users.csv'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # Load model and vectorizer once
# try:
#     model = joblib.load('rf_resume_model.pkl')
#     vectorizer = joblib.load('rf_vectorizer.pkl')
#     logger.info("Model and vectorizer loaded successfully")
# except FileNotFoundError as e:
#     logger.error(f"Model or vectorizer file not found: {e}")
#     exit(1)
# except Exception as e:
#     logger.error(f"Error loading model or vectorizer: {e}")
#     exit(1)

# # NLTK setup
# lemmatizer = WordNetLemmatizer()
# stop_words = set(stopwords.words('english'))

# # Skills list
# SKILLS = [
#     "Python", "R", "SQL", "Machine Learning", "Deep Learning", "Pandas", "NumPy", "Scikit-learn", "TensorFlow",
#     "Power BI", "Tableau", "Data Visualization", "Data Wrangling", "Statistics", "Big Data", "NLP",
#     "Recruitment", "Employee Relations", "Talent Acquisition", "Performance Management", "HRIS", "Payroll",
#     "Organizational Development", "Communication Skills", "Conflict Resolution", "Training & Development",
#     "Legal Research", "Drafting", "Litigation", "Legal Compliance", "Case Analysis", "Negotiation", "Civil Law",
#     "Criminal Law", "Contract Law", "Creativity", "Graphic Design", "Adobe Photoshop", "Illustrator", "Sketching",
#     "Visual Storytelling", "Fine Arts", "Color Theory", "Art History", "Portfolio Management", "HTML", "CSS",
#     "JavaScript", "UI/UX Design", "Adobe XD", "Figma", "Responsive Design", "Bootstrap", "Web Accessibility",
#     "AutoCAD", "SolidWorks", "Ansys", "Mechanical Design", "Thermodynamics", "Fluid Mechanics", "MATLAB",
#     "CNC Programming", "Problem-Solving", "Technical Drawing", "CRM Tools", "Customer Service", "Lead Generation",
#     "Sales Strategy", "Market Analysis", "Product Knowledge", "Target Achievement", "Personal Training",
#     "Nutrition", "Fitness Assessment", "Exercise Planning", "CPR & First Aid", "Coaching", "Wellness Programs",
#     "Strength Training", "Flexibility Training", "STAAD Pro", "Project Management", "Structural Analysis",
#     "Construction Management", "Surveying", "Estimation", "Site Supervision", "Safety Regulations", "Spring Boot",
#     "Hibernate", "REST APIs", "OOP", "Maven", "Jenkins", "JUnit", "Git", "Microservices", "Multithreading",
#     "Eclipse", "IntelliJ IDEA", "Requirement Gathering", "Documentation", "Agile", "JIRA", "Use Case Modeling",
#     "BPMN", "Stakeholder Management", "MS Excel", "SAP ABAP", "SAP FICO", "SAP HANA", "SAP MM", "SAP SD", "BAPI",
#     "BADI", "Data Migration", "SAP UI5", "SAP BASIS", "Functional Specifications", "Selenium", "TestNG",
#     "Automation Frameworks", "API Testing", "Postman", "Cucumber", "Unit Testing", "CI/CD", "Circuit Design",
#     "Power Systems", "PCB Design", "Electrical Maintenance", "SCADA", "PLC", "Control Systems", "Load Flow Analysis",
#     "Wiring Diagrams", "Operations Management", "Supply Chain", "Strategic Planning", "Team Leadership",
#     "Logistics", "Budgeting", "ERP Systems", "Lean Manufacturing", "Flask", "Web Scraping", "SQLite", "PostgreSQL",
#     "Data Structures", "Docker", "Kubernetes", "Ansible", "Terraform", "AWS", "Azure", "Monitoring Tools",
#     "Bash Scripting", "Shell Scripting", "Network Protocols", "Firewalls", "IDS", "IPS", "VPNs", "Cybersecurity",
#     "Ethical Hacking", "Wireshark", "Network Monitoring", "ISO Standards", "Risk Assessment", "Project Planning",
#     "MS Project", "Risk Management", "Reporting", "Project Lifecycle", "PMBOK", "Oracle", "MySQL", "MongoDB",
#     "PL/SQL", "Data Modeling", "Backup & Recovery", "Database Administration", "Indexing", "Query Optimization",
#     "HDFS", "MapReduce", "Hive", "Pig", "HBase", "Spark", "Sqoop", "Kafka", "YARN", "Cloudera", "Data Lake",
#     "Informatica", "Talend", "Data Warehousing", "Data Mapping", "SSIS", "Data Quality", "Data Transformation",
#     "DBMS", "C#", ".NET Framework", "ASP.NET", "MVC", "Entity Framework", "LINQ", "SQL Server", "Web APIs",
#     "Visual Studio", "Azure DevOps", "Ethereum", "Solidity", "Smart Contracts", "Web3.js", "Hyperledger",
#     "Cryptography", "DApps", "Consensus Algorithms", "Manual Testing", "Test Cases", "Bug Tracking", "SDLC",
#     "STLC", "Regression Testing", "Performance Testing", "QA Processes"
# ]

# Precompute category_skills_dict at startup
# try:
#     df = pd.read_csv("UpdatedResumeDataSet.csv")
#     df['Resume'] = df['Resume'].astype(str)
#     job_skill_map = df.groupby('Category')['Resume'].apply(lambda x: ' '.join(x)).reset_index()
#     vectorized_resume = vectorizer.transform(job_skill_map['Resume'])
#     feature_names = vectorizer.get_feature_names_out()
#     logger.info("Dataset loaded and processed successfully")
# except FileNotFoundError as e:
#     logger.error(f"Dataset file not found: {e}")
#     exit(1)
# except Exception as e:
#     logger.error(f"Error processing dataset: {e}")
#     exit(1)

# def get_top_skills(vector, top_n=10):
#     top_indices = np.array(vector.toarray())[0].argsort()[-top_n:][::-1]
#     # Filter skills to match SKILLS list and preserve original case
#     all_top_skills = [feature_names[i].lower() for i in top_indices]
#     filtered_skills = [
#         skill for skill in all_top_skills
#         if any(skill.lower() == s.lower() for s in SKILLS)
#     ]
#     # Map to original case from SKILLS
#     final_skills = [
#         next(s for s in SKILLS if s.lower() == skill.lower())
#         for skill in filtered_skills
#     ]
#     logger.debug(f"Top skills filtered: {final_skills}")
#     return final_skills[:top_n]

# category_skills_dict = {
#     row['Category']: get_top_skills(vectorized_resume[i])
#     for i, row in job_skill_map.iterrows()
# }

# # Use category_skills_dict keys as job_categories
# job_categories = sorted(category_skills_dict.keys())
# logger.info(f"Job categories: {job_categories}")

# def load_users_from_csv(file_path=USERS_CSV):
#     if not os.path.exists(file_path):
#         try:
#             with open(file_path, 'w', newline='', encoding='utf-8') as f:
#                 writer = csv.writer(f)
#                 writer.writerow(['username', 'password', 'role'])
#                 writer.writerow(['admin@gmail.com', 'admin123', 'admin'])
#                 logger.info("Default admin user created in users.csv")
#         except IOError as e:
#             logger.error(f"Failed to create users CSV: {e}")
#             return {}
#     users = {}
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             reader = csv.DictReader(f)
#             for row in reader:
#                 users[row['username']] = {'password': row['password'], 'role': row['role']}
#                 logger.debug(f"Loaded user: {row['username']}")
#     except (IOError, csv.Error) as e:
#         logger.error(f"Failed to read users CSV: {e}")
#         return {}
#     except Exception as e:
#         logger.error(f"Unexpected error reading users CSV: {e}")
#         return {}
#     return users

# def add_user(email, password, role='user'):
#     try:
#         if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
#             return False, "Invalid email format"
#         if len(password) < 8:
#             return False, "Password must be at least 8 characters"
#         users = load_users_from_csv()
#         if email in users:
#             return False, "Email already registered"
#         with open(USERS_CSV, 'a', newline='', encoding='utf-8') as f:
#             writer = csv.writer(f)
#             writer.writerow([email, password, role])
#         return True, "User registered successfully"
#     except Exception as e:
#         logger.error(f"Error in add_user: {e}")
#         return False, "Failed to register user"

# def cleaned_text(text):
#     text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text).lower()
#     words = text.split()
#     words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
#     return " ".join(words)

# def extract_text_from_pdf(pdf_file):
#     try:
#         reader = PyPDF2.PdfReader(pdf_file)
#         text = " ".join(page.extract_text() or "" for page in reader.pages)
#         return text
#     except Exception as e:
#         logger.error(f"Failed to extract text from PDF: {e}")
#         return ""

# def extract_skills(text):
#     text_words = set(re.findall(r'\b[\w\s&]+(?=\b|$)', text.lower()))
#     return [skill for skill in SKILLS if skill.lower() in text_words]

# def save_resume_data(email, filename, skills, category, upload_date):
#     new_entry = {"email": email, "filename": filename, "skills": skills, "category": category, "upload_date": upload_date}
#     data = []
#     try:
#         if os.path.exists(PARSED_RESUMES_JSON):
#             with open(PARSED_RESUMES_JSON, 'r', encoding='utf-8') as f:
#                 content = f.read().strip()
#                 data = json.loads(content) if content else []
#                 if not isinstance(data, list):
#                     logger.warning(f"PARSED_RESUMES_JSON content is not a list, resetting to empty list: {data}")
#                     data = []
#     except (IOError, json.JSONDecodeError) as e:
#         logger.error(f"Failed to read parsed resumes: {e}")
#         data = []
#     data.append(new_entry)
#     try:
#         with open(PARSED_RESUMES_JSON, 'w', encoding='utf-8') as f:
#             json.dump(data, f, indent=4)
#     except IOError as e:
#         logger.error(f"Failed to write parsed resumes: {e}")

# def process_resume_upload(file):
#     if file and file.filename.endswith('.pdf'):
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         try:
#             file.save(filepath)
#             with open(filepath, 'rb') as f:
#                 text = extract_text_from_pdf(f)
#             cleaned = cleaned_text(text)
#             vect_text = vectorizer.transform([cleaned]).toarray()
#             prediction = model.predict(vect_text)[0]
#             skills = extract_skills(text)
#             upload_date = datetime.utcnow().isoformat()
#             # Map prediction to a category in job_categories if possible
#             prediction_lower = prediction.lower()
#             matched_category = next((cat for cat in job_categories if cat.lower() == prediction_lower), prediction)
#             return matched_category, skills, filename, upload_date
#         except Exception as e:
#             logger.error(f"Failed to process resume: {e}")
#             return None, [], None, None
#     return None, [], None, None

# @app.route('/')
# def home():
#     try:
#         return render_template('index.html')
#     except Exception as e:
#         logger.error(f"Error rendering home page: {e}")
#         return "Error rendering home page. Check if index.html exists in the templates folder.", 500

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     try:
#         if request.method == 'POST':
#             username = request.form.get('username', '').strip()
#             password = request.form.get('password', '').strip()
#             role = request.form.get('role', '').strip()
#             logger.debug(f"Login attempt: username={username}, role={role}")
#             if not all([username, password, role]):
#                 flash("All fields are required!", 'error')
#                 return render_template('login.html')
#             users = load_users_from_csv()
#             user = users.get(username)
#             if user is None:
#                 flash("User not found!", 'error')
#                 logger.info(f"User {username} not found in users.csv")
#                 return render_template('login.html')
#             if user['password'] == password and user['role'] == role:
#                 session.permanent = True
#                 session['username'] = username
#                 session['role'] = role
#                 flash("Login successful!", 'success')
#                 logger.info(f"Login successful for {username}")
#                 return redirect('/admin' if role == 'admin' else '/user')
#             else:
#                 flash("Invalid credentials or role!", 'error')
#                 logger.info(f"Login failed for {username}: password match={user['password'] == password}, role match={user['role'] == role}")
#                 return render_template('login.html')
#         return render_template('login.html')
#     except Exception as e:
#         logger.error(f"Error in login route: {e}")
#         return "Error in login route. Check server logs for details.", 500

# @app.route('/register', methods=['GET', 'POST'])
# def register():
#     try:
#         if request.method == 'POST':
#             email = request.form.get('email')
#             password = request.form.get('password')
#             role = request.form.get('role')
#             success, message = add_user(email, password, role)
#             if success:
#                 flash(message, 'success')
#                 session['username'] = email
#                 session['role'] = role
#                 return redirect('/user' if role == 'user' else '/admin')
#             else:
#                 flash(message, 'error')
#         return render_template('register.html')
#     except Exception as e:
#         logger.error(f"Error in register route: {e}")
#         return "Internal Server Error", 500

# @app.route('/logout')
# def logout():
#     try:
#         session.clear()
#         flash("Logged out successfully!", 'success')
#         return redirect('/')
#     except Exception as e:
#         logger.error(f"Error in logout route: {e}")
#         return "Internal Server Error", 500

# @app.route('/user', methods=['GET', 'POST'])
# def user_dashboard():
#     try:
#         if session.get('role') != 'user':
#             flash("Access denied!", 'error')
#             return redirect('/')
#         prediction = None
#         skills = []
#         user_resumes = []
#         try:
#             with open(PARSED_RESUMES_JSON, 'r', encoding='utf-8') as f:
#                 content = f.read().strip()
#                 all_resumes = json.loads(content) if content else []
#                 if not isinstance(all_resumes, list):
#                     logger.warning(f"PARSED_RESUMES_JSON content is not a list in user_dashboard: {all_resumes}")
#                     all_resumes = []
#                 user_resumes = [r for r in all_resumes if r.get('email') == session['username']]
#         except (FileNotFoundError, json.JSONDecodeError):
#             logger.info("Failed to load resumes in user_dashboard, using empty list")
#         except Exception as e:
#             logger.error(f"Unexpected error loading resumes in user_dashboard: {e}")
#         if request.method == 'POST':
#             uploaded_file = request.files.get('resume_file')
#             prediction, skills, filename, upload_date = process_resume_upload(uploaded_file)
#             if filename:
#                 save_resume_data(session['username'], filename, skills, prediction, upload_date)
#                 flash("Resume uploaded successfully!", 'success')
#                 return redirect('/user')
#             else:
#                 flash("Invalid file format. Please upload a PDF file.", 'error')
#         return render_template('dashboard_user.html', username=session['username'], prediction=prediction, skills=skills, user_resumes=user_resumes)
#     except Exception as e:
#         logger.error(f"Error in user_dashboard route: {e}")
#         return "Internal Server Error", 500

# @app.route('/upload', methods=['GET', 'POST'])
# def upload_resume():
#     try:
#         if session.get('role') != 'user':
#             flash("Access denied!", 'error')
#             return redirect('/')
#         if request.method == 'POST':
#             uploaded_file = request.files.get('resume_file')
#             prediction, skills, filename, upload_date = process_resume_upload(uploaded_file)
#             if filename:
#                 save_resume_data(session['username'], filename, skills, prediction, upload_date)
#                 flash("Resume uploaded successfully!", 'success')
#                 return redirect('/user')
#             else:
#                 flash("Invalid file format. Please upload a PDF file.", 'error')
#                 return redirect('/upload')
#         return render_template('upload_resume.html')
#     except Exception as e:
#         logger.error(f"Error in upload_resume route: {e}")
#         return "Internal Server Error", 500

# @app.route('/recommendations', methods=['GET', 'POST'])
# def recommendations():
#     try:
#         if session.get('role') != 'user':
#             logger.warning("Access denied for non-user attempting /recommendations")
#             flash("Access denied! Please log in as a user.", 'error')
#             return redirect('/')
#         recommended_skills = []
#         selected_category = None
#         if request.method == 'POST':
#             selected_category = request.form.get('category', '').strip()
#             logger.debug(f"Selected category: {selected_category}")
#             if not selected_category:
#                 logger.warning("No category selected in POST request")
#                 flash("Please select a job category.", 'error')
#             elif selected_category not in category_skills_dict:
#                 logger.warning(f"Invalid category selected: {selected_category}")
#                 flash(f"Invalid job category: {selected_category}.", 'error')
#             else:
#                 recommended_skills = category_skills_dict.get(selected_category, [])
#                 logger.info(f"Recommended skills for {selected_category}: {recommended_skills}")
#                 if not recommended_skills:
#                     flash(f"No relevant skills found for category '{selected_category}'. Try another category.", 'info')
#                 else:
#                     flash(f"Recommended skills for {selected_category} loaded successfully!", 'success')
#         return render_template(
#             'recommendations.html',
#             job_categories=job_categories,
#             recommended_skills=recommended_skills,
#             selected_category=selected_category
#         )
#     except Exception as e:
#         logger.error(f"Error in recommendations route: {str(e)}")
#         flash("An error occurred while loading recommendations. Please try again later.", 'error')
#         return redirect('/user')

# @app.route('/admin')
# def admin_dashboard():
#     try:
#         if session.get('role') != 'admin':
#             flash("Access denied!", 'error')
#             return redirect('/')
#         resumes = []
#         try:
#             with open(PARSED_RESUMES_JSON, 'r', encoding='utf-8') as f:
#                 content = f.read().strip()
#                 resumes = json.loads(content) if content else []
#                 if not isinstance(resumes, list):
#                     logger.warning(f"PARSED_RESUMES_JSON content is not a list in admin_dashboard: {resumes}")
#                     resumes = []
#         except (FileNotFoundError, json.JSONDecodeError):
#             logger.info("Failed to load resumes in admin_dashboard, using empty list")
#         except Exception as e:
#             logger.error(f"Unexpected error reading resumes: {e}")
#             resumes = []
#         users = load_users_from_csv()
#         total_resumes = len(resumes)
#         total_users = len(users)
#         recent_uploads = 0
#         try:
#             recent_uploads = len([r for r in resumes if datetime.fromisoformat(r.get('upload_date', '1970-01-01T00:00:00')) > datetime.utcnow() - timedelta(days=7)])
#         except ValueError as e:
#             logger.error(f"Invalid date format in resumes: {e}")
#         category_counts = {}
#         for resume in resumes:
#             category = resume.get('category', 'Unknown')
#             category_counts[category] = category_counts.get(category, 0) + 1
#         chart_labels = json.dumps(list(category_counts.keys()))
#         chart_data = json.dumps(list(category_counts.values()))
#         return render_template('dashboard_admin.html',
#                              admin_email=session['username'],
#                              total_resumes=total_resumes,
#                              total_users=total_users,
#                              recent_uploads=recent_uploads,
#                              chart_labels=chart_labels,
#                              chart_data=chart_data)
#     except Exception as e:
#         logger.error(f"Error in admin_dashboard route: {e}")
#         return "Error in admin dashboard. Check server logs for details.", 500

# from urllib.parse import quote

# @app.route('/resumes')
# def view_resumes():
#     try:
#         if session.get('role') != 'admin':
#             flash("Access denied!", 'error')
#             return redirect('/')
#         resumes = []
#         try:
#             with open(PARSED_RESUMES_JSON, 'r', encoding='utf-8') as f:
#                 content = f.read().strip()
#                 resumes = json.loads(content) if content else []
#                 if not isinstance(resumes, list):
#                     logger.warning(f"PARSED_RESUMES_JSON content is not a list in view_resumes: {resumes}")
#                     resumes = []
#         except (FileNotFoundError, json.JSONDecodeError):
#             logger.info("Failed to load resumes in view_resumes, using empty list")
#         except Exception as e:
#             logger.error(f"Unexpected error reading resumes: {e}")
#             resumes = []
#         return render_template('view_resumes.html', resumes=resumes, admin_email=session.get('username'))
#     except Exception as e:
#         logger.error(f"Error in view_resumes route: {e}")
#         flash("A system error occurred while loading resumes.", 'error')
#         return redirect('/admin')

# from urllib.parse import unquote

# @app.route('/download_resume/<path:filename>')
# def download_resume(filename):
#     try:
#         if session.get('role') not in ['admin', 'user']:
#             flash("Access denied!", 'error')
#             return redirect('/')
#         decoded_filename = unquote(filename)
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], decoded_filename)
#         if os.path.exists(file_path):
#             return send_file(file_path, as_attachment=True)
#         flash("File not found!", 'error')
#         return redirect('/resumes' if session.get('role') == 'admin' else '/user')
#     except Exception as e:
#         logger.error(f"Error in download_resume route: {e}")
#         return "Internal Server Error", 500

# @app.route('/manage_users', methods=['GET', 'POST'])
# def manage_users():
#     try:
#         if session.get('role') != 'admin':
#             flash("Access denied!", 'error')
#             return redirect('/')
#         users = load_users_from_csv()
#         if request.method == 'POST':
#             action = request.form.get('action')
#             username = request.form.get('username')
#             if action == 'delete' and username in users:
#                 try:
#                     with open(USERS_CSV, 'w', newline='', encoding='utf-8') as f:
#                         writer = csv.writer(f)
#                         writer.writerow(['username', 'password', 'role'])
#                         for u, data in users.items():
#                             if u != username:
#                                 writer.writerow([u, data['password'], data['role']])
#                     flash(f"User {username} deleted successfully!", 'success')
#                 except IOError as e:
#                     logger.error(f"Failed to update users CSV: {e}")
#                     flash("Failed to delete user!", 'error')
#             else:
#                 flash("Invalid action or user not found!", 'error')
#             return redirect('/manage_users')
#         return render_template('manage_users.html', users=users)
#     except Exception as e:
#         logger.error(f"Error in manage_users route: {e}")
#         return "Internal Server Error", 500

# @app.route('/filter_by_skills', methods=['GET', 'POST'])
# def filter_by_skills():
#     try:
#         if session.get('role') != 'admin':
#             logger.warning("Access denied for non-admin user attempting /filter_by_skills")
#             flash("Access denied!", 'error')
#             return redirect('/')
#         selected = request.form.get('category') or request.args.get('category', '')
#         search_query = request.form.get('search', '').lower().strip()
#         matched_resumes = []
#         resumes = []
#         try:
#             with open(PARSED_RESUMES_JSON, 'r', encoding='utf-8') as f:
#                 content = f.read().strip()
#                 resumes = json.loads(content) if content else []
#                 if not isinstance(resumes, list):
#                     logger.error(f"Invalid data type in {PARSED_RESUMES_JSON}: {type(resumes)}")
#                     resumes = []
#         except (FileNotFoundError, json.JSONDecodeError) as e:
#             logger.info(f"Failed to load resumes in filter_by_skills: {e}")
#             resumes = []
#         except Exception as e:
#             logger.error(f"Unexpected error loading {PARSED_RESUMES_JSON}: {str(e)}")
#             flash("Unexpected error loading resume data.", 'error')
#             resumes = []
#         logger.debug(f"Loaded {len(resumes)} resumes from file")
#         if selected:
#             required_skills = set(category_skills_dict.get(selected, []))
#             if not required_skills:
#                 logger.warning(f"No skills defined for category: {selected}")
#                 flash(f"No skills defined for category '{selected}'.", 'error')
#             else:
#                 logger.debug(f"Filtering with required_skills={required_skills}")
#                 for r in resumes:
#                     resume_skills = set(map(str.lower, r.get('skills', [])))
#                     email = str(r.get('email', '')).lower()
#                     filename = str(r.get('filename', '')).lower()
#                     if required_skills & resume_skills and (
#                         not search_query or search_query in email or search_query in filename
#                     ):
#                         matched_resumes.append(r)
#                         logger.debug(f"Matched resume: {r.get('email')}")
#         logger.debug(f"Found {len(matched_resumes)} matched resumes")
#         if not job_categories:
#             logger.warning("No valid job categories found")
#             flash("No job categories available.", 'error')
#         return render_template(
#             'filter_by_skills.html',
#             job_categories=job_categories,
#             matched_resumes=matched_resumes,
#             selected=selected,
#             search_query=search_query
#         )
#     except Exception as e:
#         logger.error(f"Error in filter_by_skills route: {str(e)}")
#         flash("Error in filter by skills. Check server logs for details.", 'error')
#         return redirect('/admin')

# @app.route('/export_filtered_resumes')
# def export_filtered_resumes():
#     try:
#         if session.get('role') != 'admin':
#             flash("Access denied!", 'error')
#             return redirect('/')
#         category = request.args.get('category')
#         if not category:
#             flash("No category selected for export!", 'error')
#             return redirect('/filter_by_skills')
#         resumes = []
#         try:
#             with open(PARSED_RESUMES_JSON, 'r', encoding='utf-8') as f:
#                 content = f.read().strip()
#                 resumes = json.loads(content) if content else []
#                 if not isinstance(resumes, list):
#                     logger.warning(f"PARSED_RESUMES_JSON content is not a list in export_filtered_resumes: {resumes}")
#                     resumes = []
#         except (FileNotFoundError, json.JSONDecodeError):
#             logger.info("Failed to load resumes in export_filtered_resumes, using empty list")
#         except Exception as e:
#             logger.error(f"Unexpected error reading resumes in export_filtered_resumes: {e}")
#             resumes = []
#         required_skills = set(category_skills_dict.get(category, []))
#         matched = [
#             r for r in resumes
#             if required_skills & set(map(str.lower, r.get('skills', [])))
#         ]
#         output = StringIO()
#         writer = csv.writer(output)
#         writer.writerow(['Email', 'Filename', 'Skills'])
#         for r in matched:
#             writer.writerow([r.get('email', ''), r.get('filename', ''), ', '.join(r.get('skills', []))])
#         output.seek(0)
#         return send_file(
#             BytesIO(output.getvalue().encode()),
#             mimetype='text/csv',
#             as_attachment=True,
#             download_name=f'{category}_resumes.csv'
#         )
#     except Exception as e:
#         logger.error(f"Error in export_filtered_resumes route: {e}")
#         return "Internal Server Error", 500

# if __name__ == '__main__':
#     logger.info("Starting Flask app...")
#     app.run(debug=True)