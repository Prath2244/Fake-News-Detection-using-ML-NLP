"""
NeuroCare - EEG Depression Classifier
Flask Web Application
Developer: Ajay
"""

import os
import csv
import sys
import logging
import joblib
import numpy as np
from datetime import datetime
from io import StringIO, BytesIO

from flask import (Flask, render_template, redirect, url_for, request,
                   flash, jsonify, send_file, make_response)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (LoginManager, UserMixin, login_user, logout_user,
                         login_required, current_user)
from werkzeug.security import generate_password_hash, check_password_hash

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

# ── BASE DIR ──
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# ── CREATE FOLDERS ──
for folder in [
    os.path.join(BASE_DIR, 'static', 'logs'),
    os.path.join(BASE_DIR, 'static', 'reports'),
    os.path.join(BASE_DIR, 'database'),
    os.path.join(BASE_DIR, 'static', 'css'),
    os.path.join(BASE_DIR, 'static', 'js'),
]:
    os.makedirs(folder, exist_ok=True)

# ── LOGGING ──
LOG_FILE = os.path.join(BASE_DIR, 'static', 'logs', 'app.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

DEPRESSION_LOG    = os.path.join(BASE_DIR, 'static', 'logs', 'depression_log.txt')
NO_DEPRESSION_LOG = os.path.join(BASE_DIR, 'static', 'logs', 'not_depression_log.txt')

# ── FLASK APP ──
app = Flask(__name__)
app.config['SECRET_KEY'] = 'neurocare-eeg-secret-ajay-2025'
app.config['SQLALCHEMY_DATABASE_URI'] = (
    'sqlite:///' + os.path.join(BASE_DIR, 'database', 'users.db')
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db            = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view             = 'login'
login_manager.login_message          = 'Please login to access this page.'
login_manager.login_message_category = 'info'

# ── EEG FEATURES ──
FEATURE_COLS = [
    'Fp1_alpha','Fp1_beta','Fp1_mean','Fp1_std',
    'Fp2_alpha','Fp2_beta','Fp2_mean','Fp2_std',
    'F3_alpha', 'F3_beta', 'F3_mean', 'F3_std',
    'F4_alpha', 'F4_beta', 'F4_mean', 'F4_std',
    'C3_alpha', 'C3_beta', 'C3_mean', 'C3_std',
    'C4_alpha', 'C4_beta', 'C4_mean', 'C4_std',
    'P3_alpha', 'P3_beta', 'P3_mean', 'P3_std',
    'P4_alpha', 'P4_beta', 'P4_mean', 'P4_std',
    'O1_alpha', 'O1_beta', 'O1_mean', 'O1_std',
    'O2_alpha', 'O2_beta', 'O2_mean', 'O2_std',
    'F7_alpha', 'F7_beta', 'F7_mean', 'F7_std',
    'F8_alpha', 'F8_beta', 'F8_mean', 'F8_std',
    'T3_alpha', 'T3_beta', 'T3_mean', 'T3_std',
    'T4_alpha', 'T4_beta', 'T4_mean', 'T4_std',
]

ELECTRODE_GROUPS = {
    'Frontal Polar (Fp1, Fp2)': [
        'Fp1_alpha','Fp1_beta','Fp1_mean','Fp1_std',
        'Fp2_alpha','Fp2_beta','Fp2_mean','Fp2_std',
    ],
    'Frontal (F3, F4, F7, F8)': [
        'F3_alpha','F3_beta','F3_mean','F3_std',
        'F4_alpha','F4_beta','F4_mean','F4_std',
        'F7_alpha','F7_beta','F7_mean','F7_std',
        'F8_alpha','F8_beta','F8_mean','F8_std',
    ],
    'Central (C3, C4)': [
        'C3_alpha','C3_beta','C3_mean','C3_std',
        'C4_alpha','C4_beta','C4_mean','C4_std',
    ],
    'Parietal (P3, P4)': [
        'P3_alpha','P3_beta','P3_mean','P3_std',
        'P4_alpha','P4_beta','P4_mean','P4_std',
    ],
    'Occipital (O1, O2)': [
        'O1_alpha','O1_beta','O1_mean','O1_std',
        'O2_alpha','O2_beta','O2_mean','O2_std',
    ],
    'Temporal (T3, T4)': [
        'T3_alpha','T3_beta','T3_mean','T3_std',
        'T4_alpha','T4_beta','T4_mean','T4_std',
    ],
}

# ── LOAD MODEL ──
# ── MODEL LOADING ──
def _train_and_save_model():
    """Train a fresh model using all CSV files in sample_files folder."""
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    import warnings
    warnings.filterwarnings('ignore')

    sample_dir = os.path.join(BASE_DIR, 'sample_files')
    
    # Look for main training data file
    csv_paths = []
    for fname in ['EEG_HRV_Updated.csv', 'EEG_Real.csv', 'eeg_data.csv']:
        fpath = os.path.join(BASE_DIR, fname)
        if os.path.exists(fpath):
            csv_paths.append(fpath)
            break  # Use only the first found file
    
    if not csv_paths:
        print(f"[ERROR] No CSV files found in {sample_dir}", flush=True)
        print(f"[ERROR] Please run: python retrain_model.py", flush=True)
        return None

    csv_paths.sort()  # Sort for consistency
    
    # Load and concatenate all CSV files
    dfs = []
    for csv_path in csv_paths:
        print(f"[AUTO-TRAIN] Loading: {os.path.basename(csv_path)}", flush=True)
        df_temp = pd.read_csv(csv_path)
        dfs.append(df_temp)
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"[AUTO-TRAIN] Combined {len(dfs)} CSV file(s), total rows: {len(df)}", flush=True)
    
    skip = {'Depressed','Depression_Level','HRV','label','Label','target','Expected_Result'}
    feat = [c for c in df.columns if c not in skip]
    # Convert to numeric, coercing errors to NaN
    X = df[feat].apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.mean())
    
    # Handle target column - support multi-class classification
    def map_category(val):
        val_str = str(val).lower()
        if 'normal' in val_str or 'healthy' in val_str: return 0
        if 'mild' in val_str: return 1
        if 'moderate' in val_str: return 2
        if 'severe' in val_str: return 3
        return 1  # Default to Mild
    
    if 'Expected_Result' in df.columns:
        y = df['Expected_Result'].apply(map_category)
    elif 'Depressed' in df.columns:
        y = (df['Depressed'] > 0).astype(int)
    elif 'label' in df.columns:
        y = df['label'].astype(int)
    elif 'Label' in df.columns:
        y = df['Label'].astype(int)
    else:
        y = df['label'].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('clf', GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                                         learning_rate=0.1, random_state=42))])
    pipe.fit(X_tr, y_tr)
    acc = pipe.score(X_te, y_te)
    print(f"[AUTO-TRAIN] Accuracy: {acc*100:.2f}%", flush=True)

    for fname in ['best_EEG_model.pkl', 'model.pkl']:
        out = os.path.join(BASE_DIR, fname)
        joblib.dump(pipe, out)
        print(f"[AUTO-TRAIN] Saved: {out}", flush=True)
    return pipe


_MODEL_CANDIDATES = ['best_EEG_model.pkl', 'model.pkl', 'eeg_model.pkl']

MODEL_PATH = None
print(f"[INFO] Looking for model in: {BASE_DIR}", flush=True)
for _cand in _MODEL_CANDIDATES:
    _p = os.path.join(BASE_DIR, _cand)
    if os.path.exists(_p):
        MODEL_PATH = _p
        break

model = None
if MODEL_PATH:
    try:
        model = joblib.load(MODEL_PATH)
        print(f"[OK] Model loaded: {os.path.basename(MODEL_PATH)}", flush=True)
        logger.info('Model loaded OK: %s', MODEL_PATH)
    except Exception as e:
        print(f"[WARNING] Model file corrupt/incompatible: {e}", flush=True)
        print(f"[WARNING] Version mismatch detected. Auto-retraining with your sklearn version...", flush=True)
        logger.warning('Model incompatible (%s), auto-retraining...', e)
        # Delete bad pkl files
        for _cand in _MODEL_CANDIDATES:
            _bad = os.path.join(BASE_DIR, _cand)
            if os.path.exists(_bad):
                try:
                    os.remove(_bad)
                    print(f"[CLEANUP] Removed incompatible: {_cand}", flush=True)
                except Exception:
                    pass
        model = _train_and_save_model()
        if model:
            print(f"[OK] Auto-retrain complete. Model is ready!", flush=True)
        else:
            print(f"[ERROR] Auto-retrain failed. Run: python retrain_model.py", flush=True)
else:
    print(f"[WARNING] No model file found. Auto-training now...", flush=True)
    model = _train_and_save_model()
    if model:
        print(f"[OK] Auto-train complete!", flush=True)

# ── DATABASE MODELS ──
class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id            = db.Column(db.Integer, primary_key=True)
    name          = db.Column(db.String(100), nullable=False)
    email         = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)
    predictions   = db.relationship('Prediction', backref='user', lazy=True)

    def set_password(self, pw):
        self.password_hash = generate_password_hash(pw)

    def check_password(self, pw):
        return check_password_hash(self.password_hash, pw)


class Prediction(db.Model):
    __tablename__ = 'predictions'
    id            = db.Column(db.Integer, primary_key=True)
    user_id       = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    timestamp     = db.Column(db.DateTime, default=datetime.utcnow)
    prediction    = db.Column(db.String(50), nullable=False)
    probability   = db.Column(db.Float,     nullable=False)
    level         = db.Column(db.String(50), nullable=False)
    features_json = db.Column(db.Text, nullable=True)


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# ── HELPERS ──
def get_level(prob_pct):
    if prob_pct < 40: return 'Normal'
    if prob_pct < 65: return 'Mild Risk'
    if prob_pct < 85: return 'Moderate'
    return 'Severe'


def get_suggestions(level):
    return {
        'Normal': [
            'Continue maintaining a healthy lifestyle.',
            'Regular physical activity and a balanced diet.',
            'Practice mindfulness or light meditation daily.',
            'Maintain social connections with friends and family.',
        ],
        'Mild Risk': [
            'Improve sleep schedule - aim for 7-8 hours nightly.',
            'Engage in light exercise: walking, yoga, or cycling.',
            'Practice daily meditation or deep-breathing exercises.',
            'Limit caffeine and screen time before bed.',
            'Journal your thoughts and feelings regularly.',
        ],
        'Moderate': [
            'Counseling or talk therapy is strongly recommended.',
            'Identify and reduce known stress triggers in daily life.',
            'Increase meaningful social interactions.',
            'Consider joining a support group.',
            'Consult a general physician for further evaluation.',
            'Reduce alcohol or substance intake if applicable.',
        ],
        'Severe': [
            'Consult a licensed psychiatrist immediately.',
            'Professional therapy (CBT or DBT) is strongly advised.',
            'Do not remain alone - contact a trusted person.',
            'National Mental Health Helpline: iCall 9152987821',
            'NIMHANS Helpline: 080-46110007',
            'Avoid making any major life decisions during this period.',
        ],
    }.get(level, [])


def write_log(is_depression, username, probability, level, hrv, timestamp):
    path = DEPRESSION_LOG if is_depression else NO_DEPRESSION_LOG
    try:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(f"{timestamp} | {username} | {probability:.2f}% | {level} | HRV:{hrv:.1f}ms\n")
        print(f"[LOG] Written to {path}", flush=True)
    except Exception as e:
        logger.error('Log write error: %s', e)
        print(f"[LOG ERROR] {e}", flush=True)


def generate_pdf(user, rec):
    buf  = BytesIO()
    doc  = SimpleDocTemplate(buf, pagesize=A4,
                              rightMargin=0.75*inch, leftMargin=0.75*inch,
                              topMargin=0.75*inch, bottomMargin=0.75*inch)
    st    = getSampleStyleSheet()
    story = []
    title_s = ParagraphStyle('T2', parent=st['Title'], fontSize=20,
                              textColor=colors.HexColor('#1a6fc4'),
                              spaceAfter=6, alignment=TA_CENTER)
    sub_s   = ParagraphStyle('Sub', parent=st['Normal'], fontSize=11,
                              textColor=colors.HexColor('#555555'),
                              alignment=TA_CENTER, spaceAfter=4)
    head_s  = ParagraphStyle('Head', parent=st['Heading2'], fontSize=13,
                              textColor=colors.HexColor('#1a6fc4'),
                              spaceBefore=14, spaceAfter=4)
    body_s  = ParagraphStyle('Body', parent=st['Normal'], fontSize=11, leading=16)

    story.append(Paragraph("EEG-Based Depression Classification", title_s))
    story.append(Paragraph("Hybrid Meta-Learning Model - Clinical Report", sub_s))
    story.append(HRFlowable(width="100%", thickness=2,
                             color=colors.HexColor('#1a6fc4'), spaceAfter=14))
    info = [
        ['Patient:', user.name,  'Date:', rec.timestamp.strftime('%d %B %Y')],
        ['Email:',   user.email, 'Time:', rec.timestamp.strftime('%H:%M:%S')],
        ['Report ID:', f'RPT-{rec.id:04d}', 'Model:', 'Hybrid Meta-Learner v1.0'],
    ]
    it = Table(info, colWidths=[1.2*inch, 2.6*inch, 1.0*inch, 2.3*inch])
    it.setStyle(TableStyle([
        ('FONTSIZE',       (0,0), (-1,-1), 10),
        ('FONTNAME',       (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME',       (2,0), (2,-1), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.HexColor('#f0f7ff'), colors.white]),
        ('GRID',           (0,0), (-1,-1), 0.3, colors.HexColor('#ccddee')),
        ('PADDING',        (0,0), (-1,-1), 6),
    ]))
    story.append(it)
    story.append(Spacer(1, 14))
    story.append(Paragraph("Prediction Result", head_s))
    lc   = {'Normal':'#27ae60','Mild Risk':'#f39c12','Moderate':'#e67e22','Severe':'#e74c3c'}
    lhex = lc.get(rec.level, '#333333')
    rd   = [
        ['Diagnosis',         rec.prediction],
        ['Probability Score', f"{rec.probability:.2f}%"],
        ['Risk Level',        rec.level],
    ]
    rt = Table(rd, colWidths=[2.5*inch, 4.6*inch])
    rt.setStyle(TableStyle([
        ('FONTSIZE',   (0,0), (-1,-1), 11),
        ('FONTNAME',   (0,0), (0,-1), 'Helvetica-Bold'),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#dceeff')),
        ('BACKGROUND', (0,1), (-1,1), colors.HexColor('#f0f7ff')),
        ('BACKGROUND', (0,2), (-1,2), colors.HexColor('#dceeff')),
        ('TEXTCOLOR',  (1,2), (1,2),  colors.HexColor(lhex)),
        ('FONTNAME',   (1,2), (1,2),  'Helvetica-Bold'),
        ('GRID',       (0,0), (-1,-1), 0.4, colors.HexColor('#aaccee')),
        ('PADDING',    (0,0), (-1,-1), 8),
    ]))
    story.append(rt)
    story.append(Spacer(1, 14))
    story.append(Paragraph("Clinical Suggestions", head_s))
    for i, s in enumerate(get_suggestions(rec.level), 1):
        story.append(Paragraph(f"{i}. {s}", body_s))
    story.append(Spacer(1, 18))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor('#cccccc'), spaceAfter=8))
    disc_s = ParagraphStyle('D', parent=st['Normal'], fontSize=9,
                             textColor=colors.HexColor('#888888'), leading=13)
    story.append(Paragraph(
        "DISCLAIMER: This report is for research purposes only. "
        "It does not constitute a medical diagnosis. "
        "Consult a qualified mental health professional for clinical evaluation.",
        disc_s))
    story.append(Spacer(1, 18))
    sig = [['', 'Authorised By', ''],
           ['', '________________________', ''],
           ['', 'Dr. / Clinical Reviewer', ''],
           ['', 'Date: ' + rec.timestamp.strftime('%d/%m/%Y'), '']]
    st2 = Table(sig, colWidths=[2.3*inch, 2.5*inch, 2.3*inch])
    st2.setStyle(TableStyle([
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('ALIGN',    (1,0), (1,-1), 'CENTER'),
        ('FONTNAME', (1,0), (1,0), 'Helvetica-Bold'),
    ]))
    story.append(st2)
    doc.build(story)
    buf.seek(0)
    return buf

# ── ROUTES ──
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        name    = request.form.get('name', '').strip()
        email   = request.form.get('email', '').strip().lower()
        pw      = request.form.get('password', '')
        confirm = request.form.get('confirm_password', '')
        errors  = []
        if not name:        errors.append('Name is required.')
        if not email:       errors.append('Email is required.')
        if len(pw) < 6:     errors.append('Password must be at least 6 characters.')
        if pw != confirm:   errors.append('Passwords do not match.')
        if User.query.filter_by(email=email).first():
            errors.append('Email already registered.')
        if errors:
            for e in errors:
                flash(e, 'danger')
            return render_template('signup.html', name=name, email=email)
        u = User(name=name, email=email)
        u.set_password(pw)
        db.session.add(u)
        db.session.commit()
        flash('Account created! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        pw       = request.form.get('password', '')
        remember = request.form.get('remember') == 'on'
        u = User.query.filter_by(email=email).first()
        if u and u.check_password(pw):
            login_user(u, remember=remember)
            logger.info('Login OK: %s', email)
            return redirect(request.args.get('next') or url_for('dashboard'))
        flash('Invalid email or password.', 'danger')
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


@app.route('/dashboard')
@login_required
def dashboard():
    preds = (Prediction.query
             .filter_by(user_id=current_user.id)
             .order_by(Prediction.timestamp.desc()).all())
    total     = len(preds)
    dep_count = sum(1 for p in preds if p.prediction == 'Depression')
    no_dep    = total - dep_count
    latest_5  = preds[:5]
    lc        = {'Normal': 0, 'Mild Risk': 0, 'Moderate': 0, 'Severe': 0}
    for p in preds:
        if p.level in lc: lc[p.level] += 1
    prob_values = [round(p.probability, 2) for p in preds[-20:]]
    timestamps  = [p.timestamp.strftime('%d %b') for p in preds[-20:]]
    return render_template('dashboard.html',
                           total=total, dep_count=dep_count, no_dep=no_dep,
                           latest_5=latest_5, level_counts=lc,
                           prob_values=prob_values, timestamps=timestamps)


@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    result = None
    if request.method == 'POST':
        try:
            # Read HRV
            hrv_raw = (request.form.get('hrv_hidden') or request.form.get('hrv') or '').strip()
            if not hrv_raw:
                raise ValueError('HRV (Heart Rate Variability) value is required.')
            hrv_val = float(hrv_raw)

            # Read 56 EEG features
            vals = []
            for col in FEATURE_COLS:
                v = request.form.get(col, '').strip()
                if not v:
                    raise ValueError(f'Missing value for EEG feature: {col}')
                vals.append(float(v))

            X = np.array(vals).reshape(1, -1)
            if model is None:
                flash('model.pkl not found. Place it in the app folder.', 'danger')
                return render_template('predict.html', groups=ELECTRODE_GROUPS)

            pred_label = int(model.predict(X)[0])
            prob_arr   = model.predict_proba(X)[0]
            
            # Map class predictions to labels and levels
            class_labels = {0: 'Not Depression', 1: 'Mild Risk', 2: 'Moderate', 3: 'Severe Depression'}
            class_levels = {0: 'Normal', 1: 'Mild Risk', 2: 'Moderate', 3: 'Severe'}
            
            pred_str = class_labels.get(pred_label, 'Not Depression')
            level = class_levels.get(pred_label, 'Normal')
            
            # Map class to depression probability range
            # Class 0 (Normal): 0-30%, Class 1 (Mild): 40-65%, Class 2 (Moderate): 65-85%, Class 3 (Severe): 85-100%
            class_prob_ranges = {
                0: (0, 30),      # Normal
                1: (40, 65),     # Mild Risk
                2: (65, 85),     # Moderate
                3: (85, 100),    # Severe
            }
            
            min_prob, max_prob = class_prob_ranges.get(pred_label, (0, 50))
            # Use model confidence to interpolate within the range
            confidence = float(prob_arr[pred_label]) if pred_label < len(prob_arr) else 0.5
            prob_dep = min_prob + (max_prob - min_prob) * confidence
            
            suggs      = get_suggestions(level)
            ts         = datetime.utcnow()

            rec = Prediction(user_id=current_user.id, prediction=pred_str,
                             probability=prob_dep, level=level, timestamp=ts)
            db.session.add(rec)
            db.session.commit()

            write_log(pred_label > 0, current_user.name, prob_dep,
                      level, hrv_val, ts.strftime('%Y-%m-%d %H:%M:%S'))
            logger.info('Prediction: %s %.2f%% %s HRV=%.1f user=%s',
                        pred_str, prob_dep, level, hrv_val, current_user.email)

            result = {
                'prediction':    pred_str,
                'probability':   round(prob_dep, 2),
                'level':         level,
                'suggestions':   suggs,
                'record_id':     rec.id,
                'is_depression': pred_label > 0,
                'hrv':           hrv_val,
            }
        except ValueError as e:
            flash(f'Input error: {e}', 'danger')
        except Exception as e:
            flash(f'Prediction failed: {e}', 'danger')
            logger.error('Predict error: %s', e)
    return render_template('predict.html', groups=ELECTRODE_GROUPS, result=result)


@app.route('/upload_csv_predict', methods=['POST'])
@login_required
def upload_csv_predict():
    """Accept a CSV file upload and return the feature values as JSON for auto-fill."""
    try:
        f = request.files.get('csv_file')
        if not f or f.filename == '':
            return jsonify({'error': 'No file uploaded'}), 400
        if not f.filename.lower().endswith('.csv'):
            return jsonify({'error': 'Please upload a CSV file'}), 400

        import io
        import csv as csv_mod
        stream = io.StringIO(f.stream.read().decode('utf-8', errors='ignore'))
        reader = csv_mod.DictReader(stream)
        rows   = list(reader)
        if not rows:
            return jsonify({'error': 'CSV file is empty'}), 400

        # Use first data row
        row = rows[0]
        values = {}
        missing = []
        for col in FEATURE_COLS:
            val = row.get(col, '').strip()
            if val:
                try:
                    values[col] = float(val)
                except ValueError:
                    missing.append(col)
            else:
                missing.append(col)

        # Try to get HRV
        hrv = row.get('HRV', row.get('hrv', row.get('HRV_mean', ''))).strip()
        hrv_val = None
        if hrv:
            try:
                hrv_val = float(hrv)
            except ValueError:
                pass

        return jsonify({
            'success':  True,
            'values':   values,
            'hrv':      hrv_val,
            'missing':  missing,
            'row_count': len(rows),
            'filename': f.filename,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/history')
@login_required
def history():
    page    = request.args.get('page', 1, type=int)
    sort_by = request.args.get('sort', 'timestamp')
    order   = request.args.get('order', 'desc')
    col_map = {
        'timestamp': Prediction.timestamp, 'prediction': Prediction.prediction,
        'probability': Prediction.probability, 'level': Prediction.level,
    }
    col = col_map.get(sort_by, Prediction.timestamp)
    col = col.asc() if order == 'asc' else col.desc()
    pagination = (Prediction.query.filter_by(user_id=current_user.id)
                  .order_by(col).paginate(page=page, per_page=10, error_out=False))
    return render_template('history.html', pagination=pagination,
                           sort_by=sort_by, order=order)


@app.route('/download_report/<int:pred_id>')
@login_required
def download_report(pred_id):
    rec = Prediction.query.filter_by(id=pred_id, user_id=current_user.id).first_or_404()
    buf = generate_pdf(current_user, rec)
    fname = f"EEG_Report_{current_user.name.replace(' ','_')}_{rec.id}.pdf"
    return send_file(buf, as_attachment=True,
                     download_name=fname, mimetype='application/pdf')


@app.route('/export_csv')
@login_required
def export_csv():
    preds = (Prediction.query.filter_by(user_id=current_user.id)
             .order_by(Prediction.timestamp.desc()).all())
    si = StringIO()
    cw = csv.writer(si)
    cw.writerow(['ID', 'Timestamp', 'Prediction', 'Probability (%)', 'Level'])
    for p in preds:
        cw.writerow([p.id, p.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                     p.prediction, f'{p.probability:.2f}', p.level])
    out = make_response(si.getvalue())
    out.headers['Content-Disposition'] = 'attachment; filename=prediction_history.csv'
    out.headers['Content-type'] = 'text/csv'
    return out


@app.route('/logs')
@login_required
def logs():
    return render_template('logs.html')


@app.route('/api/logs/<log_type>')
@login_required
def api_logs(log_type):
    if log_type == 'depression':       path = DEPRESSION_LOG
    elif log_type == 'not_depression': path = NO_DEPRESSION_LOG
    else: return jsonify({'error': 'Invalid log type'}), 400
    entries = []
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [x.strip() for x in line.split('|')]
                if len(parts) >= 4:
                    # Parse level and HRV - level field may contain "Normal | HRV:65.0ms"
                    level_raw = parts[3]
                    hrv_val   = '—'
                    # New format: parts[4] has HRV if 5 parts
                    if len(parts) >= 5:
                        hrv_raw = parts[4]
                        hrv_val = hrv_raw.replace('HRV:', '').strip()
                    else:
                        # Try to extract HRV embedded in level string
                        if 'HRV:' in level_raw:
                            bits    = level_raw.split('HRV:')
                            level_raw = bits[0].strip()
                            hrv_val   = bits[1].replace('ms', '').strip() + ' ms'
                    entries.append({
                        'timestamp':   parts[0],
                        'username':    parts[1],
                        'probability': parts[2],
                        'level':       level_raw,
                        'hrv':         hrv_val,
                    })
    entries.reverse()
    return jsonify(entries)


@app.route('/api/stats')
@login_required
def api_stats():
    preds = Prediction.query.filter_by(user_id=current_user.id).all()
    lc    = {'Normal': 0, 'Mild Risk': 0, 'Moderate': 0, 'Severe': 0}
    for p in preds:
        if p.level in lc: lc[p.level] += 1
    return jsonify({'total': len(preds),
                    'depression':     sum(1 for p in preds if p.prediction == 'Depression'),
                    'not_depression': sum(1 for p in preds if p.prediction == 'Not Depression'),
                    'levels': lc})


# ── STARTUP ──
def create_tables():
    with app.app_context():
        db.create_all()
        logger.info('Database ready.')


if __name__ == '__main__':
    create_tables()
    logger.info('NeuroCare starting on http://127.0.0.1:5000')
    app.run(
        debug=True,
        host='127.0.0.1',
        port=5000,
        use_reloader=False,   # FIXES Windows freeze
    )
