"""
retrain_model.py
Run this ONCE inside your .venv to rebuild the model with YOUR sklearn version.
Usage:  python retrain_model.py

Place this file in the same folder as app.py (F:/Ajay R/abn_51/)
and make sure EEG_Real.csv or EEG_HRV_Updated.csv is also in that folder.
"""

import os, sys, warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  NeuroCare — Model Retraining Script")
print("=" * 60)

# ── Check dependencies ──
try:
    import numpy as np
    import pandas as pd
    import joblib
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    import sklearn
    print(f"[OK] sklearn  version : {sklearn.__version__}")
    print(f"[OK] numpy    version : {np.__version__}")
    print(f"[OK] pandas   version : {pd.__version__}")
except ImportError as e:
    print(f"[ERROR] Missing package: {e}")
    print("Run:  pip install scikit-learn numpy pandas joblib")
    sys.exit(1)

# ── Find CSV files ──
BASE_DIR   = os.path.abspath(os.path.dirname(__file__))

# Look for main training data file
csv_paths = []
for fname in ['EEG_4class_training.csv', 'EEG_HRV_Updated.csv', 'EEG_Real.csv', 'eeg_data.csv']:
    fpath = os.path.join(BASE_DIR, fname)
    if os.path.exists(fpath):
        csv_paths.append(fpath)
        print(f"[OK] Found: {fname}")
        break  # Use only the first found file

if not csv_paths:
    print(f"\n[ERROR] No CSV files found in {SAMPLE_DIR}")
    sys.exit(1)

csv_paths.sort()  # Sort for consistency
print(f"\n[OK] Found {len(csv_paths)} CSV file(s) in sample_files:")
for p in csv_paths:
    print(f"    - {os.path.basename(p)}")

# ── Load and concatenate data ──
dfs = []
for csv_path in csv_paths:
    df_temp = pd.read_csv(csv_path)
    dfs.append(df_temp)
    print(f"[OK] Loaded {os.path.basename(csv_path)}: {df_temp.shape}")

df = pd.concat(dfs, ignore_index=True)
print(f"\n[OK] Combined data shape: {df.shape}")
print(f"[OK] Columns   : {list(df.columns)}")

FEATURE_COLS = [c for c in df.columns if c not in [
    'Depressed', 'Depression_Level', 'HRV', 'label', 'Label', 'target', 'Expected_Result'
]]
print(f"\n[OK] Features used : {len(FEATURE_COLS)} columns")
print(f"     {FEATURE_COLS[:8]} ...")

# Convert to numeric, coercing errors to NaN
X = df[FEATURE_COLS].apply(pd.to_numeric, errors='coerce')
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
elif 'Depression_Level' in df.columns:
    y = df['Depression_Level'].apply(map_category)
elif 'label' in df.columns:
    y = df['label'].astype(int)
elif 'Label' in df.columns:
    y = df['Label'].astype(int)
elif 'Depressed' in df.columns:
    y = (df['Depressed'] > 0).astype(int)
else:
    print("[ERROR] Could not find target column (Expected_Result / Depression_Level / label / Label / Depressed)")
    sys.exit(1)

print(f"[OK] Target distribution: {dict(y.value_counts())}")

# ── Train model ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n[TRAINING] Training Gradient Boosting Classifier...")
print(f"           Train samples: {len(X_train)}  |  Test samples: {len(X_test)}")

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf',    GradientBoostingClassifier(
        n_estimators  = 150,
        max_depth     = 4,
        learning_rate = 0.1,
        random_state  = 42,
    ))
])

pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)

print(f"\n[RESULT] Test Accuracy : {accuracy * 100:.2f}%")

# ── Save model ──
out1 = os.path.join(BASE_DIR, 'best_EEG_model.pkl')
out2 = os.path.join(BASE_DIR, 'model.pkl')

joblib.dump(pipeline, out1)
joblib.dump(pipeline, out2)

print(f"\n[SAVED] {out1}")
print(f"[SAVED] {out2}")
print(f"\n{'='*60}")
print(f"  Model trained successfully with accuracy {accuracy*100:.2f}%")
print(f"  Now run:  python app.py")
print(f"{'='*60}\n")
