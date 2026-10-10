import pandas as pd
import numpy as np
import joblib

model = joblib.load('best_EEG_model.pkl')

# Test on sample files
files = {
    'sample_NORMAL_healthy.csv': 'Normal',
    'sample_MILD_risk.csv': 'Mild Risk',
    'sample_MODERATE_depression.csv': 'Moderate',
    'sample_SEVERE_depression.csv': 'Severe'
}

for fname, expected in files.items():
    df = pd.read_csv(f'sample_files/{fname}')
    feat_cols = [c for c in df.columns if c not in ['Expected_Result', 'HRV']]
    X = df[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(df[feat_cols].mean())
    
    pred = model.predict(X)
    proba = model.predict_proba(X)
    
    class_names = {0: 'Normal', 1: 'Mild Risk', 2: 'Moderate', 3: 'Severe'}
    pred_labels = [class_names[p] for p in pred]
    
    print(f'\n{fname}:')
    print(f'  Expected: {expected}')
    print(f'  Predictions: {pred_labels[:3]}')
    print(f'  Probabilities:')
    for i, row in enumerate(proba[:1]):
        for cls, name in class_names.items():
            print(f'    {name}: {row[cls]*100:.1f}%')
