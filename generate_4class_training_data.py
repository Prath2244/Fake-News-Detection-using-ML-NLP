"""
generate_4class_training_data.py
Generate 4-class training data by expanding sample files and creating balanced dataset
"""
import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, 'sample_files')

print("=" * 60)
print("  Generating 4-Class Training Dataset")
print("=" * 60)

# Load sample files
sample_files = {
    'sample_NORMAL_healthy.csv': 0,
    'sample_MILD_risk.csv': 1,
    'sample_MODERATE_depression.csv': 2,
    'sample_SEVERE_depression.csv': 3,
}

dfs = []
for fname, class_label in sample_files.items():
    fpath = os.path.join(SAMPLE_DIR, fname)
    if os.path.exists(fpath):
        df = pd.read_csv(fpath)
        # Ensure Expected_Result column exists
        if 'Expected_Result' not in df.columns:
            df['Expected_Result'] = fname.split('_')[1].replace('.csv', '')
        print(f"[OK] Loaded {fname}: {len(df)} rows")
        dfs.append(df)
    else:
        print(f"[WARNING] {fname} not found")

if not dfs:
    print("[ERROR] No sample files found!")
    exit(1)

# Combine all data
df_all = pd.concat(dfs, ignore_index=True)
print(f"\n[OK] Combined shape: {df_all.shape}")
print(f"[OK] Expected_Result value counts:\n{df_all['Expected_Result'].value_counts()}")

# Ensure numeric columns
feature_cols = [c for c in df_all.columns if c not in ['Expected_Result', 'Depressed', 'Depression_Level', 'label', 'Label']]
X = df_all[feature_cols].apply(pd.to_numeric, errors='coerce')
X = X.fillna(X.mean())

# Map Expected_Result to 4-class labels
def map_category(val):
    val_str = str(val).lower()
    if 'normal' in val_str or 'healthy' in val_str: return 0
    if 'mild' in val_str: return 1
    if 'moderate' in val_str: return 2
    if 'severe' in val_str: return 3
    return 1

y = df_all['Expected_Result'].apply(map_category)

# Create augmented training data by replicating minority classes
class_counts = y.value_counts().sort_index()
print(f"\n[INFO] Class distribution before augmentation:\n{class_counts}")

# Target: 500 samples per class = 2000 total (balanced)
target_per_class = 500
all_dfs = []

for class_label in range(4):
    class_mask = y == class_label
    class_data = X[class_mask].copy()
    
    if len(class_data) == 0:
        print(f"[WARNING] No data for class {class_label}")
        continue
    
    # Replicate to reach target
    current_count = len(class_data)
    if current_count < target_per_class:
        # How many copies needed
        copies_needed = target_per_class // current_count
        remainder = target_per_class % current_count
        
        replicated = []
        for _ in range(copies_needed):
            replicated.append(class_data.copy())
        if remainder > 0:
            replicated.append(class_data.iloc[:remainder].copy())
        
        class_data = pd.concat(replicated, ignore_index=True)
    else:
        # If already enough, just take the first target_per_class rows
        class_data = class_data.iloc[:target_per_class].copy()
    
    # Add HRV if missing (random between 40-120)
    if 'HRV' not in class_data.columns:
        class_data['HRV'] = np.random.uniform(40, 120, len(class_data))
    
    # Add Expected_Result
    class_names = {0: 'Normal', 1: 'Mild Risk', 2: 'Moderate', 3: 'Severe'}
    class_data['Expected_Result'] = class_names[class_label]
    
    all_dfs.append(class_data)
    print(f"[OK] Class {class_label} ({class_names[class_label]}): {len(class_data)} samples")

# Combine all augmented data
df_augmented = pd.concat(all_dfs, ignore_index=True)
df_augmented = df_augmented.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n[OK] Final augmented dataset shape: {df_augmented.shape}")
print(f"[OK] Final class distribution:\n{df_augmented['Expected_Result'].value_counts()}")

# Save
output_path = os.path.join(BASE_DIR, 'EEG_4class_training.csv')
df_augmented.to_csv(output_path, index=False)
print(f"\n[SAVED] {output_path}")
print(f"  This file can be used for training with retrain_model.py")
