"""
create_genuine_samples.py
Extract genuine samples from EEG_HRV_Updated.csv based on actual depression levels
and save them as 4 representative datasets.
"""

import pandas as pd
import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'EEG_HRV_Updated.csv')
SAMPLE_DIR = os.path.join(BASE_DIR, 'sample_files')

print("=" * 70)
print("  Creating Genuine Sample Datasets")
print("=" * 70)

# Load the main training data
if not os.path.exists(CSV_PATH):
    print(f"[ERROR] {CSV_PATH} not found!")
    exit(1)

df = pd.read_csv(CSV_PATH)
print(f"\n[OK] Loaded {CSV_PATH}: {df.shape[0]} rows")

# Check which column contains depression level
if 'Depression_Level' in df.columns:
    level_col = 'Depression_Level'
elif 'Depressed' in df.columns:
    level_col = 'Depressed'
else:
    print("[ERROR] No depression level column found!")
    exit(1)

print(f"[OK] Using column: {level_col}")

# Group by depression level
grouped = df.groupby(level_col)
print(f"\n[OK] Found {len(grouped)} depression level(s):")
for level, group in grouped:
    print(f"     {level}: {len(group)} samples")

# Create 4 representative datasets
level_mapping = {
    0: ('Normal', 'sample_NORMAL_healthy.csv'),
    1: ('Mild Risk', 'sample_MILD_risk.csv'),
    2: ('Moderate', 'sample_MODERATE_depression.csv'),
    3: ('Severe', 'sample_SEVERE_depression.csv'),
}

# If using string levels from Depression_Level column
if df[level_col].dtype == 'object':
    level_mapping = {}
    unique_levels = sorted(df[level_col].unique())
    filenames = [
        'sample_NORMAL_healthy.csv',
        'sample_MILD_risk.csv',
        'sample_MODERATE_depression.csv',
        'sample_SEVERE_depression.csv'
    ]
    for i, level in enumerate(unique_levels[:4]):
        level_mapping[level] = (level, filenames[i])

print(f"\n[PROCESSING] Creating sample files...\n")

os.makedirs(SAMPLE_DIR, exist_ok=True)

for level_key, (level_name, filename) in level_mapping.items():
    # Get samples for this level
    level_data = df[df[level_col] == level_key]
    
    if len(level_data) == 0:
        print(f"[SKIP] No data for {level_name}")
        continue
    
    # Take up to 5 samples (or all if less than 5)
    sample_size = min(5, len(level_data))
    samples = level_data.sample(n=sample_size, random_state=42)
    
    # Save to file
    output_path = os.path.join(SAMPLE_DIR, filename)
    samples.to_csv(output_path, index=False)
    print(f"[SAVED] {filename}: {sample_size} samples from {level_name}")

print(f"\n{'='*70}")
print(f"  ✓ Genuine sample datasets created successfully!")
print(f"  These files contain real EEG patterns from {level_name} depression cases")
print(f"  The model will predict based on actual feature patterns, not filenames")
print(f"{'='*70}\n")
