"""
Create realistic synthetic training data with distinct EEG patterns for each depression severity
"""
import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, 'sample_files')

print("=" * 60)
print("  Creating Synthetic Training Data (Distinct Patterns)")
print("=" * 60)

# Define base feature patterns for each class
# Depression severity correlates with: increased theta/delta (low freq), decreased alpha/beta (high freq)

np.random.seed(42)

def create_class_data(class_name, n_samples, alpha_mean, beta_mean, std_mean):
    """
    Create synthetic EEG data with depression-severity patterns
    
    Normal: High alpha/beta, low theta - normal brain activity
    Mild: Slightly reduced alpha/beta, slightly elevated theta
    Moderate: Further reduced alpha/beta, more elevated theta
    Severe: Very low alpha/beta, very high theta - severely depressed
    """
    
    data = {}
    
    # EEG electrodes
    electrodes = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4']
    
    for electrode in electrodes:
        # Alpha waves (8-13 Hz) - reduced in depression
        data[f'{electrode}_alpha'] = np.random.normal(alpha_mean, 5, n_samples)
        
        # Beta waves (13-30 Hz) - reduced in depression  
        data[f'{electrode}_beta'] = np.random.normal(beta_mean, 4, n_samples)
        
        # Mean amplitude (DC component)
        data[f'{electrode}_mean'] = np.random.normal(0, std_mean, n_samples)
        
        # Std of signal (variability)
        data[f'{electrode}_std'] = np.random.uniform(2, 6, n_samples)
    
    # HRV (Heart Rate Variability) - also affected by depression
    hrv_mean = 80 if class_name == 'Normal' else (70 if class_name == 'Mild Risk' else (60 if class_name == 'Moderate' else 45))
    data['HRV'] = np.random.normal(hrv_mean, 8, n_samples)
    
    # Expected result
    data['Expected_Result'] = class_name
    
    df = pd.DataFrame(data)
    return df

# Create distinct training patterns for each class
# Increasing theta/decreasing alpha with depression severity
classes_config = [
    ('Normal', 35, 25, 3),        # High alpha/beta, low variability
    ('Mild Risk', 28, 20, 3.5),   # Reduced alpha/beta
    ('Moderate', 20, 14, 4),      # Further reduced
    ('Severe', 12, 8, 4.5),       # Very low alpha/beta, high variability
]

all_dfs = []
for class_name, alpha_mean, beta_mean, std_mean in classes_config:
    df = create_class_data(class_name, 500, alpha_mean, beta_mean, std_mean)
    all_dfs.append(df)
    
    # Save individual sample file
    if class_name == 'Normal':
        fname = 'sample_NORMAL_healthy.csv'
    elif class_name == 'Mild Risk':
        fname = 'sample_MILD_risk.csv'
    elif class_name == 'Moderate':
        fname = 'sample_MODERATE_depression.csv'
    else:
        fname = 'sample_SEVERE_depression.csv'
    
    sample_path = os.path.join(SAMPLE_DIR, fname)
    df.head(3).to_csv(sample_path, index=False)  # Save only first 3 for samples
    print(f"[OK] {fname}: {class_name}")
    print(f"      Alpha mean: {alpha_mean}, Beta mean: {beta_mean}")

# Create combined training dataset
df_all = pd.concat(all_dfs, ignore_index=True)
df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

output_path = os.path.join(BASE_DIR, 'EEG_4class_training.csv')
df_all.to_csv(output_path, index=False)

print(f"\n[OK] Combined training dataset: {df_all.shape}")
print(f"[OK] Class distribution:")
print(df_all['Expected_Result'].value_counts())
print(f"\n[SAVED] {output_path}")
print(f"[READY] Run: python retrain_model.py")
