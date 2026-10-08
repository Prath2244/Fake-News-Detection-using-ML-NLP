import pandas as pd
import numpy as np

files = ['sample_MILD_risk.csv', 'sample_MODERATE_depression.csv', 'sample_SEVERE_depression.csv']
for fname in files:
    df = pd.read_csv(f'sample_files/{fname}')
    feat_cols = [c for c in df.columns if c not in ['Expected_Result', 'HRV']]
    means = df[feat_cols].mean()
    alpha_cols = [c for c in means.index if 'alpha' in c]
    beta_cols = [c for c in means.index if 'beta' in c]
    
    print(f'\n{fname}:')
    print(f'  Expected: {df["Expected_Result"].iloc[0]}')
    print(f'  Mean Fp1_alpha: {means["Fp1_alpha"]:.2f}')
    print(f'  Mean alpha avg: {means[alpha_cols].mean():.2f}')
    print(f'  Mean beta avg: {means[beta_cols].mean():.2f}')
