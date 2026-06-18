# -*- coding: utf-8 -*-
# ============================================================
# feature_engineering.py
# Task 2 — Feature Engineering for Fraud Detection
#
# Project : Financial Transaction Fraud Detection System
# Author  : Member 2
# Dataset : PaySim Synthetic Financial Dataset
#
# What this script does:
#   Loads the raw PaySim data, engineers new features, encodes
#   categories, scales numeric columns, handles class imbalance
#   using class weights, and saves train/test splits as Parquet.
# ============================================================

import pandas as pd          # for loading and manipulating data
import numpy as np           # for math operations
from sklearn.preprocessing import StandardScaler  # for feature scaling
from sklearn.model_selection import train_test_split  # for splitting data
import warnings

warnings.filterwarnings('ignore')  # suppress non-critical warnings

# ─────────────────────────────────────────────────────────────
# CONFIGURATION — change these paths if needed
# ─────────────────────────────────────────────────────────────
DATA_PATH        = 'paysim.csv'          # raw dataset
TRAIN_OUT_PATH   = 'train_features.parquet'
TEST_OUT_PATH    = 'test_features.parquet'
RANDOM_SEED      = 42                    # for reproducibility

print('=' * 60)
print('  TASK 2 — Feature Engineering')
print('=' * 60)


# ────────────────────────────────────────────────────────────
# STEP 1 — Load & Filter Dataset (re-creates df_filtered)
# ────────────────────────────────────────────────────────────
# We re-create df_filtered from scratch so this script can run
# independently without needing Task 1 to have run first.
print('\n[Step 1] Loading paysim.csv ...')

df = pd.read_csv(DATA_PATH)

# Filter to ONLY TRANSFER and CASH_OUT — fraud never occurs elsewhere
# This is a domain fact confirmed in Task 1 EDA
df_filtered = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()

print(f'  Original rows   : {len(df):,}')
print(f'  Filtered rows   : {len(df_filtered):,}  (TRANSFER + CASH_OUT only)')
print(f'  Fraud rows      : {df_filtered["isFraud"].sum():,}')


# ────────────────────────────────────────────────────────────
# STEP 2 — Engineer Balance Features (same as Task 1)
# ────────────────────────────────────────────────────────────
# These 4 features capture balance manipulation — the core
# signal of fraud. Engineered here so this script is self-contained.
print('\n[Step 2] Engineering balance features ...')

# How much did the SENDER balance change?
df_filtered['balance_diff_orig'] = (
    df_filtered['oldbalanceOrg'] - df_filtered['newbalanceOrig']
)

# How much did the RECEIVER balance change?
df_filtered['balance_diff_dest'] = (
    df_filtered['newbalanceDest'] - df_filtered['oldbalanceDest']
)

# Discrepancy: amount vs sender balance change → fraud signal
# In honest transactions this should be ≈ 0
df_filtered['error_orig'] = (
    df_filtered['amount'] - df_filtered['balance_diff_orig']
)

# Discrepancy: amount vs receiver balance change → fraud signal
df_filtered['error_dest'] = (
    df_filtered['amount'] - df_filtered['balance_diff_dest']
)

print('  Created: balance_diff_orig, balance_diff_dest, error_orig, error_dest')


# ────────────────────────────────────────────────────────────
# STEP 3 — Keep Only Relevant Columns
# ────────────────────────────────────────────────────────────
# We drop: nameOrig, nameDest  → just account IDs, not useful for ML
#          isFlaggedFraud       → only 16 flagged rows, useless
# We keep: all numeric features + type + isFraud (target)
print('\n[Step 3] Selecting model-relevant columns ...')

KEEP_COLUMNS = [
    'step',               # hour of simulation
    'type',               # transaction type (will be encoded below)
    'amount',             # transaction amount
    'oldbalanceOrg',      # sender balance before
    'newbalanceOrig',     # sender balance after
    'oldbalanceDest',     # receiver balance before
    'newbalanceDest',     # receiver balance after
    'balance_diff_orig',  # engineered: sender balance change
    'balance_diff_dest',  # engineered: receiver balance change
    'error_orig',         # engineered: sender discrepancy
    'error_dest',         # engineered: receiver discrepancy
    'isFraud'             # TARGET column — what we want to predict
]

df_model = df_filtered[KEEP_COLUMNS].copy()

print(f'  Columns kept    : {len(KEEP_COLUMNS)}')
print(f'  Columns dropped : nameOrig, nameDest, isFlaggedFraud')
print(f'  DataFrame shape : {df_model.shape}')


# ────────────────────────────────────────────────────────────
# STEP 4 — Encode 'type' Column (Label Encoding)
# ────────────────────────────────────────────────────────────
# ML models only understand numbers, not text like "TRANSFER".
# We map: TRANSFER → 1, CASH_OUT → 2
# (Using 1 and 2 instead of 0 and 1 to avoid confusion with isFraud labels)
print('\n[Step 4] Encoding transaction type column ...')

type_mapping = {
    'TRANSFER': 1,   # sender sends money to another account
    'CASH_OUT': 2    # sender withdraws cash via merchant
}

df_model['type'] = df_model['type'].map(type_mapping)

# Verify no nulls introduced (would happen if unexpected type values exist)
null_check = df_model['type'].isnull().sum()
if null_check > 0:
    print(f'  [WARN] {null_check} rows had unexpected type values!')
else:
    print('  [OK] type encoded: TRANSFER=1, CASH_OUT=2')
    print(f'  TRANSFER rows  : {(df_model["type"]==1).sum():,}')
    print(f'  CASH_OUT rows  : {(df_model["type"]==2).sum():,}')


# ────────────────────────────────────────────────────────────
# STEP 5 — Apply StandardScaler to Numeric Columns
# ────────────────────────────────────────────────────────────
# Why scale? Random Forests don't strictly need scaling, but
# StandardScaler ensures features are on the same range,
# which helps PySpark's internal vector operations perform better.
#
# StandardScaler transforms each value to: (value - mean) / std_dev
# After scaling, each column has mean≈0 and std≈1.
# This prevents large-value columns (like 'amount') from dominating.
print('\n[Step 5] Applying StandardScaler to numeric columns ...')

SCALE_COLUMNS = [
    'amount',
    'oldbalanceOrg',
    'newbalanceOrig',
    'oldbalanceDest',
    'newbalanceDest',
    'balance_diff_orig',
    'balance_diff_dest'
]

# Initialize the scaler — it learns the mean and std of each column
scaler = StandardScaler()

# .fit_transform() does two things in one:
#   fit      → compute mean and std for each column
#   transform → apply the scaling formula to each value
df_model[SCALE_COLUMNS] = scaler.fit_transform(df_model[SCALE_COLUMNS])

print(f'  [OK] Scaled {len(SCALE_COLUMNS)} columns using StandardScaler')
print('  Scaled columns: ' + ', '.join(SCALE_COLUMNS))

# Quick sanity check — mean of each scaled column should be ≈ 0
print('\n  Post-scaling means (should all be near 0):')
for col in SCALE_COLUMNS:
    mean_val = df_model[col].mean()
    print(f'    {col:<22}: {mean_val:.6f}')


# ────────────────────────────────────────────────────────────
# STEP 6 — Class Imbalance: Why Weights, Not SMOTE
# ────────────────────────────────────────────────────────────
print('\n[Step 6] Handling class imbalance with class weights ...')

total_rows = len(df_model)
fraud_rows = df_model['isFraud'].sum()
legit_rows = total_rows - fraud_rows

imbalance_ratio = legit_rows / fraud_rows

print(f'\n  Total rows      : {total_rows:,}')
print(f'  Legit rows (0)  : {legit_rows:,}  ({legit_rows/total_rows*100:.2f}%)')
print(f'  Fraud rows (1)  : {fraud_rows:,}    ({fraud_rows/total_rows*100:.4f}%)')
print(f'  Imbalance ratio : ~{imbalance_ratio:.0f}:1  (legit:fraud)')

print("""
  WHY CLASS WEIGHTS INSTEAD OF SMOTE?
  ─────────────────────────────────────────────────────────
  SMOTE (Synthetic Minority Over-sampling Technique) creates 
  fake synthetic fraud rows to balance the dataset.

  Problem: PySpark MLlib does NOT natively support SMOTE, and 
  applying SMOTE before PySpark means doing it in pandas (slow,
  memory-intensive on 6M rows).

  Better approach: PySpark MLlib's RandomForestClassifier has a
  built-in 'weightCol' parameter. By assigning higher weight to 
  fraud rows, we tell the model "mistakes on fraud matter MORE".
  This is mathematically equivalent to oversampling, but runs
  entirely inside Spark — fast and memory-efficient.
  ─────────────────────────────────────────────────────────
""")


# ────────────────────────────────────────────────────────────
# STEP 7 — Calculate & Assign Class Weights
# ────────────────────────────────────────────────────────────
# Formula: fraud_weight = total / (2 × fraud_count)
# This formula makes the effective contribution of fraud rows
# equal to the contribution of legit rows during training.
#
# Example: if there are 770 legit per 1 fraud, fraud_weight ≈ 385
# So each fraud row "counts" 385× more during model fitting.
print('[Step 7] Calculating and assigning class weights ...')

fraud_weight = total_rows / (2 * fraud_rows)
legit_weight = 1.0   # legit rows keep their natural weight of 1

print(f'\n  fraud_weight formula: {total_rows:,} / (2 × {fraud_rows:,}) = {fraud_weight:.4f}')
print(f'  Fraud row weight : {fraud_weight:.4f}  ← each fraud row counts this much')
print(f'  Legit row weight : {legit_weight:.4f}   ← legit rows keep base weight')

# Add the 'classWeight' column to the DataFrame
# .apply() applies a function to every row in the 'isFraud' column
# If isFraud == 1 (fraud), assign fraud_weight; else assign legit_weight
df_model['classWeight'] = df_model['isFraud'].apply(
    lambda label: fraud_weight if label == 1 else legit_weight
)

print(f'\n  [OK] classWeight column added')
print(f'  Rows with weight {fraud_weight:.2f} : {(df_model["classWeight"] == fraud_weight).sum():,}  (fraud)')
print(f'  Rows with weight {legit_weight:.2f}   : {(df_model["classWeight"] == legit_weight).sum():,}  (legit)')


# ────────────────────────────────────────────────────────────
# STEP 8 — Stratified Train/Test Split (80% / 20%)
# ────────────────────────────────────────────────────────────
# Why stratified? With such extreme imbalance (0.13% fraud),
# a random split might put ALL fraud rows in train or test by chance.
# stratify=df_model['isFraud'] ensures BOTH sets keep the same
# ~0.13% fraud ratio — so test results are representative.
print('\n[Step 8] Splitting into train (80%) and test (20%) with stratification ...')

# Separate features (X) and target (y) — standard ML convention
# X = everything except the label we want to predict
# y = the label column we want to predict
X = df_model.drop(columns=['isFraud'])   # all feature columns + classWeight
y = df_model['isFraud']                  # target label

X_train, X_test, y_train, y_test = train_test_split(
    X,                      # feature matrix
    y,                      # target vector
    test_size=0.20,         # 20% goes to test
    random_state=RANDOM_SEED,  # makes the split reproducible
    stratify=y              # maintain fraud ratio in both splits
)

# Recombine X and y back into full DataFrames for saving
train_df = X_train.copy()
train_df['isFraud'] = y_train.values

test_df = X_test.copy()
test_df['isFraud'] = y_test.values

# Verify fraud ratio is preserved in both splits
train_fraud_pct = train_df['isFraud'].mean() * 100
test_fraud_pct  = test_df['isFraud'].mean()  * 100

print(f'\n  Train set: {len(train_df):,} rows  | Fraud: {train_df["isFraud"].sum():,} ({train_fraud_pct:.4f}%)')
print(f'  Test set : {len(test_df):,} rows  | Fraud: {test_df["isFraud"].sum():,} ({test_fraud_pct:.4f}%)')
print(f'  [OK] Fraud rate preserved in both splits (stratification worked!)')


# ────────────────────────────────────────────────────────────
# STEP 9 — Save Train Set as Parquet
# ────────────────────────────────────────────────────────────
# Parquet is a compressed columnar format — much faster than CSV
# for PySpark to read. This is the industry standard for Spark pipelines.
print(f'\n[Step 9] Saving train set to {TRAIN_OUT_PATH} ...')

train_df.to_parquet(TRAIN_OUT_PATH, index=False)
print(f'  [OK] Train set saved: {TRAIN_OUT_PATH}')


# ────────────────────────────────────────────────────────────
# STEP 10 — Save Test Set as Parquet & Print Final Summary
# ────────────────────────────────────────────────────────────
print(f'\n[Step 10] Saving test set to {TEST_OUT_PATH} ...')

test_df.to_parquet(TEST_OUT_PATH, index=False)
print(f'  [OK] Test set saved : {TEST_OUT_PATH}')

# Final summary
print('\n' + '=' * 60)
print('  FEATURE ENGINEERING COMPLETE — FINAL SUMMARY')
print('=' * 60)
print(f'''
  Input file       : {DATA_PATH}
  Filtered rows    : {len(df_model):,}  (TRANSFER + CASH_OUT only)

  Features used    : {len(KEEP_COLUMNS) - 1}  (excluding isFraud target)
  Columns scaled   : {len(SCALE_COLUMNS)}
  Fraud weight     : {fraud_weight:.4f}

  Train set shape  : {train_df.shape}   → {TRAIN_OUT_PATH}
  Test  set shape  : {test_df.shape}    → {TEST_OUT_PATH}

  Train fraud count: {train_df["isFraud"].sum():,}  ({train_fraud_pct:.4f}%)
  Test  fraud count: {test_df["isFraud"].sum():,}    ({test_fraud_pct:.4f}%)

  Column order in Parquet files:
''')

# Print the final column schema — this is what train_model.py will read
for i, col in enumerate(train_df.columns):
    dtype = str(train_df[col].dtype)
    marker = '← TARGET' if col == 'isFraud' else ('← WEIGHT' if col == 'classWeight' else '')
    print(f'    {i+1:>2}. {col:<22} {dtype:<10} {marker}')

print(f'''
  Next step → Run train_model.py (Task 3) to train the
              RandomForest model using PySpark MLlib.
''')
print('=' * 60)
