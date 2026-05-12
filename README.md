# STEG Fraud Detection — Zindi Africa

## Challenge
Detect fraudulent manipulation of electricity and gas meters using 14 years 
of billing history for 135,000 clients in Tunisia.

**Metric:** ROC-AUC  
**My Score:** 0.874  
**Leaderboard Rank:** #161

## Approach
- Cleaned and merged two datasets: client profiles + 4.5M invoice records
- Engineered 80+ features per client: consumption stats, monthly patterns, 
  yearly trends, zero-consumption ratios, account duration, fraud rate encoding
- Trained XGBoost + LightGBM ensemble with 5-fold cross-validation
- Submitted raw probabilities (not binary labels) for better ROC-AUC scoring

## Key Learning
Submitting raw probabilities instead of hard 0/1 predictions jumped my 
score from 0.669 to 0.871 in a single submission — ROC-AUC is rank-based.

## Files
- `STEG_Fraud_Detection.ipynb` — Full pipeline notebook
- `submission_v5.csv` — Best submission file
- `STEG_Fraud_Detection.pptx` — Project presentation

## Tools
Python · XGBoost · LightGBM · Scikit-learn · Pandas · Google Colab
