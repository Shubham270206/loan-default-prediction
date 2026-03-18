# Loan Default Prediction

A machine learning project to predict whether a borrower will experience financial distress within two years, using logistic regression on real-world credit data.

## Problem Statement
Banks and financial institutions need to assess credit risk before approving loans. This project builds a binary classifier to identify high-risk applicants using historical borrower data.

## Dataset
- Source: [Give Me Some Credit - Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit)
- 150,000 borrower records, 10 features
- Target variable: `SeriousDlqin2yrs` (1 = defaulted, 0 = did not)

## Approach
1. Exploratory Data Analysis (EDA)
2. Data cleaning — handled missing values and outliers
3. Logistic Regression baseline model
4. Addressed class imbalance using `class_weight='balanced'`
5. Model evaluation using classification report, confusion matrix, ROC-AUC

## Results
| Model | Recall (Defaulters) | ROC-AUC |
|---|---|---|
| Baseline | 6% | 0.80 |
| Balanced | 73% | 0.82 |

## Tech Stack
- Python, pandas, scikit-learn, matplotlib, seaborn

## Key Insight
A 93% accurate model was actually failing — it missed 94% of real defaulters due to class imbalance. Fixing this with balanced class weights improved defaulter recall from 6% to 73%.