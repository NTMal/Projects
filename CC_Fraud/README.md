# Credit Card Fraud Detection

## Overview
This project is a machine learning project to detect fraudulent credit card
transactions using Decision Tree, Support Vector Machine (SVM), and Random 
Forest classifiers. Models are evaluated across two feature sets; all 28 features
and a reduced set of the top 6 most correlated features, to assess the impact of
feature selection on model performance.

## Dataset
- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions with 492 fraud cases (0.17% of total)
- **Features:** 28 anonymised PCA components (V1–V28), transaction Amount, and Time
- **Target:** Class (0 = Not Fraud, 1 = Fraud)

## Requirements
Install dependencies with:
`pip install -r requirements.txt`

## Setup
1. Create a Kaggle API token at https://www.kaggle.com/settings → API → Create New Token
2. Set the environment variable: `export KAGGLE_API_TOKEN=your_token_here`
3. Run the notebook — the dataset downloads automatically to `/tmp`

## Results
| Model | All Features (ROC-AUC) | Top 6 Features (ROC-AUC) |
|-------|----------------------|--------------------------|
| Decision Tree | 0.939 | 0.952 |
| Linear SVM | 0.986 | 0.937 |
| Random Forest | 0.974 | 0.954 |

## Key Findings
- Random Forest with all features is the strongest model overall, achieving the 
  highest fraud precision (0.71) and recall (0.83).
- Feature selection improves Decision Tree and Random Forest performance but 
  significantly degrades SVM, which relies on high-dimensional space to find 
  an optimal separating hyperplane.
- ROC-AUC alone is insufficient for evaluating fraud detection models. Precision 
  and recall for the minority fraud class are critical metrics for real-world deployment.

  