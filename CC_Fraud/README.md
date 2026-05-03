# Credit Card Fraud Detection & Anomaly Detection

## Overview
This folder contains two related notebooks that approach credit card fraud detection from different angles using the same Kaggle dataset:

1. **Credit Card Fraud Detection** — supervised learning using labelled fraud data
2. **Anomaly Detection** — unsupervised learning without using fraud labels during training

## Dataset
- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions with 492 fraud cases (0.172% of total)
- **Features:** 28 anonymised PCA components (V1–V28), transaction Amount and Time
- **Target:** Class (0 = Valid, 1 = Fraud)

## Requirements
Install dependencies with:
`pip install -r requirements.txt`

## Setup
1. Create a Kaggle API token at https://www.kaggle.com/settings → API → Create New Token
2. Set the environment variable: `export KAGGLE_API_TOKEN=your_token_here`
3. Run the notebook — the dataset downloads automatically to `/tmp`

---

## Notebook 1: Credit Card Fraud Detection (Supervised Learning)
'CC_Fraud_Detection_Tree_SVM_RF.ipynb'

- Detects fraudulent transactions using three supervised classifiers evaluated across two feature sets; all 28 features and a reduced set of the top 6 most 
correlated features to assess the impact of feature selection on model performance.

### Models
| Model | All Features (ROC-AUC) | Top 6 Features (ROC-AUC) |
|-------|----------------------|--------------------------|
| Decision Tree | 0.939 | 0.952 |
| Linear SVM | 0.986 | 0.937 |
| Random Forest | 0.974 | 0.954 |

### Key Findings
- Random Forest with all features is the strongest model overall, achieving the highest fraud precision (0.71) and recall (0.83).
- Feature selection improves Decision Tree and Random Forest but significantly degrades SVM, which relies on high-dimensional space to find an optimal separating hyperplane.
- ROC-AUC alone is insufficient for evaluating fraud detection models. Precision and Recall for the minority fraud class are critical metrics for real-world deployment.

---

## Notebook 2: Anomaly Detection (Unsupervised Learning)
'Anomaly_Detection_CCFraud.ipynb'

- Detects fraudulent transactions as anomalies using two unsupervised ML models trained without fraud labels. Performance is evaluated against ground truth labels to benchmark detection capability.

### Models
| Model | Fraud Recall | ROC-AUC | PR-AUC |
|-------|-------------|---------|--------|
| Isolation Forest | 0.84 | 0.95 | 0.17 |
| Local Outlier Factor | 0.00 | 0.51 | 0.00 |

### Key Findings
- Isolation Forest correctly identified 82 out of 98 fraud cases (recall = 0.84) without ever seeing a fraud label during training.
- Local Outlier Factor failed to detect any fraud cases indicating distance-based density metrics degrade in the 29-dimensional feature space.
- ROC-AUC of 0.95 for Isolation Forest is partly inflated by the large number of true negatives. PR-AUC of 0.17 is the more honest metric for this imbalanced dataset.
- High dimensionality is a key factor. Isolation Forest's random split approach significantly outperforms LOF's distance-based approach in high dimensional space.

---

## Comparison: Supervised vs Unsupervised
| Approach | Best Model | Fraud Recall | ROC-AUC |
|----------|-----------|-------------|---------|
| Supervised | Random Forest | 0.83 | 0.974 |
| Unsupervised | Isolation Forest | 0.84 | 0.95 |

Interestingly, Isolation Forest achieves comparable fraud recall to the best supervised model despite never seeing fraud labels during training, highlighting 
the power of unsupervised anomaly detection for fraud detection.