"""
train.py — Train and save both fraud detection models.

Run this script ONCE locally before building the Docker image.
It will produce two .joblib files in the app/models/ folder:
  - models/random_forest.joblib  (supervised fraud classifier)
  - models/isolation_forest.joblib (unsupervised anomaly detector)
  - models/scaler.joblib          (the Amount scaler, needed at inference time)

Usage:
    python app/train.py

Requirements:
    - Kaggle API credentials set up (~/.kaggle/kaggle.json)
    - OR manually place creditcard.csv in /tmp/creditcard.csv
"""

import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, roc_auc_score

# For reproducibility set seed as 42
RANDOM_SEED = 42

# Paths where trained model will be saved
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# Download data

def download_data():
    """
    Download the creditcard.csv dataset from Kaggle.
    Requires Kaggle API credentials at ~/.kaggle/kaggle.json
    If you already have the file, this step is skipped.
    """
    csv_path = "/tmp/creditcard.csv"

    if os.path.exists(csv_path):
        print(f"Dataset already exists at {csv_path}, skipping download.")
        return csv_path

    print("Downloading dataset from Kaggle...")
    result = os.system(
        "kaggle datasets download -d mlg-ulb/creditcardfraud -p /tmp --unzip"
    )
    if result != 0:
        raise RuntimeError(
            "Kaggle download failed. Make sure ~/.kaggle/kaggle.json exists "
            "and contains your API credentials."
        )

    print(f"Dataset downloaded to {csv_path}")
    return csv_path


# Unified preprocessing

def preprocess(df, scaler=None, fit_scaler=False):
    """
    Unified preprocessing pipeline used by both models.
    This ensures the API receives data in exactly the same format
    that both models were trained on.

    Steps:
      1. Drop the 'Time' column (not predictive, create noise)
      2. Scale 'Amount' to zero mean / unit variance
         (V1 to V28 are already PCA-transformed, so they don't need scaling)
      3. Return feature matrix X, label vector y, and the fitted scaler

    Args:
        df         : raw DataFrame loaded from creditcard.csv
        scaler     : a pre-fitted StandardScaler (used at inference time)
        fit_scaler : if True, fit a new scaler on this data (used at training time)

    Returns:
        X          : feature DataFrame (29 columns: V1 to V28 + Amount)
        y          : label Series (0 = legit, 1 = fraud)
        scaler     : the StandardScaler (fitted or passed-through)
    """

    # Drop Time as it is not useful for prediction
    df = df.drop(columns=["Time"])

    # Scale Amount because it's raw £/$ values and has very different scale to V1 to V28
    if fit_scaler:
        # Training time to fit a new scaler and remember it for later
        scaler = StandardScaler()
        df["Amount"] = scaler.fit_transform(df[["Amount"]])
    else:
        # Inference time will use the already-fitted scaler
        if scaler is None:
            raise ValueError("A fitted scaler must be provided when fit_scaler=False")
        df["Amount"] = scaler.transform(df[["Amount"]])

    # Separate features (X) from labels (y)
    X = df.drop(columns=["Class"])  # 29 features: V1 to V28 + Amount
    y = df["Class"]                 # 0 = legitimate, 1 = fraud

    return X, y, scaler


# Train Random Forest (supervised)

def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Train a Random Forest classifier with hyperparameter tuning.

    Why Random Forest?
      - Best fraud precision (0.71) and recall (0.83) in models comparison experiments
      - Handles class imbalance well with class_weight='balanced'
      - Robust to outliers and doesn't need feature normalisation

    Why GridSearchCV + StratifiedKFold?
      - StratifiedKFold preserves the fraud/non-fraud ratio in each fold
        (important when fraud is only ~0.17% of data)
      - GridSearchCV tries all combinations of hyperparameters and picks the best
    """
    print("\nTraining Random Forest ───────────────────────────────────")

    # Hyperparameter grid to search over
    param_grid = {
        "n_estimators": [50, 100],       # number of trees in the forest
        "max_depth": [None, 10, 20],     # how deep each tree can grow
        "min_samples_split": [2, 5],     # minimum samples needed to split a node
    }

    # Stratified 5-fold cross-validation
    # 'stratify' means each fold has the same fraud ratio as the full dataset
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    # Base model with class_weight='balanced' to automatically handles class imbalance
    # by weighting fraud cases more heavily during training
    rf = RandomForestClassifier(class_weight="balanced", random_state=RANDOM_SEED)

    # Grid search: try all param combinations, score by ROC-AUC
    rf_grid = GridSearchCV(
        rf,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",   # optimise for ability to distinguish fraud vs legit
        n_jobs=-1,           # use all available CPU cores
        verbose=1,
    )

    rf_grid.fit(X_train, y_train)

    print(f"Best parameters: {rf_grid.best_params_}")

    # Evaluate on held-out test set
    best_rf = rf_grid.best_estimator_
    y_prob = best_rf.predict_proba(X_test)[:, 1]  # probability of fraud
    y_pred = best_rf.predict(X_test)

    print(f"  ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    return best_rf


# Train Isolation Forest (unsupervised)

def train_isolation_forest(X_train, y_test, X_test):
    """
    Train an Isolation Forest anomaly detector.

    Why Isolation Forest?
      - Trains WITHOUT fraud labels as it only sees 'normal' transactions
      - Achieved recall of 0.84, matching the supervised model
      - Scales well to high-dimensional data (29 features)
      - LOF failed completely (recall = 0.00) in high dimensions

    How it works:
      - Randomly selects a feature and a split value
      - Anomalies (fraud) are isolated in fewer splits than normal transactions
      - contamination='auto' means it doesn't assume a fixed fraud rate
        (more realistic since in production you won't know the true fraud rate)

    Note: we train on X_train only (unsupervised, no labels used)
    """
    print("\n── Training Isolation Forest ───────────────────────────────────")

    iso = IsolationForest(
        n_estimators=100,        # number of isolation trees
        contamination="auto",    # don't assume a fixed fraud rate
        random_state=RANDOM_SEED,
    )

    # Unsupervised: fit on training features only, no labels needed
    iso.fit(X_train)

    # Evaluate against ground truth to check performance
    # predict() returns: 1 = normal, -1 = anomaly
    y_pred_raw = iso.predict(X_test)
    y_pred = (y_pred_raw == -1).astype(int)  # convert to 0/1 for comparison

    print(f"  Predicted fraud cases : {y_pred.sum()}")
    print(f"  Actual fraud cases    : {y_test.sum()}")
    print(f"  Correct detections    : {((y_pred == 1) & (y_test.values == 1)).sum()}")
    print(f"  ROC-AUC               : {roc_auc_score(y_test, y_pred):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    return iso



# Save models

def save_models(rf_model, iso_model, scaler):
    """
    Save trained models and scaler to disk as .joblib files.

    Why joblib instead of pickle?
      - joblib is faster and more efficient for large NumPy arrays
      - sklearn officially recommends joblib for persisting models

    These files will be copied into the Docker image so the API
    can load them without retraining every time the container starts.
    """
    rf_path  = os.path.join(MODELS_DIR, "random_forest.joblib")
    iso_path = os.path.join(MODELS_DIR, "isolation_forest.joblib")
    sc_path  = os.path.join(MODELS_DIR, "scaler.joblib")

    joblib.dump(rf_model, rf_path)
    joblib.dump(iso_model, iso_path)
    joblib.dump(scaler, sc_path)

    print(f"\nSaved: {rf_path}")
    print(f"Saved: {iso_path}")
    print(f"Saved: {sc_path}")


# MAIN

if __name__ == "__main__":

    # 1. Get data
    csv_path = download_data()
    raw_data = pd.read_csv(csv_path)
    print(f"\nLoaded dataset: {raw_data.shape[0]:,} rows, {raw_data.shape[1]} columns")
    print(f"Fraud cases: {raw_data.Class.sum():,} ({raw_data.Class.mean()*100:.3f}%)")

    # 2. Preprocess (fit the scaler here — training time)
    X, y, scaler = preprocess(raw_data, fit_scaler=True)

    # 3. Train/test split
    # test_size=0.2 → 80% train, 20% test
    # stratify=y    → preserve fraud ratio in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"\nTrain set: {X_train.shape[0]:,} rows | Test set: {X_test.shape[0]:,} rows")

    # 4. Train models
    rf_model  = train_random_forest(X_train, y_train, X_test, y_test)
    iso_model = train_isolation_forest(X_train, y_test, X_test)

    # 5. Save everything
    save_models(rf_model, iso_model, scaler)

    print("\nTraining complete. Ready to build Docker image.")
