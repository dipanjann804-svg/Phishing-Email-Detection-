"""
Train the phishing email detection model.

Pipeline:
  1. Load data/emails.csv (subject, body, sender, num_links, label)
  2. Extract TF-IDF text features + hand-crafted numeric/sender features
  3. Train a RandomForestClassifier on the combined feature matrix
  4. Evaluate on a held-out test set: accuracy, precision/recall/F1,
     and a confusion matrix (saved as model/confusion_matrix.png)
  5. Save the trained vectorizer, scaler, model, accuracy, and feature
     importances to model/phishing_model.joblib so app.py can serve them

Run:
    python generate_dataset.py   # only needed once, or to regenerate data
    python train_model.py
"""

import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from feature_extraction import combine_text, extract_numeric_features, NUMERIC_FEATURE_NAMES

DATA_PATH = "data/emails.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "phishing_model.joblib")
CONFUSION_MATRIX_PATH = os.path.join(MODEL_DIR, "confusion_matrix.png")


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No dataset found at {path}. Run `python generate_dataset.py` first "
            "(or drop in your own emails.csv with subject, body, sender, label columns)."
        )
    df = pd.read_csv(path)
    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")
    df["sender"] = df["sender"].fillna("") if "sender" in df.columns else ""
    return df


def build_feature_matrix(df: pd.DataFrame, vectorizer: TfidfVectorizer, scaler: StandardScaler, fit: bool):
    combined_text = [combine_text(s, b) for s, b in zip(df["subject"], df["body"])]
    numeric_rows = [
        extract_numeric_features(s, b, sd)
        for s, b, sd in zip(df["subject"], df["body"], df["sender"])
    ]
    numeric_df = pd.DataFrame(numeric_rows)[NUMERIC_FEATURE_NAMES]

    if fit:
        text_features = vectorizer.fit_transform(combined_text)
        numeric_scaled = scaler.fit_transform(numeric_df.values)
    else:
        text_features = vectorizer.transform(combined_text)
        numeric_scaled = scaler.transform(numeric_df.values)

    combined = hstack([text_features, csr_matrix(numeric_scaled)])
    return combined


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Loading dataset from {DATA_PATH} ...")
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} emails ({df['label'].value_counts().to_dict()})")

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        df, df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(max_features=3000, stop_words="english", ngram_range=(1, 2))
    scaler = StandardScaler()

    print("Extracting features...")
    X_train = build_feature_matrix(X_train_df, vectorizer, scaler, fit=True)
    X_test = build_feature_matrix(X_test_df, vectorizer, scaler, fit=False)

    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=300, max_depth=None, random_state=42, class_weight="balanced", n_jobs=-1
    )
    model.fit(X_train, y_train)

    print("Evaluating on held-out test set...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)
    print(f"\nAccuracy: {acc:.4f}\n")
    print("Classification report:")
    print(report)

    labels = sorted(df["label"].unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(pd.DataFrame(cm, index=labels, columns=labels))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix (Accuracy: {acc:.2%})")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    print(f"\nSaved confusion matrix plot to {CONFUSION_MATRIX_PATH}")

    # Feature importances for the hand-crafted numeric/sender features only
    # (the last len(NUMERIC_FEATURE_NAMES) columns of the combined matrix).
    # TF-IDF importances aren't included since individual n-gram importances
    # aren't meaningful to show a user one-by-one.
    importances = model.feature_importances_[-len(NUMERIC_FEATURE_NAMES):]
    feature_importances = sorted(
        zip(NUMERIC_FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True
    )
    print("\nTop numeric feature importances:")
    for name, score in feature_importances[:5]:
        print(f"  {name}: {score:.4f}")

    joblib.dump(
        {
            "model": model,
            "vectorizer": vectorizer,
            "scaler": scaler,
            "labels": labels,
            "accuracy": acc,
            "classification_report": report,
            "numeric_feature_names": NUMERIC_FEATURE_NAMES,
            "feature_importances": [
                {"feature": name, "importance": round(float(score), 4)}
                for name, score in feature_importances
            ],
        },
        MODEL_PATH,
    )
    print(f"Saved trained model bundle to {MODEL_PATH}")


if __name__ == "__main__":
    main()
