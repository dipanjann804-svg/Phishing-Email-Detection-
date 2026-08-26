# Phishing Email Detector

A small end-to-end ML project: train a scikit-learn model on phishing vs.
legitimate emails, extract and analyze email features, and serve real-time
predictions through a Flask web UI.

## Setup

```bash
pip install -r requirements.txt

# 1. Generate the synthetic training dataset -> data/emails.csv
python generate_dataset.py

# 2. Train the model -> model/phishing_model.joblib + model/confusion_matrix.png
python train_model.py

# 3. Run the web app
python app.py
```

Then open http://127.0.0.1:5000/

Run steps 1 and 2 in order before step 3 — `app.py` reads the trained model
bundle from `model/phishing_model.joblib`, which only exists after training.

## What it does

- **Trains on labeled email data** (`data/emails.csv`: subject, body, sender,
  label) using a `RandomForestClassifier` over combined TF-IDF text features
  and hand-crafted numeric features (`train_model.py`).
- **Extracts and analyzes features** (`feature_extraction.py`): link counts,
  IP-literal URLs, suspicious keyword hits, punctuation/capitalization
  patterns, and sender-domain signals (suspicious TLDs, sender-vs-link domain
  mismatch — a classic spoofing indicator).
- **Classifies emails as Phishing or Safe** with a confidence score, served
  via `POST /predict` and shown in the web UI.
- **Displays accuracy and confusion matrix** in the UI itself: `GET
  /model-info` returns held-out accuracy and top feature importances, and
  `GET /confusion-matrix` serves the confusion matrix image generated during
  training.

## Project structure

```
generate_dataset.py    # builds the synthetic training set
feature_extraction.py  # shared feature logic (used by train_model.py and app.py)
train_model.py          # trains, evaluates, and saves the model bundle
app.py                   # Flask app: /, /predict, /model-info, /confusion-matrix
templates/index.html      # web UI
data/emails.csv            # generated dataset (not committed)
model/                      # generated model + confusion matrix (not committed)
```

## Known limitations

This dataset is **synthetic** (template-based) and meant to demonstrate the
pipeline, not for production use. Real deployment should train on a larger,
real-world corpus (e.g. the Kaggle "Phishing Email Detection" dataset, or the
Nazario / SpamAssassin / Enron corpora) and re-validate the feature set
against real spoofing patterns.
