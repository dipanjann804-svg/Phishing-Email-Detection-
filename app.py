"""
Flask backend for the Phishing Email Detection project.

Endpoints:
    GET  /                 -> web UI to paste an email and check it
    POST /predict           -> JSON API: {"subject": "...", "body": "...", "sender": "..."} ->
                                {"label": "phishing"|"legitimate", "confidence": 0.0-1.0, ...}
    GET  /model-info         -> accuracy, label classes, and top feature importances
    GET  /confusion-matrix    -> the confusion matrix PNG saved during training

Run:
    python generate_dataset.py   # first time only, or to refresh the dataset
    python train_model.py        # first time only, or to retrain
    python app.py
Then open http://127.0.0.1:5000/
"""

import os
import joblib
from flask import Flask, jsonify, render_template, request, send_file, abort
from scipy.sparse import hstack, csr_matrix

from feature_extraction import combine_text, extract_numeric_features, NUMERIC_FEATURE_NAMES

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "phishing_model.joblib")
CONFUSION_MATRIX_PATH = os.path.join(MODEL_DIR, "confusion_matrix.png")

MAX_INPUT_CHARS = 20000  # guard against pathologically large pastes

app = Flask(__name__)

_bundle = None


def get_bundle():
    """Lazy-load the trained model bundle (model, vectorizer, scaler)."""
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No trained model found at {MODEL_PATH}. Run `python generate_dataset.py` "
                "then `python train_model.py` first."
            )
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def predict_email(subject: str, body: str, sender: str = ""):
    bundle = get_bundle()
    model = bundle["model"]
    vectorizer = bundle["vectorizer"]
    scaler = bundle["scaler"]

    text = combine_text(subject, body)
    numeric_feats = extract_numeric_features(subject, body, sender)

    text_features = vectorizer.transform([text])
    numeric_scaled = scaler.transform([[numeric_feats[name] for name in NUMERIC_FEATURE_NAMES]])
    X = hstack([text_features, csr_matrix(numeric_scaled)])

    pred_label = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    class_index = list(model.classes_).index(pred_label)
    confidence = float(proba[class_index])

    return {
        "label": pred_label,
        "display_label": "Phishing" if pred_label == "phishing" else "Safe",
        "confidence": round(confidence, 4),
        "features": numeric_feats,
        "class_probabilities": {
            cls: round(float(p), 4) for cls, p in zip(model.classes_, proba)
        },
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or request.form
    subject = (data.get("subject") or "")[:MAX_INPUT_CHARS]
    body = (data.get("body") or "")[:MAX_INPUT_CHARS]
    sender = (data.get("sender") or "")[:320]  # RFC 5321 max mailbox length

    if not subject and not body:
        return jsonify({"error": "Provide at least a subject or body."}), 400

    try:
        result = predict_email(subject, body, sender)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503

    return jsonify(result)


@app.route("/model-info")
def model_info():
    try:
        bundle = get_bundle()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    return jsonify({
        "accuracy": bundle.get("accuracy"),
        "labels": bundle.get("labels"),
        "feature_importances": bundle.get("feature_importances", []),
        "has_confusion_matrix": os.path.exists(CONFUSION_MATRIX_PATH),
    })


@app.route("/confusion-matrix")
def confusion_matrix_image():
    if not os.path.exists(CONFUSION_MATRIX_PATH):
        abort(404, description="Confusion matrix not found. Run train_model.py first.")
    return send_file(CONFUSION_MATRIX_PATH, mimetype="image/png")


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
