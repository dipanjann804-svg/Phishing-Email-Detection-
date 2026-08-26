#!/usr/bin/env bash
set -o errexit

pip install -r requirements.txt

# data/ and model/ must exist before these scripts run
mkdir -p data model

# Generate the synthetic dataset and train the model as part of the build,
# so model/phishing_model.joblib exists before the web server starts.
python generate_dataset.py
python train_model.py