# Phishing URL Detector

Machine learning classifier that identifies phishing URLs using feature extraction and ensemble methods. Trained on labeled datasets containing thousands of legitimate and malicious URLs.

## Overview

| | |
|---|---|
| **Domain** | Machine Learning / Cybersecurity |
| **Language** | Python |
| **Libraries** | scikit-learn, pandas, NumPy |
| **Task** | Binary URL Classification (Phishing vs Legitimate) |

## How It Works

URLs are parsed and decomposed into measurable features that distinguish phishing patterns from legitimate ones. These features are fed into a trained classifier that outputs a confidence score and binary verdict.

### Feature Engineering

| Feature Category | Examples |
|---|---|
| **Lexical** | URL length, special char count, digit ratio, subdomains |
| **Domain** | IP-based URL, HTTPS presence, domain age |
| **Path** | Depth, redirect count, suspicious keywords |
| **Statistical** | Entropy, token frequency |

### Models Evaluated

- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)

Best-performing model selected based on F1-score and ROC-AUC on held-out test set.

## Project Structure

```
Phishing-URL-Detector/
├── data/
│   ├── raw/                # Raw URL datasets
│   └── processed/          # Feature-engineered datasets
├── src/
│   ├── feature_extraction.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/                 # Serialized trained models
├── notebooks/              # EDA and experiments
├── requirements.txt
└── README.md
```

## Getting Started

```bash
git clone https://github.com/kartiksaste123/Phishing-URL-Detector
cd Phishing-URL-Detector
pip install -r requirements.txt

# Train the model
python src/train.py

# Predict on a URL
python src/predict.py --url "http://example.com/login"
```

## Performance

| Metric | Score |
|---|---|
| Accuracy | See evaluation notebook |
| F1-Score | See evaluation notebook |
| ROC-AUC | See evaluation notebook |

## Dataset

Trained on publicly available phishing URL datasets. See `data/` directory for sources and preprocessing steps.

## Dependencies

```
scikit-learn
pandas
numpy
matplotlib
seaborn
joblib
```
