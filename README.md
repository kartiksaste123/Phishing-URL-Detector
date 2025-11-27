# 🔒 Phishing URL Detection System

A comprehensive machine learning project for detecting phishing URLs using multiple algorithms (both from scratch and library implementations). This project includes 9 different models, a Streamlit web application, and comprehensive visualization tools.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Available Models](#available-models)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Streamlit Web App](#streamlit-web-app)
- [Deployment](#deployment)
- [Model Performance](#model-performance)
- [Dataset](#dataset)
- [Files Description](#files-description)

## 🎯 Overview

This project implements a phishing URL detection system that uses machine learning to classify URLs as either legitimate or phishing. The system extracts 29 features from URLs and uses various ML algorithms to make predictions. The project includes both custom implementations (from scratch) and library-based implementations for comparison.

## ✨ Features

- **9 Different ML Models**: Including from-scratch implementations and library-based models
- **Interactive Web Interface**: Beautiful Streamlit app for easy URL testing
- **Comprehensive Visualizations**: Confusion matrices, ROC curves, feature importance plots
- **Real-time Predictions**: Instant URL analysis with confidence scores
- **Model Comparison**: Side-by-side performance metrics
- **Report Generation**: Automated graph generation for reports
- **Cloud Deployment Ready**: Configured for Streamlit Cloud deployment

## 🤖 Available Models

1. **KNN (From Scratch)** - K-Nearest Neighbors implemented from scratch
2. **KNN (sklearn)** - K-Nearest Neighbors using sklearn
3. **Logistic Regression (sklearn)** - Logistic Regression classifier
4. **Decision Tree (From Scratch)** - Decision Tree implemented from scratch
5. **Random Forest (sklearn)** - Random Forest ensemble classifier
6. **XGBoost** - Gradient Boosting classifier
7. **SVM (From Scratch)** - Support Vector Machine implemented from scratch
8. **SVM (sklearn)** - Support Vector Machine using sklearn
9. **Naive Bayes (sklearn)** - Gaussian Naive Bayes classifier

## 📁 Project Structure

```
AIML/
├── CodeAndModels/                    # Main project directory
│   ├── train_model.py               # Train all models
│   ├── test_url.py                  # Test URLs via command line
│   ├── streamlit_app.py             # Streamlit web application
│   ├── generate_report_graphs.py    # Generate visualization graphs
│   ├── requirements.txt             # Python dependencies
│   ├── *.pkl                        # Trained model files (9 files)
│   ├── report_graphs/               # Generated visualization graphs
│   │   ├── *_confusion_matrix.png
│   │   ├── *_roc_curve.png
│   │   ├── *_feature_importance.png
│   │   └── model_comparison_*.png
│   └── .streamlit/
│       └── config.toml             # Streamlit configuration
├── Phishing_Legitimate_cleaned.csv  # Training dataset
└── README.md                        # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Install Dependencies

```bash
cd CodeAndModels
pip install -r ../requirements.txt
```

Or install manually:
```bash
pip install streamlit numpy pandas matplotlib seaborn scikit-learn xgboost
```

### Step 2: Verify Installation

```bash
python -c "import streamlit; print('✓ All dependencies installed')"
```

## 📖 Usage

### Training Models

Train individual models or all models at once:

```bash
cd CodeAndModels
python train_model.py
```

You'll be presented with a menu:
- Options 1-9: Train individual models
- Option 10: Train all models at once

**Note**: Training all models may take 10-30 minutes depending on your system.

### Testing URLs (Command Line)

Test URLs via command line interface:

```bash
cd CodeAndModels
python test_url.py
```

1. Select which trained model to use
2. Enter URLs to check
3. View predictions with confidence scores

### Generating Report Graphs

Generate visualization graphs for all models:

```bash
cd CodeAndModels
python generate_report_graphs.py
```

This creates a `report_graphs/` directory with:
- Individual confusion matrices for each model
- ROC curves (for models with probability support)
- Feature importance plots (for applicable models)
- Comparison charts (accuracy, precision/recall/F1)
- Performance summary CSV

## 🌐 Streamlit Web App

### Local Deployment

Run the Streamlit app locally:

```bash
cd CodeAndModels
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

### Features

- **Model Selection**: Choose from 9 different trained models
- **URL Analysis**: Enter any URL to check if it's phishing or legitimate
- **Real-time Predictions**: Get instant results with confidence scores
- **Performance Metrics**: View accuracy, precision, recall, and F1-score for each model
- **Visualizations**: See confusion matrices, ROC curves, and feature importance graphs
- **Beautiful UI**: Modern, responsive interface with color-coded results

### Usage

1. Select a model from the sidebar dropdown
2. Enter a URL in the input field
3. Click "Analyze URL" to get prediction
4. View detailed results, probabilities, and visualizations

## ☁️ Deployment

### Deploy to Streamlit Cloud

#### Step 1: Push to GitHub

```bash
# Navigate to project root
cd "D:\CODING\Github Projects\AIML"

# Initialize git (if not already done)
git init
git add .
git commit -m "Ready for Streamlit Cloud deployment"

# Add your GitHub repository
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

#### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository
5. **Set Main file path**: `CodeAndModels/streamlit_app.py`
6. Click "Deploy"

#### Step 3: Access Your App

- Wait 1-2 minutes for deployment
- Your app will be live at: `https://your-username-your-app-name.streamlit.app`

### Important Notes for Deployment

- **Model Files**: Ensure all 9 `.pkl` files are in the `CodeAndModels/` directory and committed to GitHub
- **File Size**: If files are >100MB, use Git LFS:
  ```bash
  git lfs install
  git lfs track "*.pkl"
  git add .gitattributes
  ```
- **Dataset File**: Optional for deployment (only needed for retraining or graph generation)
- **Directory Name**: The directory is named `CodeAndModels` (without spaces or special characters) to avoid deployment issues

## 📊 Model Performance

Performance metrics (approximate, may vary):

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| KNN (From Scratch) | 94.7% | 94.3% | 94.9% | 94.6% |
| KNN (sklearn) | 96.9% | 96.5% | 97.1% | 96.8% |
| Logistic Regression | 91.3% | 91.1% | 91.0% | 91.1% |
| Decision Tree (From Scratch) | 94.5% | 94.2% | 94.8% | 94.5% |
| Random Forest | 99.0% | 99.1% | 98.8% | 98.9% |
| XGBoost | 99.7% | 99.7% | 99.6% | 99.6% |
| SVM (From Scratch) | 91.8% | 91.5% | 92.1% | 91.8% |
| SVM (sklearn) | 92.1% | 91.1% | 92.9% | 92.0% |
| Naive Bayes | 82.8% | 96.1% | 67.7% | 79.4% |

**Note**: XGBoost and Random Forest typically achieve the highest accuracy.

## 📦 Dataset

The models are trained on `Phishing_Legitimate_cleaned.csv` which contains:
- **29 features** extracted from URLs
- **Binary classification**: Legitimate (0) or Phishing (1)
- **~10,000 samples** (80% train, 20% test split)

### Features Extracted

The system extracts 29 features from URLs including:
- URL length and structure
- Domain characteristics
- Special characters count
- Path and query parameters
- Security indicators (HTTPS, IP addresses)
- Suspicious patterns (sensitive words, brand names)

## 📄 Files Description

### Core Files

All core files are in the `CodeAndModels/` directory:

- **`train_model.py`**: Main training script with menu system to train any of the 9 models
- **`test_url.py`**: Command-line interface for testing URLs with trained models
- **`streamlit_app.py`**: Streamlit web application for interactive URL detection
- **`generate_report_graphs.py`**: Script to generate all visualization graphs for reports

### Model Files

- **`knn_model.pkl`** / **`knn_scratch_model.pkl`**: KNN from scratch model
- **`knn_sklearn_model.pkl`**: KNN sklearn model
- **`logistic_regression_model.pkl`**: Logistic Regression model
- **`decision_tree_scratch_model.pkl`**: Decision Tree from scratch model
- **`random_forest_model.pkl`**: Random Forest model
- **`xgboost_model.pkl`**: XGBoost model
- **`svm_scratch_model.pkl`**: SVM from scratch model
- **`svm_sklearn_model.pkl`**: SVM sklearn model
- **`naive_bayes_model.pkl`**: Naive Bayes model

### Configuration Files

- **`requirements.txt`**: Python package dependencies
- **`.streamlit/config.toml`**: Streamlit configuration (theme, server settings)

## 🔧 Technical Details

### Feature Extraction

The URL feature extractor analyzes:
- URL structure (length, depth, components)
- Domain characteristics (subdomain level, hostname length)
- Special characters and patterns
- Security indicators
- Suspicious keywords and brand names

### Model Training

- **Normalization**: Min-Max scaling for all features
- **Train-Test Split**: 80-20 split with random seed for reproducibility
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

### Custom Implementations

Three models are implemented from scratch:
1. **KNN**: Uses Euclidean distance and majority voting
2. **Decision Tree**: Uses Gini impurity for splitting
3. **SVM**: Simplified version using gradient descent

## 🎨 Visualizations

The project generates comprehensive visualizations:

- **Confusion Matrices**: Heatmaps showing TP, TN, FP, FN
- **ROC Curves**: Receiver Operating Characteristic curves with AUC scores
- **Feature Importance**: Top 15 most important features (for tree-based models)
- **Comparison Charts**: Side-by-side comparison of all models
- **Performance Summary**: CSV file with all metrics

## 🐛 Troubleshooting

### Models Not Loading

- Ensure all `.pkl` files are in the `Code & Models/` directory
- Check that `train_model.py` is in the same directory (for pickle loading)
- Verify file names match exactly

### Graphs Not Showing

- Run `generate_report_graphs.py` to generate graphs
- Ensure `report_graphs/` directory exists
- Check that graph files are named correctly

### Streamlit Deployment Issues

- Verify `requirements.txt` has all dependencies
- Check that main file path is `Code & Models/streamlit_app.py`
- View Streamlit Cloud logs for detailed error messages

## 📝 Example URLs

### Legitimate URLs
- `https://www.google.com`
- `https://youtube.com`
- `https://github.com`

### Potentially Phishing URLs
- `http://192.168.1.100/login/verify?account=update`
- `https://secure-paypal-verify.com/account/update`
- `http://suspicious-site.com/login/verify?account=update&action=required`

## 🤝 Contributing

This is an academic/educational project. Feel free to:
- Improve model implementations
- Add new features
- Optimize performance
- Enhance the UI

## 📄 License

This project is for educational purposes. Use responsibly.

## ⚠️ Disclaimer

This tool is for educational and research purposes only. Always verify URLs through official channels. The predictions are based on machine learning models and may not be 100% accurate.

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review Streamlit Cloud logs (if deployed)
3. Test locally first: `streamlit run streamlit_app.py`

---

**Built with**: Python, Streamlit, scikit-learn, XGBoost, NumPy, Pandas, Matplotlib, Seaborn

**Last Updated**: November 2025

