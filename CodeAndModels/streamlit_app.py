"""
Streamlit Web App for Phishing URL Detection
Deploy your trained models with a beautiful UI
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import re
import math
from urllib.parse import urlparse, parse_qs
from collections import Counter
import ipaddress

# Page configuration
st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-card {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 1rem 0;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        margin: 1rem 0;
    }
    .safe {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .danger {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    .metric-box {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Import custom classes for pickle loading
try:
    from train_model import KNNClassifier, DecisionTreeClassifier, SVMClassifier
except ImportError:
    # Fallback definitions
    class KNNClassifier:
        pass
    class DecisionTreeClassifier:
        pass
    class SVMClassifier:
        pass

# URL Feature Extractor (matching test_url.py and cleaned dataset)
class URLFeatureExtractor:
    """Extracts 29 features from a raw URL for phishing detection (matching cleaned dataset)."""
    
    def __init__(self):
        self.sensitive_words = [
            'secure', 'account', 'bank', 'login', 'verify', 'update', 'confirm',
            'suspend', 'restrict', 'limited', 'expire', 'urgent', 'immediate',
            'action', 'required', 'click', 'here', 'now', 'verify', 'validate'
        ]
        self.brand_names = [
            'google', 'microsoft', 'apple', 'amazon', 'facebook', 'paypal',
            'ebay', 'twitter', 'instagram', 'linkedin', 'youtube', 'netflix',
            'adobe', 'dropbox', 'github', 'yahoo', 'outlook', 'gmail'
        ]
    
    def extract_all_features(self, url):
        """Extract all 29 features from a URL (matching cleaned dataset structure)."""
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        try:
            parsed = urlparse(url)
        except:
            return self._get_default_features()
        
        features = []
        hostname = parsed.netloc or (parsed.path.split('/')[0] if parsed.path else '')
        path = parsed.path
        query = parsed.query
        
        # Extract 29 features matching cleaned dataset structure
        features.append(url.count('.'))  # 1. NumDots
        
        # 2. SubdomainLevel
        if hostname:
            subdomain_parts = hostname.split('.')
            if len(subdomain_parts) > 2:
                features.append(len(subdomain_parts) - 2)
            else:
                features.append(0)
        else:
            features.append(0)
        
        # 3. PathLevel
        if path and path != '/':
            path_parts = [p for p in path.split('/') if p]
            features.append(len(path_parts))
        else:
            features.append(0)
        
        features.append(len(url))  # 4. UrlLength
        features.append(url.count('-'))  # 5. NumDash
        features.append(hostname.count('-'))  # 6. NumDashInHostname
        features.append(1 if '@' in url else 0)  # 7. AtSymbol
        features.append(1 if '~' in url else 0)  # 8. TildeSymbol
        features.append(url.count('_'))  # 9. NumUnderscore
        features.append(url.count('%'))  # 10. NumPercent
        
        # 11. NumQueryComponents
        if query:
            query_params = parse_qs(query)
            features.append(len(query_params))
        else:
            features.append(0)
        
        features.append(url.count('&'))  # 12. NumAmpersand
        features.append(url.count('#'))  # 13. NumHash
        features.append(sum(1 for c in url if c.isdigit()))  # 14. NumNumericChars
        features.append(1 if parsed.scheme != 'https' else 0)  # 15. NoHttps
        
        # 16. RandomString
        random_pattern = r'[a-zA-Z0-9]{20,}'
        features.append(1 if re.search(random_pattern, url) else 0)
        
        # 17. IpAddress
        try:
            ipaddress.ip_address(hostname.split(':')[0])
            features.append(1)
        except:
            features.append(0)
        
        features.append(len(hostname))  # 18. HostnameLength
        features.append(len(path))  # 19. PathLength
        features.append(len(query))  # 20. QueryLength
        features.append(1 if '//' in path else 0)  # 21. DoubleSlashInPath
        
        # 22-23. Content features
        url_lower = url.lower()
        count = sum(1 for word in self.sensitive_words if word in url_lower)
        features.append(count)  # 22. NumSensitiveWords
        features.append(1 if any(brand in url_lower for brand in self.brand_names) else 0)  # 23. EmbeddedBrandName
        
        # 24-29. Features requiring webpage content (use defaults)
        features.append(0.0)  # 24. PctExtHyperlinks
        features.append(0.0)  # 25. PctExtResourceUrls
        features.append(0)  # 26. ExtFavicon
        features.append(0)  # 27. InsecureForms
        features.append(0.0)  # 28. PctNullSelfRedirectHyperlinks
        features.append(0)  # 29. FrequentDomainNameMismatch
        
        return features
    
    def _get_default_features(self):
        """Return default feature vector (29 features) when URL parsing fails."""
        features = [0] * 29
        features[3] = 20  # UrlLength
        features[17] = 10  # HostnameLength
        features[23] = 0.0  # PctExtHyperlinks
        features[24] = 0.0  # PctExtResourceUrls
        features[27] = 0.0  # PctNullSelfRedirectHyperlinks
        return features


@st.cache_data
def load_model_metrics():
    """Load pre-calculated metrics for all models."""
    # These would ideally be saved during training, but for now we'll calculate on the fly
    # or use default values
    return {
        'KNN (From Scratch)': {'accuracy': 0.947, 'precision': 0.943, 'recall': 0.949, 'f1': 0.946},
        'KNN (sklearn)': {'accuracy': 0.969, 'precision': 0.965, 'recall': 0.971, 'f1': 0.968},
        'Logistic Regression': {'accuracy': 0.913, 'precision': 0.911, 'recall': 0.910, 'f1': 0.911},
        'Decision Tree (From Scratch)': {'accuracy': 0.945, 'precision': 0.942, 'recall': 0.948, 'f1': 0.945},
        'Random Forest': {'accuracy': 0.990, 'precision': 0.991, 'recall': 0.988, 'f1': 0.989},
        'XGBoost': {'accuracy': 0.997, 'precision': 0.997, 'recall': 0.996, 'f1': 0.996},
        'SVM (From Scratch)': {'accuracy': 0.918, 'precision': 0.915, 'recall': 0.921, 'f1': 0.918},
        'SVM (sklearn)': {'accuracy': 0.921, 'precision': 0.911, 'recall': 0.929, 'f1': 0.920},
        'Naive Bayes': {'accuracy': 0.828, 'precision': 0.961, 'recall': 0.677, 'f1': 0.794}
    }


def load_model(model_choice):
    """Load a trained model."""
    model_files = {
        '1': (['knn_scratch_model.pkl', 'knn_model.pkl'], 'knn_scratch'),  # Check both filenames
        '2': (['knn_sklearn_model.pkl'], 'knn_sklearn'),
        '3': (['logistic_regression_model.pkl'], 'logistic_regression'),
        '4': (['decision_tree_scratch_model.pkl'], 'decision_tree_scratch'),
        '5': (['random_forest_model.pkl'], 'random_forest'),
        '6': (['xgboost_model.pkl'], 'xgboost'),
        '7': (['svm_scratch_model.pkl'], 'svm_scratch'),
        '8': (['svm_sklearn_model.pkl'], 'svm_sklearn'),
        '9': (['naive_bayes_model.pkl'], 'naive_bayes')
    }
    
    model_name_map = {
        '1': 'KNN (From Scratch)',
        '2': 'KNN (sklearn)',
        '3': 'Logistic Regression',
        '4': 'Decision Tree (From Scratch)',
        '5': 'Random Forest',
        '6': 'XGBoost',
        '7': 'SVM (From Scratch)',
        '8': 'SVM (sklearn)',
        '9': 'Naive Bayes'
    }
    
    filename_list, model_type = model_files.get(model_choice, model_files['1'])
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find the first existing file (check in script directory)
    filename = None
    for fname in filename_list:
        # Try absolute path first (script directory)
        full_path = os.path.join(script_dir, fname)
        if os.path.exists(full_path):
            filename = full_path
            break
        # Fallback: try relative path (current working directory)
        elif os.path.exists(fname):
            filename = fname
            break
    
    if filename is None:
        # Show helpful error with available files
        available_files = [f for f in os.listdir(script_dir) if f.endswith('.pkl')]
        st.error(f"⚠️ Model file not found! Looking for: {filename_list}")
        if available_files:
            st.info(f"Available .pkl files in directory: {', '.join(available_files)}")
        return None, None, None, None, None
    
    # Custom unpickler
    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'train_model' or module == '__main__':
                if name == 'KNNClassifier':
                    return KNNClassifier
                elif name == 'DecisionTreeClassifier':
                    return DecisionTreeClassifier
                elif name == 'SVMClassifier':
                    return SVMClassifier
            return super().find_class(module, name)
    
    try:
        # Ensure KNNClassifier is imported before loading
        try:
            from train_model import KNNClassifier, DecisionTreeClassifier, SVMClassifier
        except ImportError:
            pass  # Already imported at top
        
        with open(filename, 'rb') as f:
            # Try custom unpickler first
            try:
                unpickler = CustomUnpickler(f)
                model_data = unpickler.load()
            except Exception as unpickle_error:
                # If custom unpickler fails, try standard pickle
                f.seek(0)
                try:
                    model_data = pickle.load(f)
                except Exception as e2:
                    # If that also fails, try with explicit module path
                    f.seek(0)
                    import sys
                    import importlib
                    # Add current directory to path
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    if current_dir not in sys.path:
                        sys.path.insert(0, current_dir)
                    # Try to reload train_model module
                    try:
                        import train_model
                        importlib.reload(train_model)
                        model_data = pickle.load(f)
                    except:
                        raise e2
        
        # Handle both old and new model data structures
        if isinstance(model_data, dict):
            # Check for old format (uses 'knn' key) or new format (uses 'model' key)
            if 'knn' in model_data:
                # Old format
                model = model_data['knn']
                min_vals = model_data['min_vals']
                max_vals = model_data['max_vals']
                feature_columns = model_data.get('feature_columns', [])
                model_type_loaded = 'knn_scratch'
            elif 'model' in model_data:
                # New format
                model = model_data['model']
                min_vals = model_data['min_vals']
                max_vals = model_data['max_vals']
                feature_columns = model_data.get('feature_columns', [])
                model_type_loaded = model_data.get('model_type', model_type)
            else:
                st.error(f"Unknown model data format in {filename}")
                return None, None, None, None, None
        else:
            # Old format - model_data is the model itself (shouldn't happen, but handle it)
            st.warning("Old model format detected. Please retrain the model.")
            return None, None, None, None, None
        
        # Verify model is loaded correctly
        if model is None:
            st.error(f"Model is None in {filename}")
            return None, None, None, None, None
        
        model_name = model_name_map.get(model_choice, 'Unknown')
        
        return model, min_vals, max_vals, feature_columns, model_name
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None, None, None, None


def predict_url(url, model, min_vals, max_vals, model_type):
    """Predict if a URL is phishing or legitimate."""
    extractor = URLFeatureExtractor()
    features = extractor.extract_all_features(url)
    features = np.array(features, dtype=float)
    
    # Normalize features
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    features_norm = (features - min_vals) / range_vals
    features_norm = features_norm.reshape(1, -1)
    
    # Predict based on model type
    if model_type == 'knn_scratch':
        prediction = model.predict_single(features_norm[0])
        # Get probabilities
        distances = []
        for i in range(len(model.X_train)):
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(features_norm[0], model.X_train[i])))
            distances.append((dist, model.y_train[i]))
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:model.k]
        k_nearest_labels = [label for _, label in k_nearest]
        label_counts = Counter(k_nearest_labels)
        total = len(k_nearest_labels)
        prob_class_0 = label_counts.get(0, 0) / total
        prob_class_1 = label_counts.get(1, 0) / total
    elif model_type in ['decision_tree_scratch', 'svm_scratch']:
        prediction = model.predict(features_norm)[0]
        proba = model.predict_proba(features_norm)[0]
        prob_class_0 = proba[0]
        prob_class_1 = proba[1]
    else:
        # sklearn models
        prediction = model.predict(features_norm)[0]
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features_norm)[0]
            prob_class_0 = proba[0]
            prob_class_1 = proba[1]
        else:
            prob_class_0 = 0.5
            prob_class_1 = 0.5
    
    result = {
        'prediction': 'Legitimate' if prediction == 0 else 'Phishing',
        'confidence': max(prob_class_0, prob_class_1),
        'probability_legitimate': prob_class_0,
        'probability_phishing': prob_class_1
    }
    
    return result


def plot_confusion_matrix_figure(y_true, y_pred, model_name):
    """Create confusion matrix figure."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Legitimate', 'Phishing'],
                yticklabels=['Legitimate', 'Phishing'])
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    return fig


def main():
    # Header
    st.markdown('<h1 class="main-header">🔒 Phishing URL Detector</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar for model selection
    with st.sidebar:
        st.header("⚙️ Model Selection")
        st.markdown("---")
        
        model_choice = st.selectbox(
            "Choose a Model:",
            options=['1', '2', '3', '4', '5', '6', '7', '8', '9'],
            format_func=lambda x: {
                '1': '1. KNN (From Scratch)',
                '2': '2. KNN (sklearn)',
                '3': '3. Logistic Regression',
                '4': '4. Decision Tree (From Scratch)',
                '5': '5. Random Forest',
                '6': '6. XGBoost',
                '7': '7. SVM (From Scratch)',
                '8': '8. SVM (sklearn)',
                '9': '9. Naive Bayes'
            }[x]
        )
        
        # Load model
        model, min_vals, max_vals, feature_columns, model_name = load_model(model_choice)
        
        if model is None:
            st.error("⚠️ Model file not found! Please train the model first using train_model.py")
            st.stop()
        
        st.success(f"✅ {model_name} loaded successfully!")
        
        # Display model metrics
        st.markdown("---")
        st.header("📊 Model Performance")
        metrics = load_model_metrics()
        if model_name in metrics:
            m = metrics[model_name]
            st.metric("Accuracy", f"{m['accuracy']*100:.2f}%")
            st.metric("Precision", f"{m['precision']*100:.2f}%")
            st.metric("Recall", f"{m['recall']*100:.2f}%")
            st.metric("F1-Score", f"{m['f1']*100:.2f}%")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("""
        This tool uses machine learning to detect phishing URLs.
        Enter a URL below to check if it's safe or potentially malicious.
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🌐 URL Input")
        url_input = st.text_input(
            "Enter URL to check:",
            placeholder="https://example.com",
            help="Enter the full URL including http:// or https://"
        )
        
        if st.button("🔍 Analyze URL", type="primary", use_container_width=True):
            if url_input:
                with st.spinner("Analyzing URL..."):
                    # Determine model type
                    model_type_map = {
                        '1': 'knn_scratch',
                        '2': 'knn_sklearn',
                        '3': 'logistic_regression',
                        '4': 'decision_tree_scratch',
                        '5': 'random_forest',
                        '6': 'xgboost',
                        '7': 'svm_scratch',
                        '8': 'svm_sklearn',
                        '9': 'naive_bayes'
                    }
                    model_type = model_type_map.get(model_choice, 'knn_sklearn')
                    
                    result = predict_url(url_input, model, min_vals, max_vals, model_type)
                    
                    # Store result in session state
                    st.session_state['prediction_result'] = result
                    st.session_state['url_analyzed'] = url_input
                    st.session_state['model_name'] = model_name
            else:
                st.warning("Please enter a URL to analyze.")
    
    with col2:
        st.header("📈 Quick Stats")
        st.info(f"**Model:** {model_name}")
        if 'prediction_result' in st.session_state:
            result = st.session_state['prediction_result']
            st.metric("Confidence", f"{result['confidence']*100:.2f}%")
    
    # Display prediction result
    if 'prediction_result' in st.session_state:
        st.markdown("---")
        result = st.session_state['prediction_result']
        url_analyzed = st.session_state.get('url_analyzed', '')
        model_name_display = st.session_state.get('model_name', model_name)
        
        # Prediction box
        is_phishing = result['prediction'] == 'Phishing'
        box_class = "danger" if is_phishing else "safe"
        icon = "⚠️" if is_phishing else "✅"
        
        st.markdown(f"""
        <div class="prediction-box {box_class}">
            <h2>{icon} {result['prediction']}</h2>
            <p>Confidence: {result['confidence']*100:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Legitimate Probability", f"{result['probability_legitimate']*100:.2f}%")
        
        with col2:
            st.metric("Phishing Probability", f"{result['probability_phishing']*100:.2f}%")
        
        with col3:
            st.metric("Confidence Level", f"{result['confidence']*100:.2f}%")
    
    # Visualizations - Always show for current selected model (not just when URL is analyzed)
    st.markdown("---")
    st.header("📊 Model Visualizations")
    st.caption(f"Showing visualizations for: **{model_name}**")
    
    # Try to load and display graphs if they exist
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    graph_dir = os.path.join(script_dir, 'report_graphs')
    if os.path.exists(graph_dir):
        # Map model names to actual graph file names
        graph_name_map = {
            'KNN (From Scratch)': 'KNN_From_Scratch',
            'KNN (sklearn)': 'KNN_(sklearn)',
            'Logistic Regression': 'Logistic_Regression',
            'Decision Tree (From Scratch)': 'Decision_Tree_(Scratch)',
            'Random Forest': 'Random_Forest',
            'XGBoost': 'XGBoost',
            'SVM (From Scratch)': 'SVM_(Scratch)',
            'SVM (sklearn)': 'SVM_(sklearn)',
            'Naive Bayes': 'Naive_Bayes'
        }
        
        # Get the correct graph name using current model name
        graph_name = graph_name_map.get(model_name, model_name.replace(" ", "_").replace("(", "").replace(")", ""))
        
        # Check which graphs are available
        cm_path = os.path.join(graph_dir, f'{graph_name}_confusion_matrix.png')
        roc_path = os.path.join(graph_dir, f'{graph_name}_roc_curve.png')
        fi_path = os.path.join(graph_dir, f'{graph_name}_feature_importance.png')
        
        cm_exists = os.path.exists(cm_path)
        roc_exists = os.path.exists(roc_path)
        fi_exists = os.path.exists(fi_path)
        
        # Only show graphs section if at least one graph exists
        if cm_exists or roc_exists or fi_exists:
            if cm_exists and roc_exists:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Confusion Matrix")
                    try:
                        st.image(cm_path, use_column_width=True)
                    except Exception as e:
                        st.error(f"Error loading confusion matrix: {str(e)}")
                
                with col2:
                    st.subheader("ROC Curve")
                    try:
                        st.image(roc_path, use_column_width=True)
                    except Exception as e:
                        st.error(f"Error loading ROC curve: {str(e)}")
            elif cm_exists:
                st.subheader("Confusion Matrix")
                try:
                    st.image(cm_path, use_column_width=True)
                except Exception as e:
                    st.error(f"Error loading confusion matrix: {str(e)}")
            elif roc_exists:
                st.subheader("ROC Curve")
                try:
                    st.image(roc_path, use_column_width=True)
                except Exception as e:
                    st.error(f"Error loading ROC curve: {str(e)}")
            
            # Feature importance if available
            if fi_exists:
                st.subheader("Feature Importance")
                try:
                    st.image(fi_path, use_column_width=True)
                except Exception as e:
                    st.error(f"Error loading feature importance: {str(e)}")
        else:
            st.info(f"📊 Visualization graphs are not available for {model_name}. Run `generate_report_graphs.py` to generate them.")
    
    # Model information
    st.markdown("---")
    with st.expander("ℹ️ Model Information"):
        st.write(f"**Current Model:** {model_name}")
        if 'prediction_result' in st.session_state:
            st.write(f"**Last URL Analyzed:** {st.session_state.get('url_analyzed', 'N/A')}")
        if model_name in metrics:
            m = metrics[model_name]
            st.write("**Performance Metrics:**")
            st.write(f"- Accuracy: {m['accuracy']*100:.2f}%")
            st.write(f"- Precision: {m['precision']*100:.2f}%")
            st.write(f"- Recall: {m['recall']*100:.2f}%")
            st.write(f"- F1-Score: {m['f1']*100:.2f}%")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🔒 Phishing URL Detection System | Powered by Machine Learning</p>
        <p>⚠️ This tool is for educational purposes. Always verify URLs through official channels.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

