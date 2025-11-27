"""
Test URL for Phishing Detection
Enter any URL and get a prediction if it's phishing or legitimate.
"""

import numpy as np
import pickle
import re
import math
from urllib.parse import urlparse, parse_qs
from collections import Counter
import ipaddress
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import seaborn as sns

# Import custom classes from train_model so pickle can load the saved models
# Always define fallback classes at module level for pickle compatibility
from collections import Counter as Counter_
import math as math_

# Try to import from train_model, otherwise use fallback definitions
try:
    from train_model import KNNClassifier, DecisionTreeClassifier, SVMClassifier
except ImportError:
    # Fallback: define classes here (same as in train_model.py)
    class KNNClassifier:
        """K-Nearest Neighbors Classifier implemented from scratch."""
        
        def __init__(self, k=5):
            self.k = k
            self.X_train = None
            self.y_train = None
        
        def euclidean_distance(self, point1, point2):
            return math_.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))
        
        def fit(self, X, y):
            self.X_train = np.array(X)
            self.y_train = np.array(y)
        
        def predict_single(self, x):
            distances = []
            for i in range(len(self.X_train)):
                dist = self.euclidean_distance(x, self.X_train[i])
                distances.append((dist, self.y_train[i]))
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:self.k]
            k_nearest_labels = [label for _, label in k_nearest]
            most_common = Counter_(k_nearest_labels).most_common(1)
            return most_common[0][0]

# Always define DecisionTreeClassifier at module level
class DecisionTreeClassifier:
        """Decision Tree Classifier implemented from scratch."""
        
        def __init__(self, max_depth=5, min_samples_split=20):
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.tree = None
        
        def predict(self, X):
            """Make predictions for multiple samples."""
            X_arr = np.array(X)
            predictions = []
            for row in X_arr:
                prediction = self._predict_single(self.tree, row)
                predictions.append(prediction)
            return np.array(predictions)
        
        def _predict_single(self, node, row):
            """Make a prediction with a decision tree."""
            if isinstance(node, dict):
                if row[node['index']] < node['value']:
                    return self._predict_single(node['left'], row)
                else:
                    return self._predict_single(node['right'], row)
            else:
                return node
        
        def predict_proba(self, X):
            """Predict class probabilities (simplified)."""
            predictions = self.predict(X)
            probabilities = []
            for pred in predictions:
                if pred == 0:
                    probabilities.append([1.0, 0.0])
                else:
                    probabilities.append([0.0, 1.0])
            return np.array(probabilities)

# Always define SVMClassifier at module level
class SVMClassifier:
        """Support Vector Machine Classifier implemented from scratch."""
        
        def __init__(self, learning_rate=0.01, max_iter=500, C=1.0):
            self.learning_rate = learning_rate
            self.max_iter = max_iter
            self.C = C
            self.weights = None
            self.bias = None
        
        def predict(self, X):
            """Predict classes for samples."""
            predictions = []
            for sample in X:
                decision = np.dot(sample, self.weights) + self.bias
                predictions.append(1 if decision >= 0 else 0)
            return np.array(predictions)
        
        def predict_proba(self, X):
            """Predict class probabilities (simplified)."""
            predictions = self.predict(X)
            probabilities = []
            for pred in predictions:
                if pred == 0:
                    probabilities.append([0.9, 0.1])
                else:
                    probabilities.append([0.1, 0.9])
            return np.array(probabilities)


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
    
    def _extract_domain(self, hostname):
        if not hostname:
            return None
        parts = hostname.split('.')
        if len(parts) >= 2:
            return '.'.join(parts[-2:])
        return hostname
    
    def _extract_subdomain(self, hostname):
        if not hostname:
            return ''
        parts = hostname.split('.')
        if len(parts) > 2:
            return '.'.join(parts[:-2])
        return ''
    
    def _get_default_features(self):
        """Return default feature vector (29 features) when URL parsing fails."""
        features = [0] * 29
        features[3] = 20  # UrlLength
        features[17] = 10  # HostnameLength
        features[23] = 0.0  # PctExtHyperlinks
        features[24] = 0.0  # PctExtResourceUrls
        features[27] = 0.0  # PctNullSelfRedirectHyperlinks
        return features


def load_model(model_type='knn_scratch'):
    """Load the trained model and normalization parameters."""
    
    # Map model types to filenames
    model_files = {
        '1': 'knn_scratch_model.pkl',
        '2': 'knn_sklearn_model.pkl',
        '3': 'logistic_regression_model.pkl',
        '4': 'decision_tree_scratch_model.pkl',
        '5': 'random_forest_model.pkl',
        '6': 'xgboost_model.pkl',
        '7': 'svm_scratch_model.pkl',
        '8': 'svm_sklearn_model.pkl',
        '9': 'naive_bayes_model.pkl',
        'knn_scratch': 'knn_scratch_model.pkl',
        'knn_sklearn': 'knn_sklearn_model.pkl',
        'logistic_regression': 'logistic_regression_model.pkl',
        'decision_tree_scratch': 'decision_tree_scratch_model.pkl',
        'random_forest': 'random_forest_model.pkl',
        'xgboost': 'xgboost_model.pkl',
        'svm_scratch': 'svm_scratch_model.pkl',
        'svm_sklearn': 'svm_sklearn_model.pkl',
        'naive_bayes': 'naive_bayes_model.pkl'
    }
    
    filename = model_files.get(model_type, model_files.get('1', 'knn_scratch_model.pkl'))
    
    if not os.path.exists(filename):
        print(f"Error: Model file '{filename}' not found!")
        print("Please run 'train_model.py' first to train and save the model.")
        return None, None, None, None, None
    
    # Custom unpickler to handle class loading
    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # Map train_model classes to current module classes
            if module == 'train_model' or module == '__main__':
                if name == 'KNNClassifier':
                    return KNNClassifier
                elif name == 'DecisionTreeClassifier':
                    return DecisionTreeClassifier
                elif name == 'SVMClassifier':
                    return SVMClassifier
            return super().find_class(module, name)
    
    with open(filename, 'rb') as f:
        unpickler = CustomUnpickler(f)
        model_data = unpickler.load()
    
    model = model_data['model']
    model_type_loaded = model_data.get('model_type', 'unknown')
    print(f"Model loaded successfully! (Type: {model_type_loaded})")
    return model, model_data['min_vals'], model_data['max_vals'], model_data['feature_columns'], model_type_loaded


def predict_url(url, model, min_vals, max_vals, model_type='knn_scratch'):
    """Predict if a URL is phishing or legitimate."""
    # Extract features
    extractor = URLFeatureExtractor()
    features = extractor.extract_all_features(url)
    features = np.array(features, dtype=float)
    
    # Normalize features
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    features_norm = (features - min_vals) / range_vals
    features_norm = features_norm.reshape(1, -1)  # Reshape for sklearn models
    
    # Predict based on model type
    if model_type == 'knn_scratch':
        # Custom KNN from scratch
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
    elif model_type == 'decision_tree_scratch':
        # Decision Tree from scratch
        prediction = model.predict(features_norm)[0]
        proba = model.predict_proba(features_norm)[0]
        prob_class_0 = proba[0]
        prob_class_1 = proba[1]
    elif model_type == 'svm_scratch':
        # SVM from scratch
        prediction = model.predict(features_norm)[0]
        proba = model.predict_proba(features_norm)[0]
        prob_class_0 = proba[0]
        prob_class_1 = proba[1]
    else:
        # Sklearn models (KNN, Logistic Regression, Random Forest, XGBoost, SVM, Naive Bayes)
        prediction = model.predict(features_norm)[0]
        proba = model.predict_proba(features_norm)[0]
        prob_class_0 = proba[0]
        prob_class_1 = proba[1]
    
    result = {
        'prediction': 'Phishing' if prediction == 1 else 'Legitimate',
        'prediction_label': prediction,
        'probability_legitimate': prob_class_0,
        'probability_phishing': prob_class_1,
        'confidence': max(prob_class_0, prob_class_1)
    }
    
    return result


def show_model_menu():
    """Display model selection menu."""
    print("\n" + "=" * 70)
    print("SELECT MODEL FOR PREDICTION")
    print("=" * 70)
    print("1. KNN (From Scratch)")
    print("2. KNN (sklearn)")
    print("3. Logistic Regression (sklearn)")
    print("4. Decision Tree (From Scratch)")
    print("5. Random Forest (sklearn)")
    print("6. XGBoost")
    print("7. SVM (From Scratch)")
    print("8. SVM (sklearn)")
    print("9. Naive Bayes (sklearn)")
    print("=" * 70)


def main():
    """Main function for interactive URL testing."""
    print("=" * 70)
    print("Phishing URL Detection - Enter any URL to check")
    print("=" * 70)
    
    # Select model
    show_model_menu()
    model_choice = input("\nEnter model choice (1-9): ").strip()
    
    if model_choice not in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
        print("Invalid choice! Using KNN (From Scratch) as default.")
        model_choice = '1'
    
    # Load model
    print("\nLoading trained model...")
    model, min_vals, max_vals, feature_columns, model_type = load_model(model_choice)
    
    if model is None:
        return
    
    model_name = {
        '1': 'KNN (From Scratch)',
        '2': 'KNN (sklearn)',
        '3': 'Logistic Regression (sklearn)',
        '4': 'Decision Tree (From Scratch)',
        '5': 'Random Forest (sklearn)',
        '6': 'XGBoost',
        '7': 'SVM (From Scratch)',
        '8': 'SVM (sklearn)',
        '9': 'Naive Bayes (sklearn)'
    }.get(model_choice, 'Unknown')
    
    print(f"Model ready! ({model_name})")
    
    # Ask if user wants to generate report graphs
    print("\n" + "=" * 70)
    generate_graphs = input("Generate visualization graphs for report? (y/n): ").strip().lower()
    
    if generate_graphs == 'y':
        print("\nGenerating graphs...")
        try:
            from generate_report_graphs import generate_all_graphs
            generate_all_graphs()
        except Exception as e:
            print(f"Error generating graphs: {str(e)}")
            print("You can generate graphs later by running: python generate_report_graphs.py")
    
    # Example URLs
    print("\n" + "=" * 70)
    print("Testing with example URLs:")
    print("=" * 70)
    
    example_urls = [
        "https://youtube.com",
        "https://www.google.com",
        "http://suspicious-site.com/login/verify?account=update",
    ]
    
    for url in example_urls:
        try:
            result = predict_url(url, model, min_vals, max_vals, model_type)
            print(f"\nURL: {url}")
            print(f"  Prediction: {result['prediction']}")
            print(f"  Confidence: {result['confidence']:.2%}")
            print(f"  Prob (Legit): {result['probability_legitimate']:.4f}")
            print(f"  Prob (Phish): {result['probability_phishing']:.4f}")
        except Exception as e:
            print(f"\nError processing {url}: {str(e)}")
    
    # Interactive mode
    print("\n" + "=" * 70)
    print("Interactive Mode - Enter URLs to check (type 'quit' to exit)")
    print("=" * 70)
    
    while True:
        try:
            user_url = input("\nEnter URL (or 'quit' to exit): ").strip()
            
            if user_url.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not user_url:
                continue
            
            result = predict_url(user_url, model, min_vals, max_vals, model_type)
            
            print(f"\n{'='*70}")
            print(f"PREDICTION RESULT")
            print(f"{'='*70}")
            print(f"URL:            {user_url}")
            print(f"Model:          {model_name}")
            print(f"Prediction:     {result['prediction']}")
            print(f"Confidence:     {result['confidence']:.2%}")
            print(f"Prob (Legit):   {result['probability_legitimate']:.4f}")
            print(f"Prob (Phish):   {result['probability_phishing']:.4f}")
            
            if result['prediction'] == 'Phishing':
                print(f"\n⚠️  WARNING: This URL shows characteristics of a phishing site!")
            else:
                print(f"\n✓ This URL appears to be legitimate.")
            
            # Option to generate graphs for this model
            if user_url.lower() not in ['quit', 'exit', 'q']:
                gen_graph = input("\nGenerate evaluation graphs for this model? (y/n): ").strip().lower()
                if gen_graph == 'y':
                    try:
                        from generate_report_graphs import generate_single_model_graphs
                        generate_single_model_graphs(model, model_name, model_type, min_vals, max_vals, feature_columns)
                    except Exception as e:
                        print(f"Error generating graphs: {str(e)}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again with a valid URL.")


if __name__ == "__main__":
    main()

