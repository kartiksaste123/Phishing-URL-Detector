"""
Train Multiple Models for Phishing URL Detection
This script allows you to train different models: KNN (from scratch), KNN (sklearn), and Logistic Regression.
"""

import numpy as np
import pandas as pd
from collections import Counter
import math
import pickle
import os

# Sklearn imports
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler

# XGBoost import
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")


class KNNClassifier:
    """
    K-Nearest Neighbors Classifier implemented from scratch.
    """
    
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def euclidean_distance(self, point1, point2):
        """Calculate Euclidean distance between two points."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(point1, point2)))
    
    def fit(self, X, y):
        """Store training data (KNN is a lazy learner - no actual training)."""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        print(f"Training data stored: {len(self.X_train)} samples with {self.X_train.shape[1]} features")
    
    def predict_single(self, x):
        """Predict class for a single sample."""
        distances = []
        for i in range(len(self.X_train)):
            dist = self.euclidean_distance(x, self.X_train[i])
            distances.append((dist, self.y_train[i]))
        
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:self.k]
        k_nearest_labels = [label for _, label in k_nearest]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
    def predict(self, X):
        """Predict classes for multiple samples."""
        X = np.array(X)
        predictions = []
        for i, sample in enumerate(X):
            pred = self.predict_single(sample)
            predictions.append(pred)
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(X)} samples...")
        return np.array(predictions)
    
    def predict_proba(self, X):
        """Predict class probabilities for multiple samples."""
        X = np.array(X)
        probabilities = []
        
        for sample in X:
            distances = []
            for i in range(len(self.X_train)):
                dist = self.euclidean_distance(sample, self.X_train[i])
                distances.append((dist, self.y_train[i]))
            
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:self.k]
            k_nearest_labels = [label for _, label in k_nearest]
            
            label_counts = Counter(k_nearest_labels)
            total = len(k_nearest_labels)
            
            prob_class_0 = label_counts.get(0, 0) / total
            prob_class_1 = label_counts.get(1, 0) / total
            
            probabilities.append([prob_class_0, prob_class_1])
        
        return np.array(probabilities)


class SVMClassifier:
    """
    Support Vector Machine Classifier implemented from scratch.
    Simplified version using gradient descent for faster training.
    Uses linear kernel for speed.
    """
    
    def __init__(self, learning_rate=0.01, max_iter=1000, C=1.0):
        """
        Initialize SVM Classifier.
        
        Parameters:
        -----------
        learning_rate : float, default=0.01
            Learning rate for gradient descent
        max_iter : int, default=1000
            Maximum iterations (reduced for faster training)
        C : float, default=1.0
            Regularization parameter
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.C = C
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        """Train SVM using gradient descent (simplified and optimized for speed)."""
        print(f"Training SVM (max_iter={self.max_iter}, learning_rate={self.learning_rate})...")
        
        # Convert labels to -1 and 1 for SVM
        y_svm = np.where(y == 0, -1, 1)
        n_samples, n_features = X.shape
        
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        print("Running gradient descent optimization...")
        print("(Processing samples in batches for faster training)")
        
        # Use batch processing for faster training
        batch_size = min(100, n_samples)
        
        for iteration in range(self.max_iter):
            # Process in batches
            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)
                batch_X = X[batch_start:batch_end]
                batch_y = y_svm[batch_start:batch_end]
                
                # Vectorized computation for batch
                decisions = np.dot(batch_X, self.weights) + self.bias
                margins = batch_y * decisions
                
                # Update for misclassified samples
                misclassified = margins < 1
                if np.any(misclassified):
                    misclassified_X = batch_X[misclassified]
                    misclassified_y = batch_y[misclassified]
                    
                    # Update weights
                    weight_update = np.mean(misclassified_y[:, np.newaxis] * misclassified_X, axis=0)
                    self.weights += self.learning_rate * (self.C * weight_update - 2 * self.C * self.weights)
                    
                    # Update bias
                    self.bias += self.learning_rate * self.C * np.mean(misclassified_y)
                else:
                    # Only regularization update
                    self.weights -= self.learning_rate * (2 * self.C * self.weights)
            
            if (iteration + 1) % 100 == 0:
                print(f"  Iteration {iteration + 1}/{self.max_iter}...")
        
        print("✓ SVM training completed")
    
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
                probabilities.append([0.9, 0.1])  # High confidence for legitimate
            else:
                probabilities.append([0.1, 0.9])  # High confidence for phishing
        return np.array(probabilities)


class DecisionTreeClassifier:
    """
    Decision Tree Classifier implemented from scratch.
    Uses Gini impurity for splitting.
    Optimized for faster training with reduced complexity.
    """
    
    def __init__(self, max_depth=5, min_samples_split=20):
        """
        Initialize Decision Tree.
        
        Parameters:
        -----------
        max_depth : int, default=5
            Maximum depth of the tree (reduced for faster training)
        min_samples_split : int, default=20
            Minimum samples required to split a node (increased for faster training)
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
    
    def gini_impurity(self, groups, classes):
        """Calculate Gini impurity for a split."""
        total = sum(len(group) for group in groups)
        if total == 0:
            return 0
        
        gini = 0.0
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            score = 0.0
            for class_val in classes:
                p = sum(1 for row in group if row[-1] == class_val) / size
                score += p * p
            gini += (1.0 - score) * (size / total)
        return gini
    
    def split_dataset(self, index, value, dataset):
        """Split a dataset based on an attribute and attribute value."""
        left, right = [], []
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right
    
    def get_best_split(self, dataset, show_progress=False):
        """Select the best split point for a dataset (optimized for speed)."""
        class_values = list(set(row[-1] for row in dataset))
        best_index, best_value, best_score, best_groups = None, None, float('inf'), None
        n_features = len(dataset[0]) - 1
        
        # Optimization: Sample fewer rows for faster training
        # Instead of checking every row, sample up to 100 unique values per feature
        dataset_size = len(dataset)
        sample_size = min(100, dataset_size)  # Sample at most 100 values per feature
        
        total_splits = n_features * sample_size
        processed = 0
        
        for index in range(n_features):
            # Get unique values for this feature (sorted)
            feature_values = sorted(set(row[index] for row in dataset))
            
            # Sample values if too many
            if len(feature_values) > sample_size:
                step = len(feature_values) // sample_size
                feature_values = feature_values[::step]
            
            for val in feature_values:
                groups = self.split_dataset(index, val, dataset)
                gini = self.gini_impurity(groups, class_values)
                if gini < best_score:
                    best_index, best_value, best_score, best_groups = index, val, gini, groups
                processed += 1
                if show_progress and processed % 500 == 0:
                    print(f"    Evaluating splits: {processed}/{total_splits} ({100*processed/total_splits:.1f}%)")
        
        if show_progress:
            print(f"    Best split found: Feature {best_index}, Value {best_value:.4f}, Gini {best_score:.4f}")
        return {'index': best_index, 'value': best_value, 'groups': best_groups}
    
    def to_terminal(self, group):
        """Create a terminal node value (majority class)."""
        outcomes = [row[-1] for row in group]
        return Counter(outcomes).most_common(1)[0][0]
    
    def split(self, node, depth, show_progress=False):
        """Create child splits for a node or make terminal."""
        left, right = node['groups']
        del node['groups']
        
        if show_progress:
            if depth == 1:
                print(f"  Building level {depth}...")
            elif depth <= 3:
                print(f"  Building level {depth} (left: {len(left)} samples, right: {len(right)} samples)...")
        
        # Check for a no split
        if not left or not right:
            node['left'] = node['right'] = self.to_terminal(left + right)
            return
        
        # Check for max depth
        if depth >= self.max_depth:
            node['left'] = self.to_terminal(left)
            node['right'] = self.to_terminal(right)
            if show_progress and depth == self.max_depth:
                print(f"  Reached max_depth={self.max_depth}, creating terminal nodes...")
            return
        
        # Process left child
        if len(left) <= self.min_samples_split:
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.get_best_split(left, show_progress=False)
            self.split(node['left'], depth + 1, show_progress=show_progress)
        
        # Process right child
        if len(right) <= self.min_samples_split:
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.get_best_split(right, show_progress=False)
            self.split(node['right'], depth + 1, show_progress=show_progress)
    
    def fit(self, X, y):
        """Build decision tree from training set."""
        print(f"Building Decision Tree (max_depth={self.max_depth}, min_samples_split={self.min_samples_split})...")
        print("Combining features and labels...")
        
        # Combine X and y into dataset format
        dataset = []
        X_arr = np.array(X)
        y_arr = np.array(y)
        total_samples = len(X_arr)
        
        for i in range(len(X_arr)):
            dataset.append(list(X_arr[i]) + [int(y_arr[i])])
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{total_samples} samples...")
        
        print(f"Dataset prepared: {len(dataset)} samples")
        print("Finding best root split (this may take a moment)...")
        root = self.get_best_split(dataset, show_progress=True)
        print("Root split found. Building tree structure...")
        
        self.split(root, 1, show_progress=True)
        self.tree = root
        print(f"✓ Decision Tree built successfully with max_depth={self.max_depth}")
    
    def predict_single(self, node, row):
        """Make a prediction with a decision tree."""
        if isinstance(node, dict):
            if row[node['index']] < node['value']:
                return self.predict_single(node['left'], row)
            else:
                return self.predict_single(node['right'], row)
        else:
            return node
    
    def predict(self, X):
        """Make predictions for multiple samples."""
        X_arr = np.array(X)
        predictions = []
        for row in X_arr:
            prediction = self.predict_single(self.tree, row)
            predictions.append(prediction)
        return np.array(predictions)
    
    def predict_proba(self, X):
        """Predict class probabilities (simplified - returns 1.0 for predicted class)."""
        predictions = self.predict(X)
        probabilities = []
        for pred in predictions:
            if pred == 0:
                probabilities.append([1.0, 0.0])
            else:
                probabilities.append([0.0, 1.0])
        return np.array(probabilities)


def normalize_features(X_train, X_test):
    """Normalize features using min-max scaling."""
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    
    min_vals = X_train.min(axis=0)
    max_vals = X_train.max(axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    
    X_train_norm = (X_train - min_vals) / range_vals
    X_test_norm = (X_test - min_vals) / range_vals
    
    return X_train_norm, X_test_norm, min_vals, max_vals


def calculate_accuracy(y_true, y_pred):
    """Calculate classification accuracy."""
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)


def calculate_confusion_matrix(y_true, y_pred):
    """Calculate confusion matrix."""
    tp = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
    tn = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 0)
    fp = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)
    fn = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)
    
    return {
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'Recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'F1-Score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    }


def save_model(model, min_vals, max_vals, feature_columns, model_type, filename=None):
    """Save the trained model and normalization parameters."""
    if filename is None:
        filename = f'{model_type}_model.pkl'
    
    model_data = {
        'model': model,
        'min_vals': min_vals,
        'max_vals': max_vals,
        'feature_columns': feature_columns,
        'model_type': model_type
    }
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Model saved to {filename}")


def train_knn_scratch(X_train, X_test, y_train, y_test, X_full, y_full, min_vals, max_vals, feature_columns):
    """Train KNN model from scratch."""
    print("\n" + "=" * 60)
    print("Training KNN Classifier (From Scratch)")
    print("=" * 60)
    
    k = 5
    knn = KNNClassifier(k=k)
    knn.fit(X_train, y_train)
    print(f"KNN model initialized with k={k}")
    
    # Make predictions on test set
    print("\nMaking predictions on test set...")
    print("This may take a while as we're computing distances for each test sample...")
    y_pred = knn.predict(X_test)
    
    # Calculate metrics
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    accuracy = calculate_accuracy(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    cm = calculate_confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  True Positives (TP):  {cm['TP']}")
    print(f"  True Negatives (TN):  {cm['TN']}")
    print(f"  False Positives (FP): {cm['FP']}")
    print(f"  False Negatives (FN): {cm['FN']}")
    
    print(f"\nClassification Metrics:")
    print(f"  Precision: {cm['Precision']:.4f}")
    print(f"  Recall:    {cm['Recall']:.4f}")
    print(f"  F1-Score:  {cm['F1-Score']:.4f}")
    
    # Train on full dataset
    print("\nTraining on full dataset for production model...")
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    X_full_norm = (X_full - min_vals) / range_vals
    knn_full = KNNClassifier(k=k)
    knn_full.fit(X_full_norm, y_full)
    
    # Save model
    save_model(knn_full, min_vals, max_vals, feature_columns, 'knn_scratch')
    
    return knn_full


def train_knn_sklearn(X_train, X_test, y_train, y_test, X_full, y_full, min_vals, max_vals, feature_columns):
    """Train KNN model using sklearn."""
    print("\n" + "=" * 60)
    print("Training KNN Classifier (sklearn)")
    print("=" * 60)
    
    k = 5
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    print(f"KNN model trained with k={k}")
    
    # Make predictions on test set
    print("\nMaking predictions on test set...")
    y_pred = knn.predict(X_test)
    
    # Calculate metrics
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    accuracy = calculate_accuracy(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    cm = calculate_confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  True Positives (TP):  {cm['TP']}")
    print(f"  True Negatives (TN):  {cm['TN']}")
    print(f"  False Positives (FP): {cm['FP']}")
    print(f"  False Negatives (FN): {cm['FN']}")
    
    print(f"\nClassification Metrics:")
    print(f"  Precision: {cm['Precision']:.4f}")
    print(f"  Recall:    {cm['Recall']:.4f}")
    print(f"  F1-Score:  {cm['F1-Score']:.4f}")
    
    # Train on full dataset
    print("\nTraining on full dataset for production model...")
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    X_full_norm = (X_full - min_vals) / range_vals
    knn_full = KNeighborsClassifier(n_neighbors=k)
    knn_full.fit(X_full_norm, y_full)
    
    # Save model
    save_model(knn_full, min_vals, max_vals, feature_columns, 'knn_sklearn')
    
    return knn_full


def train_logistic_regression(X_train, X_test, y_train, y_test, X_full, y_full, min_vals, max_vals, feature_columns):
    """Train Logistic Regression model using sklearn."""
    print("\n" + "=" * 60)
    print("Training Logistic Regression (sklearn)")
    print("=" * 60)
    
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    print("Logistic Regression model trained")
    
    # Make predictions on test set
    print("\nMaking predictions on test set...")
    y_pred = lr.predict(X_test)
    
    # Calculate metrics
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    accuracy = calculate_accuracy(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    cm = calculate_confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  True Positives (TP):  {cm['TP']}")
    print(f"  True Negatives (TN):  {cm['TN']}")
    print(f"  False Positives (FP): {cm['FP']}")
    print(f"  False Negatives (FN): {cm['FN']}")
    
    print(f"\nClassification Metrics:")
    print(f"  Precision: {cm['Precision']:.4f}")
    print(f"  Recall:    {cm['Recall']:.4f}")
    print(f"  F1-Score:  {cm['F1-Score']:.4f}")
    
    # Train on full dataset
    print("\nTraining on full dataset for production model...")
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    X_full_norm = (X_full - min_vals) / range_vals
    lr_full = LogisticRegression(max_iter=1000, random_state=42)
    lr_full.fit(X_full_norm, y_full)
    
    # Save model
    save_model(lr_full, min_vals, max_vals, feature_columns, 'logistic_regression')
    
    return lr_full


def train_decision_tree_scratch(X_train, X_test, y_train, y_test, X_full, y_full, min_vals, max_vals, feature_columns):
    """Train Decision Tree model from scratch (optimized for faster training)."""
    print("\n" + "=" * 60)
    print("Training Decision Tree Classifier (From Scratch)")
    print("=" * 60)
    print(f"Training samples: {len(X_train)}")
    print(f"Features: {X_train.shape[1]}")
    print("Initializing Decision Tree (optimized for speed)...")
    print("Note: Using reduced depth and optimized split finding for faster training")
    
    # Reduced parameters for faster training
    dt = DecisionTreeClassifier(max_depth=5, min_samples_split=20)
    print("\n[Step 1/2] Training Decision Tree on training set...")
    print("(This may take 1-3 minutes depending on your system)")
    dt.fit(X_train, y_train)
    print("\n✓ Decision Tree model trained on training set")
    
    # Make predictions on test set
    print("\n[Step 2/2] Making predictions on test set...")
    print(f"Evaluating on {len(X_test)} test samples...")
    y_pred = dt.predict(X_test)
    print("✓ Predictions completed")
    
    # Calculate metrics
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    accuracy = calculate_accuracy(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    cm = calculate_confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  True Positives (TP):  {cm['TP']}")
    print(f"  True Negatives (TN):  {cm['TN']}")
    print(f"  False Positives (FP): {cm['FP']}")
    print(f"  False Negatives (FN): {cm['FN']}")
    
    print(f"\nClassification Metrics:")
    print(f"  Precision: {cm['Precision']:.4f}")
    print(f"  Recall:    {cm['Recall']:.4f}")
    print(f"  F1-Score:  {cm['F1-Score']:.4f}")
    
    # Train on full dataset
    print("\n" + "=" * 60)
    print("Training on full dataset for production model...")
    print("=" * 60)
    print(f"Full dataset samples: {len(X_full)}")
    print("(This may take 2-5 minutes depending on your system)")
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    X_full_norm = (X_full - min_vals) / range_vals
    dt_full = DecisionTreeClassifier(max_depth=5, min_samples_split=20)
    print("Building production Decision Tree...")
    dt_full.fit(X_full_norm, y_full)
    print("✓ Production model trained")
    
    # Save model
    save_model(dt_full, min_vals, max_vals, feature_columns, 'decision_tree_scratch')
    
    return dt_full


def train_random_forest(X_train, X_test, y_train, y_test, X_full, y_full, min_vals, max_vals, feature_columns):
    """Train Random Forest model using sklearn."""
    print("\n" + "=" * 60)
    print("Training Random Forest (sklearn)")
    print("=" * 60)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X_train, y_train)
    print("Random Forest model trained")
    
    # Make predictions on test set
    print("\nMaking predictions on test set...")
    y_pred = rf.predict(X_test)
    
    # Calculate metrics
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    accuracy = calculate_accuracy(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    cm = calculate_confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  True Positives (TP):  {cm['TP']}")
    print(f"  True Negatives (TN):  {cm['TN']}")
    print(f"  False Positives (FP): {cm['FP']}")
    print(f"  False Negatives (FN): {cm['FN']}")
    
    print(f"\nClassification Metrics:")
    print(f"  Precision: {cm['Precision']:.4f}")
    print(f"  Recall:    {cm['Recall']:.4f}")
    print(f"  F1-Score:  {cm['F1-Score']:.4f}")
    
    # Train on full dataset
    print("\nTraining on full dataset for production model...")
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    X_full_norm = (X_full - min_vals) / range_vals
    rf_full = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_full.fit(X_full_norm, y_full)
    
    # Save model
    save_model(rf_full, min_vals, max_vals, feature_columns, 'random_forest')
    
    return rf_full


def train_xgboost(X_train, X_test, y_train, y_test, X_full, y_full, min_vals, max_vals, feature_columns):
    """Train XGBoost model."""
    if not XGBOOST_AVAILABLE:
        print("Error: XGBoost is not installed. Please install it with: pip install xgboost")
        return None
    
    print("\n" + "=" * 60)
    print("Training XGBoost")
    print("=" * 60)
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    xgb_model.fit(X_train, y_train)
    print("XGBoost model trained")
    
    # Make predictions on test set
    print("\nMaking predictions on test set...")
    y_pred = xgb_model.predict(X_test)
    
    # Calculate metrics
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    accuracy = calculate_accuracy(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    cm = calculate_confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  True Positives (TP):  {cm['TP']}")
    print(f"  True Negatives (TN):  {cm['TN']}")
    print(f"  False Positives (FP): {cm['FP']}")
    print(f"  False Negatives (FN): {cm['FN']}")
    
    print(f"\nClassification Metrics:")
    print(f"  Precision: {cm['Precision']:.4f}")
    print(f"  Recall:    {cm['Recall']:.4f}")
    print(f"  F1-Score:  {cm['F1-Score']:.4f}")
    
    # Train on full dataset
    print("\nTraining on full dataset for production model...")
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    X_full_norm = (X_full - min_vals) / range_vals
    xgb_full = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    xgb_full.fit(X_full_norm, y_full)
    
    # Save model
    save_model(xgb_full, min_vals, max_vals, feature_columns, 'xgboost')
    
    return xgb_full


def train_svm_scratch(X_train, X_test, y_train, y_test, X_full, y_full, min_vals, max_vals, feature_columns):
    """Train SVM model from scratch (optimized for faster training)."""
    print("\n" + "=" * 60)
    print("Training SVM Classifier (From Scratch)")
    print("=" * 60)
    print(f"Training samples: {len(X_train)}")
    print(f"Features: {X_train.shape[1]}")
    print("Note: Using simplified SVM with reduced iterations for faster training")
    
    # Reduced iterations for faster training
    svm = SVMClassifier(learning_rate=0.01, max_iter=500, C=1.0)
    print("\n[Step 1/2] Training SVM on training set...")
    print("(This may take 2-5 minutes depending on your system)")
    svm.fit(X_train, y_train)
    print("\n✓ SVM model trained on training set")
    
    # Make predictions on test set
    print("\n[Step 2/2] Making predictions on test set...")
    print(f"Evaluating on {len(X_test)} test samples...")
    y_pred = svm.predict(X_test)
    print("✓ Predictions completed")
    
    # Calculate metrics
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    accuracy = calculate_accuracy(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    cm = calculate_confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  True Positives (TP):  {cm['TP']}")
    print(f"  True Negatives (TN):  {cm['TN']}")
    print(f"  False Positives (FP): {cm['FP']}")
    print(f"  False Negatives (FN): {cm['FN']}")
    
    print(f"\nClassification Metrics:")
    print(f"  Precision: {cm['Precision']:.4f}")
    print(f"  Recall:    {cm['Recall']:.4f}")
    print(f"  F1-Score:  {cm['F1-Score']:.4f}")
    
    # Train on full dataset
    print("\n" + "=" * 60)
    print("Training on full dataset for production model...")
    print("=" * 60)
    print(f"Full dataset samples: {len(X_full)}")
    print("(This may take 3-7 minutes depending on your system)")
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    X_full_norm = (X_full - min_vals) / range_vals
    svm_full = SVMClassifier(learning_rate=0.01, max_iter=500, C=1.0)
    print("Building production SVM...")
    svm_full.fit(X_full_norm, y_full)
    print("✓ Production model trained")
    
    # Save model
    save_model(svm_full, min_vals, max_vals, feature_columns, 'svm_scratch')
    
    return svm_full


def train_svm_sklearn(X_train, X_test, y_train, y_test, X_full, y_full, min_vals, max_vals, feature_columns):
    """Train SVM model using sklearn."""
    print("\n" + "=" * 60)
    print("Training SVM (sklearn)")
    print("=" * 60)
    
    # Using linear kernel for faster training
    svm = SVC(kernel='linear', C=1.0, random_state=42, probability=True)
    svm.fit(X_train, y_train)
    print("SVM model trained")
    
    # Make predictions on test set
    print("\nMaking predictions on test set...")
    y_pred = svm.predict(X_test)
    
    # Calculate metrics
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    accuracy = calculate_accuracy(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    cm = calculate_confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  True Positives (TP):  {cm['TP']}")
    print(f"  True Negatives (TN):  {cm['TN']}")
    print(f"  False Positives (FP): {cm['FP']}")
    print(f"  False Negatives (FN): {cm['FN']}")
    
    print(f"\nClassification Metrics:")
    print(f"  Precision: {cm['Precision']:.4f}")
    print(f"  Recall:    {cm['Recall']:.4f}")
    print(f"  F1-Score:  {cm['F1-Score']:.4f}")
    
    # Train on full dataset
    print("\nTraining on full dataset for production model...")
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    X_full_norm = (X_full - min_vals) / range_vals
    svm_full = SVC(kernel='linear', C=1.0, random_state=42, probability=True)
    svm_full.fit(X_full_norm, y_full)
    
    # Save model
    save_model(svm_full, min_vals, max_vals, feature_columns, 'svm_sklearn')
    
    return svm_full


def train_naive_bayes(X_train, X_test, y_train, y_test, X_full, y_full, min_vals, max_vals, feature_columns):
    """Train Naive Bayes model using sklearn."""
    print("\n" + "=" * 60)
    print("Training Naive Bayes (sklearn)")
    print("=" * 60)
    
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    print("Naive Bayes model trained")
    
    # Make predictions on test set
    print("\nMaking predictions on test set...")
    y_pred = nb.predict(X_test)
    
    # Calculate metrics
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    accuracy = calculate_accuracy(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    cm = calculate_confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  True Positives (TP):  {cm['TP']}")
    print(f"  True Negatives (TN):  {cm['TN']}")
    print(f"  False Positives (FP): {cm['FP']}")
    print(f"  False Negatives (FN): {cm['FN']}")
    
    print(f"\nClassification Metrics:")
    print(f"  Precision: {cm['Precision']:.4f}")
    print(f"  Recall:    {cm['Recall']:.4f}")
    print(f"  F1-Score:  {cm['F1-Score']:.4f}")
    
    # Train on full dataset
    print("\nTraining on full dataset for production model...")
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    X_full_norm = (X_full - min_vals) / range_vals
    nb_full = GaussianNB()
    nb_full.fit(X_full_norm, y_full)
    
    # Save model
    save_model(nb_full, min_vals, max_vals, feature_columns, 'naive_bayes')
    
    return nb_full


def load_and_prepare_data():
    """Load and prepare the dataset."""
    print("\n[1/3] Loading dataset...")
    # Try multiple possible paths for the dataset
    possible_paths = [
        'Phishing_Legitimate_cleaned.csv',
        '../Phishing_Legitimate_cleaned.csv',
        r'D:\CODING\Github Projects\AIML\Phishing_Legitimate_cleaned.csv'
    ]
    csv_path = None
    for path in possible_paths:
        if os.path.exists(path):
            csv_path = path
            break
    if csv_path is None:
        raise FileNotFoundError("Could not find Phishing_Legitimate_cleaned.csv. Please ensure the file is in the current directory or parent directory.")
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
    
    # Separate features and target
    feature_columns = [col for col in df.columns if col != 'CLASS_LABEL']
    X = df[feature_columns].values
    y = df['CLASS_LABEL'].values
    
    print(f"Features: {len(feature_columns)}")
    print(f"Class distribution: {Counter(y)}")
    
    # Split into train and test sets (80-20 split)
    print("\n[2/3] Splitting dataset into train and test sets...")
    split_idx = int(0.8 * len(X))
    indices = np.random.permutation(len(X))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Normalize features
    print("\n[3/3] Normalizing features...")
    X_train_norm, X_test_norm, min_vals, max_vals = normalize_features(X_train, X_test)
    print("Features normalized using min-max scaling")
    
    return X_train_norm, X_test_norm, y_train, y_test, X, y, min_vals, max_vals, feature_columns


def show_menu():
    """Display model selection menu."""
    print("\n" + "=" * 60)
    print("SELECT MODEL TO TRAIN")
    print("=" * 60)
    print("1. KNN (From Scratch)")
    print("2. KNN (sklearn)")
    print("3. Logistic Regression (sklearn)")
    print("4. Decision Tree (From Scratch)")
    print("5. Random Forest (sklearn)")
    print("6. XGBoost")
    print("7. SVM (From Scratch)")
    print("8. SVM (sklearn)")
    print("9. Naive Bayes (sklearn)")
    print("10. Train All Models")
    print("0. Exit")
    print("=" * 60)


def main():
    """Main function to train models."""
    print("=" * 60)
    print("Phishing URL Detection - Model Training")
    print("=" * 60)
    
    # Load and prepare data once
    X_train, X_test, y_train, y_test, X_full, y_full, min_vals, max_vals, feature_columns = load_and_prepare_data()
    
    while True:
        show_menu()
        choice = input("\nEnter your choice (0-10): ").strip()
        
        if choice == '0':
            print("\nExiting...")
            break
        elif choice == '1':
            train_knn_scratch(X_train, X_test, y_train, y_test, X_full, y_full, min_vals, max_vals, feature_columns)
        elif choice == '2':
            train_knn_sklearn(X_train, X_test, y_train, y_test, X_full, y_full, min_vals, max_vals, feature_columns)
        elif choice == '3':
            train_logistic_regression(X_train, X_test, y_train, y_test, X_full, y_full, min_vals, max_vals, feature_columns)
        elif choice == '4':
            train_decision_tree_scratch(X_train, X_test, y_train, y_test, X_full, y_full, min_vals, max_vals, feature_columns)
        elif choice == '5':
            train_random_forest(X_train, X_test, y_train, y_test, X_full, y_full, min_vals, max_vals, feature_columns)
        elif choice == '6':
            train_xgboost(X_train, X_test, y_train, y_test, X_full, y_full, min_vals, max_vals, feature_columns)
        elif choice == '7':
            train_svm_scratch(X_train, X_test, y_train, y_test, X_full, y_full, min_vals, max_vals, feature_columns)
        elif choice == '8':
            train_svm_sklearn(X_train, X_test, y_train, y_test, X_full, y_full, min_vals, max_vals, feature_columns)
        elif choice == '9':
            train_naive_bayes(X_train, X_test, y_train, y_test, X_full, y_full, min_vals, max_vals, feature_columns)
        elif choice == '10':
            print("\nTraining all models...")
            train_knn_scratch(X_train, X_test, y_train, y_test, X_full, y_full, min_vals, max_vals, feature_columns)
            train_knn_sklearn(X_train, X_test, y_train, y_test, X_full, y_full, min_vals, max_vals, feature_columns)
            train_logistic_regression(X_train, X_test, y_train, y_test, X_full, y_full, min_vals, max_vals, feature_columns)
            train_decision_tree_scratch(X_train, X_test, y_train, y_test, X_full, y_full, min_vals, max_vals, feature_columns)
            train_random_forest(X_train, X_test, y_train, y_test, X_full, y_full, min_vals, max_vals, feature_columns)
            if XGBOOST_AVAILABLE:
                train_xgboost(X_train, X_test, y_train, y_test, X_full, y_full, min_vals, max_vals, feature_columns)
            train_svm_scratch(X_train, X_test, y_train, y_test, X_full, y_full, min_vals, max_vals, feature_columns)
            train_svm_sklearn(X_train, X_test, y_train, y_test, X_full, y_full, min_vals, max_vals, feature_columns)
            train_naive_bayes(X_train, X_test, y_train, y_test, X_full, y_full, min_vals, max_vals, feature_columns)
            print("\n" + "=" * 60)
            print("All models trained and saved successfully!")
            print("=" * 60)
        else:
            print("\nInvalid choice! Please enter a number between 0-10.")
        
        if choice in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']:
            continue_training = input("\nDo you want to train another model? (y/n): ").strip().lower()
            if continue_training != 'y':
                break


if __name__ == "__main__":
    np.random.seed(42)
    main()
