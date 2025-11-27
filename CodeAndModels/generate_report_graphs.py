"""
Generate Visualization Graphs for Model Evaluation Report
This module creates various graphs and visualizations for all trained models.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import os
import pickle

# Import custom classes from train_model so pickle can load the saved models
try:
    from train_model import KNNClassifier, DecisionTreeClassifier, SVMClassifier
except ImportError:
    # Fallback definitions if import fails (for pickle loading)
    class KNNClassifier:
        pass
    class DecisionTreeClassifier:
        pass
    class SVMClassifier:
        pass

# Set style for better-looking plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('ggplot')
sns.set_palette("husl")


def load_test_data():
    """Load test data for evaluation."""
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
        raise FileNotFoundError("Could not find Phishing_Legitimate_cleaned.csv")
    df = pd.read_csv(csv_path)
    
    feature_columns = [col for col in df.columns if col != 'CLASS_LABEL']
    X = df[feature_columns].values
    y = df['CLASS_LABEL'].values
    
    # Use same split as training (80-20)
    np.random.seed(42)
    split_idx = int(0.8 * len(X))
    indices = np.random.permutation(len(X))
    test_indices = indices[split_idx:]
    
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    return X_test, y_test, feature_columns


def normalize_features(X, min_vals, max_vals):
    """Normalize features."""
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1
    return (X - min_vals) / range_vals


def load_and_evaluate_model(model_file, X_test, y_test):
    """Load model and evaluate on test set."""
    if not os.path.exists(model_file):
        return None, None, None
    
    try:
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
        
        with open(model_file, 'rb') as f:
            unpickler = CustomUnpickler(f)
            model_data = unpickler.load()
        
        # Handle both old and new model data structures
        if isinstance(model_data, dict):
            # Check for old format (uses 'knn' key) or new format (uses 'model' key)
            if 'knn' in model_data:
                # Old format
                model = model_data['knn']
                min_vals = model_data['min_vals']
                max_vals = model_data['max_vals']
                model_type = 'knn_scratch'
            elif 'model' in model_data:
                # New format
                model = model_data['model']
                min_vals = model_data['min_vals']
                max_vals = model_data['max_vals']
                model_type = model_data.get('model_type', 'unknown')
            else:
                print(f"Error: Unknown model data format in {model_file}")
                return None, None, None
        else:
            print(f"Error: Model data is not a dictionary in {model_file}")
            return None, None, None
        
        # Normalize test data
        X_test_norm = normalize_features(X_test, min_vals, max_vals)
        
        # Predict
        # Check if it's a custom KNN (has predict_single that takes only one arg)
        if hasattr(model, 'predict_single') and hasattr(model, 'X_train'):
            # Custom KNN - use predict_single for each sample
            y_pred = []
            for sample in X_test_norm:
                y_pred.append(model.predict_single(sample))
            y_pred = np.array(y_pred)
        elif hasattr(model, 'predict'):
            # Standard sklearn or custom models with predict method
            # For custom DecisionTree and SVM, they have predict method that works with arrays
            try:
                y_pred = model.predict(X_test_norm)
            except Exception as e:
                print(f"Error during prediction for {model_file}: {str(e)}")
                return None, None, None
        else:
            print(f"Warning: Model {model_file} doesn't have predict method")
            return None, None, None
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test_norm)[:, 1]
            except:
                y_proba = None
        else:
            y_proba = None
        
        return y_pred, y_proba, model_type
    except Exception as e:
        print(f"Error loading {model_file}: {str(e)}")
        return None, None, None


def plot_confusion_matrix(y_true, y_pred, model_name, save_path):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate', 'Phishing'],
                yticklabels=['Legitimate', 'Phishing'])
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_roc_curve(y_true, y_proba, model_name, save_path):
    """Plot ROC curve."""
    if y_proba is None:
        return
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_feature_importance(model, feature_columns, model_name, save_path):
    """Plot feature importance for tree-based models."""
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models, use absolute coefficients
            importances = np.abs(model.coef_[0])
        else:
            return
        
        # Get top 15 features
        indices = np.argsort(importances)[::-1][:15]
        top_features = [feature_columns[i] for i in indices]
        top_importances = importances[indices]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_importances, color='steelblue')
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Importance Score', fontsize=12)
        plt.title(f'Top 15 Feature Importances - {model_name}', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {save_path}")
    except Exception as e:
        print(f"  ⚠ Could not generate feature importance for {model_name}: {str(e)}")


def plot_model_comparison(metrics_dict, save_path):
    """Plot comparison of all models."""
    models = list(metrics_dict.keys())
    accuracies = [metrics_dict[m]['accuracy'] for m in models]
    precisions = [metrics_dict[m]['precision'] for m in models]
    recalls = [metrics_dict[m]['recall'] for m in models]
    f1_scores = [metrics_dict[m]['f1'] for m in models]
    
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(x - 1.5*width, accuracies, width, label='Accuracy', color='#3498db')
    ax.bar(x - 0.5*width, precisions, width, label='Precision', color='#2ecc71')
    ax.bar(x + 0.5*width, recalls, width, label='Recall', color='#e74c3c')
    ax.bar(x + 1.5*width, f1_scores, width, label='F1-Score', color='#f39c12')
    
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, model in enumerate(models):
        ax.text(i - 1.5*width, accuracies[i] + 0.01, f'{accuracies[i]:.2f}', 
                ha='center', va='bottom', fontsize=8)
        ax.text(i - 0.5*width, precisions[i] + 0.01, f'{precisions[i]:.2f}', 
                ha='center', va='bottom', fontsize=8)
        ax.text(i + 0.5*width, recalls[i] + 0.01, f'{recalls[i]:.2f}', 
                ha='center', va='bottom', fontsize=8)
        ax.text(i + 1.5*width, f1_scores[i] + 0.01, f'{f1_scores[i]:.2f}', 
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_accuracy_comparison(metrics_dict, save_path):
    """Plot accuracy comparison bar chart."""
    models = list(metrics_dict.keys())
    accuracies = [metrics_dict[m]['accuracy'] * 100 for m in models]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.2)
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim([0, 105])
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_precision_recall_f1_comparison(metrics_dict, save_path):
    """Plot Precision, Recall, and F1-Score comparison."""
    models = list(metrics_dict.keys())
    precisions = [metrics_dict[m]['precision'] * 100 for m in models]
    recalls = [metrics_dict[m]['recall'] * 100 for m in models]
    f1_scores = [metrics_dict[m]['f1'] * 100 for m in models]
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - width, precisions, width, label='Precision (%)', color='#3498db', edgecolor='black')
    ax.bar(x, recalls, width, label='Recall (%)', color='#2ecc71', edgecolor='black')
    ax.bar(x + width, f1_scores, width, label='F1-Score (%)', color='#e74c3c', edgecolor='black')
    
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Precision, Recall, and F1-Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=11)
    ax.set_ylim([0, 110])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def plot_confusion_matrices_grid(all_metrics, all_predictions, y_test, save_path):
    """Plot all confusion matrices in a grid."""
    n_models = len(all_metrics)
    cols = 3
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, (model_name, y_pred) in enumerate(all_predictions.items()):
        cm = confusion_matrix(y_test, y_pred)
        ax = axes[idx]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Legitimate', 'Phishing'],
                    yticklabels=['Legitimate', 'Phishing'])
        ax.set_title(f'{model_name}', fontsize=11, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=10)
        ax.set_xlabel('Predicted Label', fontsize=10)
    
    # Hide extra subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def calculate_metrics(y_true, y_pred):
    """Calculate classification metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }


def generate_all_graphs():
    """Generate all visualization graphs for report."""
    print("=" * 70)
    print("Generating Visualization Graphs for Report")
    print("=" * 70)
    
    # Create output directory
    output_dir = 'report_graphs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test data
    print("\n[1/4] Loading test data...")
    X_test, y_test, feature_columns = load_test_data()
    print(f"Test set: {len(X_test)} samples")
    
    # Model files - check both old and new formats for KNN (From Scratch)
    model_files = {
        'KNN (From Scratch)': ['knn_scratch_model.pkl', 'knn_model.pkl'],  # Check both filenames
        'KNN (sklearn)': 'knn_sklearn_model.pkl',
        'Logistic Regression': 'logistic_regression_model.pkl',
        'Decision Tree (From Scratch)': 'decision_tree_scratch_model.pkl',
        'Random Forest': 'random_forest_model.pkl',
        'XGBoost': 'xgboost_model.pkl',
        'SVM (From Scratch)': 'svm_scratch_model.pkl',
        'SVM (sklearn)': 'svm_sklearn_model.pkl',
        'Naive Bayes': 'naive_bayes_model.pkl'
    }
    
    # Custom classes are already imported at the top of the file
    
    # Evaluate all models
    print("\n[2/4] Evaluating all models...")
    all_metrics = {}
    all_predictions = {}
    all_probabilities = {}
    all_models = {}
    
    for model_name, model_file in model_files.items():
        print(f"  Evaluating {model_name}...")
        # Handle both single filename and list of filenames
        if isinstance(model_file, list):
            # Try each filename until one works
            y_pred, y_proba, model_type = None, None, None
            for filename in model_file:
                if os.path.exists(filename):
                    y_pred, y_proba, model_type = load_and_evaluate_model(filename, X_test, y_test)
                    if y_pred is not None:
                        break
        else:
            y_pred, y_proba, model_type = load_and_evaluate_model(model_file, X_test, y_test)
        
        if y_pred is not None:
            metrics = calculate_metrics(y_test, y_pred)
            all_metrics[model_name] = metrics
            all_predictions[model_name] = y_pred
            all_probabilities[model_name] = y_proba
            
            # Load model for feature importance
            try:
                # Get the actual filename that was used
                actual_file = model_file
                if isinstance(model_file, list):
                    # Find which file was actually used
                    for fname in model_file:
                        if os.path.exists(fname):
                            actual_file = fname
                            break
                
                with open(actual_file, 'rb') as f:
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
                    unpickler = CustomUnpickler(f)
                    model_data = unpickler.load()
                
                # Handle both old and new formats
                if isinstance(model_data, dict):
                    if 'knn' in model_data:
                        all_models[model_name] = model_data['knn']
                    elif 'model' in model_data:
                        all_models[model_name] = model_data['model']
            except Exception as e:
                print(f"  ⚠ Could not load model for feature importance: {str(e)}")
                pass
        else:
            print(f"  ⚠ {model_name} model file not found, skipping...")
    
    # Generate individual model graphs
    print("\n[3/4] Generating individual model visualizations...")
    for model_name in all_metrics.keys():
        # Get the model file (handle both list and single filename)
        model_file_entry = model_files.get(model_name, '')
        if isinstance(model_file_entry, list):
            model_file = model_file_entry[0]  # Use first filename for saving
        else:
            model_file = model_file_entry
        
        y_pred = all_predictions[model_name]
        y_proba = all_probabilities[model_name]
        
        # Clean model name for filename (remove parentheses and spaces)
        clean_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
        
        # Confusion Matrix
        cm_path = os.path.join(output_dir, f'{clean_name}_confusion_matrix.png')
        plot_confusion_matrix(y_test, y_pred, model_name, cm_path)
        
        # ROC Curve (if probabilities available)
        if y_proba is not None:
            roc_path = os.path.join(output_dir, f'{clean_name}_roc_curve.png')
            plot_roc_curve(y_test, y_proba, model_name, roc_path)
        
        # Feature Importance (for tree-based and linear models)
        if model_name in all_models:
            fi_path = os.path.join(output_dir, f'{model_name.replace(" ", "_").replace("(", "").replace(")", "")}_feature_importance.png')
            try:
                # Handle both old and new formats
                with open(model_file, 'rb') as f:
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
                    unpickler = CustomUnpickler(f)
                    model_data = unpickler.load()
                
                # Get model from data
                if isinstance(model_data, dict):
                    if 'knn' in model_data:
                        model_obj = model_data['knn']
                    elif 'model' in model_data:
                        model_obj = model_data['model']
                    else:
                        model_obj = all_models[model_name]
                else:
                    model_obj = all_models[model_name]
                
                plot_feature_importance(model_obj, feature_columns, model_name, fi_path)
            except Exception as e:
                print(f"  ⚠ Could not generate feature importance for {model_name}: {str(e)}")
    
    # Generate comparison graphs
    print("\n[4/4] Generating comparison visualizations...")
    
    # Model Comparison (all metrics)
    comparison_path = os.path.join(output_dir, 'model_comparison_all_metrics.png')
    plot_model_comparison(all_metrics, comparison_path)
    
    # Accuracy Comparison
    accuracy_path = os.path.join(output_dir, 'model_accuracy_comparison.png')
    plot_accuracy_comparison(all_metrics, accuracy_path)
    
    # Precision, Recall, F1 Comparison
    prf_path = os.path.join(output_dir, 'precision_recall_f1_comparison.png')
    plot_precision_recall_f1_comparison(all_metrics, prf_path)
    
    # Confusion Matrices Grid
    cm_grid_path = os.path.join(output_dir, 'confusion_matrices_grid.png')
    plot_confusion_matrices_grid(all_metrics, all_predictions, y_test, cm_grid_path)
    
    # Create summary table
    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 70)
    summary_data = []
    for model_name, metrics in all_metrics.items():
        summary_data.append({
            'Model': model_name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1']:.4f}"
        })
    
    df_summary = pd.DataFrame(summary_data)
    print("\n" + df_summary.to_string(index=False))
    
    # Save summary to CSV
    summary_path = os.path.join(output_dir, 'model_performance_summary.csv')
    df_summary.to_csv(summary_path, index=False)
    print(f"\n  ✓ Saved summary table: {summary_path}")
    
    print("\n" + "=" * 70)
    print(f"All graphs saved to '{output_dir}' directory!")
    print("=" * 70)
    
    num_roc = len([m for m in all_probabilities.values() if m is not None])
    num_fi = len([m for m in all_models.keys() if m in ['Random Forest', 'XGBoost', 'Decision Tree (Scratch)', 'Logistic Regression']])
    
    total_files = len(all_metrics) + num_roc + num_fi + 4  # Individual CMs + ROCs + FIs + 4 comparison charts
    
    print(f"\nGenerated {total_files} visualization files:")
    print(f"  - {len(all_metrics)} Individual Confusion Matrices")
    print(f"  - {num_roc} ROC Curves")
    print(f"  - {num_fi} Feature Importance plots")
    print(f"  - 1 Confusion Matrices Grid (all models)")
    print(f"  - 1 Model Comparison Chart (all metrics)")
    print(f"  - 1 Accuracy Comparison Chart")
    print(f"  - 1 Precision/Recall/F1 Comparison Chart")
    print(f"  - 1 Performance Summary CSV")
    print(f"\nAll files are saved in: {os.path.abspath(output_dir)}")


def generate_single_model_graphs(model, model_name, model_type, min_vals, max_vals, feature_columns):
    """Generate graphs for a single model."""
    print("\n" + "=" * 70)
    print(f"Generating Graphs for {model_name}")
    print("=" * 70)
    
    # Create output directory
    output_dir = 'report_graphs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test data
    X_test, y_test, _ = load_test_data()
    X_test_norm = normalize_features(X_test, min_vals, max_vals)
    
    # Predict
    if hasattr(model, 'predict_single'):
        y_pred = []
        for sample in X_test_norm:
            y_pred.append(model.predict_single(sample))
        y_pred = np.array(y_pred)
    else:
        y_pred = model.predict(X_test_norm)
    
    # Get probabilities
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test_norm)[:, 1]
    else:
        y_proba = None
    
    # Generate graphs
    print("\nGenerating visualizations...")
    
    # Confusion Matrix
    cm_path = os.path.join(output_dir, f'{model_name.replace(" ", "_")}_confusion_matrix.png')
    plot_confusion_matrix(y_test, y_pred, model_name, cm_path)
    
    # ROC Curve
    if y_proba is not None:
        roc_path = os.path.join(output_dir, f'{model_name.replace(" ", "_")}_roc_curve.png')
        plot_roc_curve(y_test, y_proba, model_name, roc_path)
    
    # Feature Importance
    fi_path = os.path.join(output_dir, f'{model_name.replace(" ", "_")}_feature_importance.png')
    plot_feature_importance(model, feature_columns, model_name, fi_path)
    
    # Calculate and display metrics
    metrics = calculate_metrics(y_test, y_pred)
    print("\n" + "=" * 70)
    print(f"PERFORMANCE METRICS - {model_name}")
    print("=" * 70)
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {metrics['tp']}, TN: {metrics['tn']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
    print(f"\nGraphs saved to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    generate_all_graphs()

