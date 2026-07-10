import os
# CRITICAL: These must be set BEFORE importing numpy or sklearn!
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import pandas as pd
import numpy as np
import scipy.io
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
matplotlib.use('Agg') # Forces matplotlib to run quietly in the background

# Global configurations
RANDOM_STATE = 6740
M_SAMPLES_SVM = 5000

def process_dataset(dataset_name, x_train, y_train, x_test, y_test):
    print(f"\n{'='*60}\nPROCESSING {dataset_name.upper()}\n{'='*60}")
    
    # ==========================================
    # 1. CREATE DATA SUBSETS
    # ==========================================
    
    # For KNN Tuning: Use the FULL 60k dataset (Split 80/20)
    x_knn_train, x_knn_val, y_knn_train, y_knn_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
    )
    
    # For SVM Tuning & Training: Extract exactly 5,000 balanced samples
    x_svm_pool, _, y_svm_pool, _ = train_test_split(
        x_train, y_train, train_size=M_SAMPLES_SVM, random_state=RANDOM_STATE, stratify=y_train
    )
    
    # Split the 5,000 SVM samples into 80/20 for sweeping hyper-parameters
    x_svm_tune_train, x_svm_tune_val, y_svm_tune_train, y_svm_tune_val = train_test_split(
        x_svm_pool, y_svm_pool, test_size=0.2, random_state=RANDOM_STATE, stratify=y_svm_pool
    )

    # ==========================================
    # 2. TUNING PHASE
    # ==========================================
    
    print(f"\n[1/3] Sweeping K for KNN (Using FULL data)...")
    k_values = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    knn_accuracies = []
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_knn_train, y_knn_train)
        acc = accuracy_score(y_knn_val, knn.predict(x_knn_val))
        knn_accuracies.append(acc)
        print(f"  K={k}: Accuracy = {acc:.4f}")
        
    best_k = k_values[np.argmax(knn_accuracies)]
    print(f"-> Best K selected: {best_k}")
    
    # Plot Knee Curve
    plt.figure(figsize=(7, 4))
    plt.plot(k_values, knn_accuracies, marker='o', linestyle='dashed', color='b')
    plt.title(f'KNN Knee Plot: {dataset_name}')
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('Validation Accuracy')
    plt.xticks(k_values)
    plt.grid(True)
    #plt.show()
    plt.savefig(f'knn_knee_plot_{dataset_name}.png', bbox_inches='tight')

    print(f"\n[2/3] Sweeping hyperparameters for SVMs (Using 5000 sample)...")
    c_values = [0.1, 1.0, 10.0]
    
    # Linear SVM Sweep
    best_linear_c, best_linear_acc = None, 0
    for c in c_values:
        svc_lin = SVC(kernel='linear', C=c, random_state=RANDOM_STATE)
        svc_lin.fit(x_svm_tune_train, y_svm_tune_train)
        acc = accuracy_score(y_svm_tune_val, svc_lin.predict(x_svm_tune_val))
        if acc > best_linear_acc:
            best_linear_acc, best_linear_c = acc, c
    print(f"-> Best Linear SVM 'C': {best_linear_c} (Acc: {best_linear_acc:.4f})")
    
    # RBF SVM Sweep
    best_rbf_c, best_rbf_acc = None, 0
    for c in c_values:
        svc_rbf = SVC(kernel='rbf', C=c, random_state=RANDOM_STATE)
        svc_rbf.fit(x_svm_tune_train, y_svm_tune_train)
        acc = accuracy_score(y_svm_tune_val, svc_rbf.predict(x_svm_tune_val))
        if acc > best_rbf_acc:
            best_rbf_acc, best_rbf_c = acc, c
    print(f"-> Best RBF SVM 'C': {best_rbf_c} (Acc: {best_rbf_acc:.4f})")

    # ==========================================
    # 3. FINAL EVALUATION & METRICS PLOTTING
    # ==========================================
    
    print(f"\n[3/3] Training final models and generating plots...")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        f"KNN (K={best_k})": KNeighborsClassifier(n_neighbors=best_k), 
        f"Linear SVM (C={best_linear_c})": SVC(kernel='linear', C=best_linear_c, random_state=RANDOM_STATE),
        f"Kernel SVM RBF (C={best_rbf_c})": SVC(kernel='rbf', C=best_rbf_c, random_state=RANDOM_STATE),
        "MLP (20, 10)": MLPClassifier(hidden_layer_sizes=(20, 10), max_iter=500, random_state=RANDOM_STATE)
    }

    for name, model in models.items():
        print(f"  -> Evaluating {name}")
        
        # SVM uses the 5000 subset; all others (including KNN) use the full 60,000 pool
        if "SVM" in name:
            model.fit(x_svm_pool, y_svm_pool)
        else:
            model.fit(x_train, y_train)
            
        y_pred = model.predict(x_test)
        
        # --- Plotting Logistics ---
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=range(10))
        classes = np.arange(10)
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width, precision, width, label='Precision', color='#1f77b4')
        ax.bar(x, recall, width, label='Recall', color='#ff7f0e')
        ax.bar(x + width, f1, width, label='F1-Score', color='#2ca02c')
        
        ax.set_ylabel('Scores')
        ax.set_title(f'Performance Metrics per Class\n{name} on {dataset_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.set_ylim(0, 1.1)
        ax.legend(loc='lower right')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        #plt.show()
        plt.savefig(f"{name}_plot_{dataset_name}.png") # Add this line instead
        plt.close()

# ==========================================
# 4. DATA LOADING AND EXECUTION
# ==========================================

# --- Process Digits ---
try:
    digits_mat = scipy.io.loadmat('data/mnist_10digits.mat')
    x_train_dig = digits_mat['xtrain'] / 255.0
    y_train_dig = digits_mat['ytrain'].flatten()
    x_test_dig = digits_mat['xtest'] / 255.0
    y_test_dig = digits_mat['ytest'].flatten()
    
    process_dataset("MNIST Digits", x_train_dig, y_train_dig, x_test_dig, y_test_dig)
except FileNotFoundError:
    print("WARNING: Could not find 'data/mnist_10digits.mat'. Skipping Digits dataset.")

# --- Process Fashion ---
try:
    fashion_train_df = pd.read_csv('data/fashion-mnist_train.csv')
    fashion_test_df = pd.read_csv('data/fashion-mnist_test.csv')

    x_train_fash = fashion_train_df.iloc[:, 1:].values / 255.0
    y_train_fash = fashion_train_df.iloc[:, 0].values
    x_test_fash = fashion_test_df.iloc[:, 1:].values / 255.0
    y_test_fash = fashion_test_df.iloc[:, 0].values
    
    process_dataset("MNIST Fashion", x_train_fash, y_train_fash, x_test_fash, y_test_fash)
except FileNotFoundError:
    print("WARNING: Could not find Fashion CSV files. Skipping Fashion dataset.")