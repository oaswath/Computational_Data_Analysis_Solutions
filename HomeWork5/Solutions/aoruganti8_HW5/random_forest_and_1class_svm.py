import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('data/spambase.data', header=None)
df.fillna(0, inplace=True)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Train/test split (75% train, 25% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 1. CART
dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X_train, y_train)

plt.figure(figsize=(24, 12))
# Plotting with max_depth=3 for legibility
plot_tree(dt_full, max_depth=3, filled=True, fontsize=12, class_names=['non-spam', 'spam'])
plt.title("CART Classification Tree (Truncated at Top 3 Levels for Legibility)", fontsize=18)
plt.savefig('cart_tree.png', bbox_inches='tight')
plt.close()

cart_test_error = 1 - accuracy_score(y_test, dt_full.predict(X_test))

# 2. Random Forest vs CART over n_estimators
n_trees = list(range(10, 210, 10))
rf_errors = []
for n in n_trees:
    rf = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_errors.append(1 - accuracy_score(y_test, rf.predict(X_test)))

plt.figure(figsize=(10, 6))
plt.plot(n_trees, rf_errors, marker='o', label='Random Forest Test Error')
plt.axhline(y=cart_test_error, color='r', linestyle='--', label=f'CART Test Error ({cart_test_error:.4f})')
plt.xlabel('Number of Trees')
plt.ylabel('Misclassification Error Rate')
plt.title('Test Error vs Number of Trees')
plt.legend()
plt.grid(True)
plt.savefig('rf_vs_cart.png', bbox_inches='tight')
plt.close()

# 3. RF sensitivity to nu (max_features)
max_features_list = list(range(1, 58, 4)) # step of 4 to keep it reasonably fast
oob_errors = []
test_errors_nu = []
for mf in max_features_list:
    rf_nu = RandomForestClassifier(n_estimators=100, max_features=mf, oob_score=True, random_state=42, n_jobs=-1)
    rf_nu.fit(X_train, y_train)
    oob_errors.append(1 - rf_nu.oob_score_)
    test_errors_nu.append(1 - accuracy_score(y_test, rf_nu.predict(X_test)))

plt.figure(figsize=(10, 6))
plt.plot(max_features_list, oob_errors, marker='s', label='OOB Error')
plt.plot(max_features_list, test_errors_nu, marker='o', label='Test Error')
plt.xlabel(r'Number of Randomly Selected Features ($\nu$)')
plt.ylabel('Misclassification Error Rate')
plt.title(r'Random Forest Error vs $\nu$ (max_features)')
plt.legend()
plt.grid(True)
plt.savefig('rf_nu_sensitivity.png', bbox_inches='tight')
plt.close()

# 4. One-Class SVM
# Extract non-spam from training
X_train_non_spam = X_train[y_train == 0]

# Standardize data for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_non_spam_scaled = scaler.transform(X_train_non_spam)

# Map y_test: 0 (non-spam) -> +1 (inlier), 1 (spam) -> -1 (outlier)
y_test_oc = np.where(y_test == 0, 1, -1)

gammas = [0.0001, 0.001, 0.01, 0.1, 1.0, 'scale', 'auto']
nus = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

# --- Scaled Data Grid Search ---
best_error = 1.0
best_params = {}
error_matrix_scaled = np.zeros((len(gammas), len(nus)))

for i, g in enumerate(gammas):
    for j, nu in enumerate(nus):
        ocsvm = OneClassSVM(kernel='rbf', gamma=g, nu=nu)
        ocsvm.fit(X_train_non_spam_scaled)
        preds = ocsvm.predict(X_test_scaled)
        error = np.mean(preds != y_test_oc)
        error_matrix_scaled[i, j] = error
        if error < best_error:
            best_error = error
            best_params = {'gamma': g, 'nu': nu}

# --- Unscaled Data Grid Search ---
best_error_unscaled = 1.0
best_params_unscaled = {}
error_matrix_unscaled = np.zeros((len(gammas), len(nus)))

for i, g in enumerate(gammas):
    for j, nu in enumerate(nus):
        ocsvm = OneClassSVM(kernel='rbf', gamma=g, nu=nu)
        ocsvm.fit(X_train_non_spam)
        preds = ocsvm.predict(X_test)
        error = np.mean(preds != y_test_oc)
        error_matrix_unscaled[i, j] = error
        if error < best_error_unscaled:
            best_error_unscaled = error
            best_params_unscaled = {'gamma': g, 'nu': nu}

# --- Plotting the OCSVM Heatmaps ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
gamma_labels = [str(g) for g in gammas]

# Plot 1: Scaled Data
cax1 = axes[0].imshow(error_matrix_scaled, cmap='viridis_r', aspect='auto')
axes[0].set_xticks(np.arange(len(nus)))
axes[0].set_yticks(np.arange(len(gammas)))
axes[0].set_xticklabels(nus)
axes[0].set_yticklabels(gamma_labels)
axes[0].set_xlabel('Nu (\u03BD)')
axes[0].set_ylabel('Gamma (\u03B3)')
axes[0].set_title(f'OCSVM Test Error (Scaled)\nBest: {best_error:.4f} at {best_params}')
fig.colorbar(cax1, ax=axes[0], label='Misclassification Error Rate')

# Annotate values for Scaled
for i in range(len(gammas)):
    for j in range(len(nus)):
        # Change text color based on background for readability
        val = error_matrix_scaled[i, j]
        color = "white" if val > np.mean(error_matrix_scaled) else "black"
        axes[0].text(j, i, f'{val:.3f}', ha="center", va="center", color=color, fontsize=9)

# Plot 2: Unscaled Data
cax2 = axes[1].imshow(error_matrix_unscaled, cmap='viridis_r', aspect='auto')
axes[1].set_xticks(np.arange(len(nus)))
axes[1].set_yticks(np.arange(len(gammas)))
axes[1].set_xticklabels(nus)
axes[1].set_yticklabels(gamma_labels)
axes[1].set_xlabel('Nu (\u03BD)')
axes[1].set_ylabel('Gamma (\u03B3)')
axes[1].set_title(f'OCSVM Test Error (Unscaled)\nBest: {best_error_unscaled:.4f} at {best_params_unscaled}')
fig.colorbar(cax2, ax=axes[1], label='Misclassification Error Rate')

# Annotate values for Unscaled
for i in range(len(gammas)):
    for j in range(len(nus)):
        val = error_matrix_unscaled[i, j]
        color = "white" if val > np.mean(error_matrix_unscaled) else "black"
        axes[1].text(j, i, f'{val:.3f}', ha="center", va="center", color=color, fontsize=9)

plt.tight_layout()
plt.savefig('ocsvm_heatmaps.png', bbox_inches='tight')
plt.close()

# Print Final Results
results = {
    'cart_test_error': cart_test_error,
    'rf_min_error': min(rf_errors),
    'rf_min_trees': n_trees[rf_errors.index(min(rf_errors))],
    'best_ocsvm_scaled': best_error,
    'best_ocsvm_params_scaled': best_params,
    'best_ocsvm_unscaled': best_error_unscaled,
    'best_ocsvm_params_unscaled': best_params_unscaled
}
print(results)