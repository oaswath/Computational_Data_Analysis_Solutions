import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# --- 1. Load the Data ---
data = scipy.io.loadmat('data/data.mat')
dataset = data['data']
X = dataset[:, 0].reshape(-1, 1)
y = dataset[:, 1]

# --- 2. Locally Weighted Linear Regression Function ---
def lwlr(x_target, X_train, y_train, h):
    """
    Computes local linear regression prediction for a single target point.
    """
    # Compute Gaussian Kernel weights
    z = x_target - X_train.flatten()
    W_diag = (1 / (np.sqrt(2 * np.pi) * h)) * np.exp(- (z**2) / (2 * h**2))
    W = np.diag(W_diag)
    
    # Formulate Design matrix X (intercept and centered features)
    X_mat = np.vstack([np.ones_like(z), z]).T
    
    # Solve for Beta: beta = (X^T W X)^-1 X^T W y
    # A small ridge term (1e-8) is added to ensure numerical stability/invertibility
    XTWX = X_mat.T @ W @ X_mat + np.eye(2) * 1e-8 
    beta = np.linalg.inv(XTWX) @ X_mat.T @ W @ y_train
    
    # Since features are centered around x_target, the prediction is exactly beta_0
    return beta[0]

# --- 3. 5-Fold Cross Validation ---
h_vals = np.logspace(-2, 1, 50) # Search range for bandwidth
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_errors = []
for h in h_vals:
    fold_errors = []
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        preds = [lwlr(x_val[0], X_train, y_train, h) for x_val in X_val]
        mse = np.mean((np.array(preds) - y_val)**2)
        fold_errors.append(mse)
    cv_errors.append(np.mean(fold_errors))

best_idx = np.argmin(cv_errors)
best_h = h_vals[best_idx]
print(f'Optimal Bandwidth (h): {best_h:.4f}')

# Plot 1: Cross Validation Curve
plt.figure(figsize=(8,5))
plt.plot(h_vals, cv_errors, marker='o', color='teal')
plt.xscale('log')
plt.xlabel('Bandwidth (h)')
plt.ylabel('5-fold CV Mean Squared Error')
plt.title('Bias-Variance Tradeoff: CV Error vs Bandwidth')
plt.axvline(best_h, color='red', linestyle='--', label=f'Optimal h = {best_h:.4f}')
plt.legend()
plt.tight_layout()
plt.savefig('cv_curve.png')
plt.show()

# --- 4. Prediction and Final Fit Plot ---
# Predict specifically for x = 1.4 using ALL data
pred_1_4 = lwlr(1.4, X, y, best_h)
print(f'Predicted value at x = 1.4: {pred_1_4:.4f}')

# Generate smooth prediction curve over the domain
x_grid = np.linspace(X.min(), X.max(), 200)
y_pred_grid = [lwlr(xi, X, y, best_h) for xi in x_grid]

# Plot 2: Final Prediction Curve
plt.figure(figsize=(10,6))
plt.scatter(X, y, alpha=0.5, label='Training Data', color='gray')
plt.plot(x_grid, y_pred_grid, color='blue', label='LWLR Prediction Curve', linewidth=2)
plt.scatter([1.4], [pred_1_4], color='red', s=120, zorder=5, 
            label=f'Prediction at x=1.4 (y={pred_1_4:.4f})')

plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Locally Weighted Linear Regression (h={best_h:.4f})')
plt.legend()
plt.tight_layout()
plt.savefig('lwlr_fit.png')
plt.show()