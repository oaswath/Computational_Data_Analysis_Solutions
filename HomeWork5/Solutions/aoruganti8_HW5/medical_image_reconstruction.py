import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# ==========================================
# 1. LOAD DATA & SIMULATE MRI MEASUREMENTS
# ==========================================
print("Loading data and generating measurements...")

# Load the MATLAB file
mat = scipy.io.loadmat('data/cs.mat')
var_name = [k for k in mat.keys() if not k.startswith('__')][0]
print(f"Shape of true image: {mat[var_name].shape}")
print(f"Keys in the loaded MATLAB file: {list(mat.keys())}")
img_true = mat[var_name]

# Flatten the 50x50 image into a 2500x1 vector
x_true = img_true.flatten()
p = len(x_true) # 2500
n = 1300        # number of measurements

# Set random seed for reproducibility
np.random.seed(42)

# Generate measurement matrix A ~ N(0, 1)
A = np.random.randn(n, p)
print(f"Shape of measurement matrix A: {A.shape}")

# Generate noise eps ~ N(0, 25). Standard deviation is sqrt(25) = 5
eps = np.random.normal(loc=0.0, scale=5.0, size=n)
print(f"Shape of noise vector eps: {eps.shape}")

# Generate observed measurements y
y = A @ x_true + eps
print(f"Shape of observed measurements y: {y.shape}")

# ==========================================
# 2. LASSO RECONSTRUCTION (10-Fold CV)
# ==========================================
print("Running Lasso with 10-fold CV...")
lasso_cv = LassoCV(cv=10, max_iter=10000, random_state=42, n_jobs=-1)
lasso_cv.fit(A, y)

best_alpha_lasso = lasso_cv.alpha_
x_lasso = lasso_cv.coef_
img_lasso = x_lasso.reshape(50, 50)

# Extract Cross-Validation errors for Lasso
m_log_alphas_lasso = np.log10(lasso_cv.alphas_)
mean_mse_lasso = lasso_cv.mse_path_.mean(axis=-1)

# ==========================================
# 3. RIDGE RECONSTRUCTION (10-Fold CV)
# ==========================================
print("Running Ridge with 10-fold CV...")
alphas_ridge = np.logspace(-1, 5, 100)
ridge_model = Ridge(max_iter=10000)
ridge_grid = GridSearchCV(ridge_model, param_grid={'alpha': alphas_ridge}, 
                          cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
ridge_grid.fit(A, y)

best_alpha_ridge = ridge_grid.best_params_['alpha']
x_ridge = ridge_grid.best_estimator_.coef_
img_ridge = x_ridge.reshape(50, 50)

# Extract Cross-Validation errors for Ridge
m_log_alphas_ridge = np.log10(alphas_ridge)
mean_mse_ridge = -ridge_grid.cv_results_['mean_test_score']

# ==========================================
# 4. PLOTTING THE RESULTS (INDIVIDUAL PLOTS)
# ==========================================
print("Generating individual plots...")

# --- Plot 1: True Image ---
plt.figure(figsize=(6, 6))
plt.imshow(img_true, cmap='gray')
plt.title('True Sparse Image\n(50x50)')
plt.axis('off')
plt.show()

# --- Plot 2: Lasso CV Curve ---
# Notice the 'rf' and 'r' prefixes to fix the \lambda SyntaxWarning
plt.figure(figsize=(8, 6))
plt.plot(m_log_alphas_lasso, mean_mse_lasso, color='blue', linewidth=2)
plt.axvline(np.log10(best_alpha_lasso), linestyle='--', color='k', 
            label=rf'Best $\lambda$: {best_alpha_lasso:.4f}')
plt.title('Lasso 10-Fold CV Error')
plt.xlabel(r'log10($\lambda$)')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.show()

# --- Plot 3: Lasso Recovered Image ---
plt.figure(figsize=(6, 6))
plt.imshow(img_lasso, cmap='gray')
plt.title(rf'Lasso Recovered Image (Best $\lambda$ = {best_alpha_lasso:.4f})')
plt.axis('off')
plt.show()

# --- Plot 4: Ridge CV Curve ---
plt.figure(figsize=(8, 6))
plt.plot(m_log_alphas_ridge, mean_mse_ridge, color='red', linewidth=2)
plt.axvline(np.log10(best_alpha_ridge), linestyle='--', color='k', 
            label=rf'Best $\lambda$: {best_alpha_ridge:.4f}')
plt.title('Ridge 10-Fold CV Error')
plt.xlabel(r'log10($\lambda$)')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.show()

# --- Plot 5: Ridge Recovered Image ---
plt.figure(figsize=(6, 6))
plt.imshow(img_ridge, cmap='gray')
plt.title(rf'Ridge Recovered Image (Best $\lambda$ = {best_alpha_ridge:.2f})')
plt.axis('off')
plt.show()