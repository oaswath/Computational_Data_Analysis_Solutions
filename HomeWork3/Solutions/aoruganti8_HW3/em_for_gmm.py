import os
from PIL import Image
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ==========================================
# 0. Data Loading and Preprocessing (PCA)
# ==========================================

# Load Data (Handling variable keys dynamically in case they are named 'data', 'images', etc.)
data_mat = scipy.io.loadmat('data/data.mat')
data_key = [k for k in data_mat.keys() if not k.startswith('_')][0]
X_raw = data_mat[data_key].astype(np.float64)  # Shape: 784 x 1990
print(f"Loaded data shape: {X_raw.shape}")

label_mat = scipy.io.loadmat('data/label.mat')
label_key = [k for k in label_mat.keys() if not k.startswith('_')][0]
y_true = label_mat[label_key].flatten()        # Shape: 1990
print(f"Loaded labels shape: {y_true.shape}")

# The columns are images. We need rows to be samples for sklearn and our EM.
X = X_raw.T  # Shape: 1990 x 784
print(f"Transformed data shape: {X.shape}")

# Apply PCA to reduce dimensionality to 4
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X)  # Shape: 1990 x 4
n_samples, n_features = X_pca.shape
print(f"PCA reduced data shape: {X_pca.shape}")

# ==========================================
# 1. EM Algorithm Implementation
# ==========================================

# Initialization constraints given by the prompt
np.random.seed(42) # For reproducibility

# Means: Random Gaussian vectors with zero mean
mu1 = np.random.randn(n_features)
mu2 = np.random.randn(n_features)
print(f"Initialized means: mu1 = {mu1}, mu2 = {mu2}")

# Covariances: S * S.T + I
S1 = np.random.randn(n_features, n_features)
S2 = np.random.randn(n_features, n_features)
cov1 = S1 @ S1.T + np.eye(n_features)
cov2 = S2 @ S2.T + np.eye(n_features)
print(f"Initialized covariances: cov1 = {cov1}, cov2 = {cov2}")

# Mixing weights: initialized uniformly
pi1, pi2 = 0.5, 0.5

max_iters = 100
log_likelihoods = []
tol = 1e-6

for i in range(max_iters):
    # --- E-Step ---
    # Compute log probabilities to prevent numerical underflow
    log_p1 = multivariate_normal.logpdf(X_pca, mean=mu1, cov=cov1) + np.log(pi1)
    log_p2 = multivariate_normal.logpdf(X_pca, mean=mu2, cov=cov2) + np.log(pi2)
    
    # Log-sum-exp trick for stable denominator calculation
    log_max = np.maximum(log_p1, log_p2)
    log_sum = log_max + np.log(np.exp(log_p1 - log_max) + np.exp(log_p2 - log_max))
    
    # Compute total log likelihood for this iteration
    current_ll = np.sum(log_sum)
    print(f"Iteration {i}: Log-Likelihood = {current_ll}")

    log_likelihoods.append(current_ll)
    
    # Check convergence
    if i > 0 and np.abs(current_ll - log_likelihoods[-2]) < tol:
        print(f"EM converged at iteration {i}")
        break
        
    # Calculate responsibilities (tau_ik)
    tau1 = np.exp(log_p1 - log_sum)
    tau2 = np.exp(log_p2 - log_sum)
    
    # --- M-Step ---
    N1 = np.sum(tau1)
    N2 = np.sum(tau2)
    
    # Update Weights
    pi1 = N1 / n_samples
    pi2 = N2 / n_samples
    
    # Update Means
    # Initialize empty arrays for the means (size: n_features)
    mu1 = np.zeros(n_features)
    mu2 = np.zeros(n_features)

    # Iterate through every single data point
    for i in range(n_samples):
        # Multiply the scalar responsibility by the 4D data vector and add to the running total
     mu1 += tau1[i] * X_pca[i]
     mu2 += tau2[i] * X_pca[i]

    # Finally, divide by the sum of the responsibilities (N1 and N2)
    mu1 /= N1
    mu2 /= N2
    
    # Update Covariances
    diff1 = X_pca - mu1
    diff2 = X_pca - mu2

    # Initialize empty DxD matrices
    cov1 = np.zeros((n_features, n_features))
    cov2 = np.zeros((n_features, n_features))

    # Sum the weighted outer products iteratively
    for i in range(n_samples):
        cov1 += tau1[i] * np.outer(diff1[i], diff1[i])
        cov2 += tau2[i] * np.outer(diff2[i], diff2[i])

    cov1 /= N1
    cov2 /= N2
    
# Plot 1: Log-Likelihood vs Iterations
plt.figure(figsize=(8, 4))
plt.plot(log_likelihoods, marker='o', linestyle='-')
plt.title('Log-Likelihood vs. EM Iterations')
plt.xlabel('Iteration')
plt.ylabel('Log-Likelihood')
plt.grid(True)
plt.show()

# ==========================================
# 2. Report Fitted GMM Model
# ==========================================

print("\n--- GMM Parameters ---")
print(f"Component 1 Weight (pi1): {pi1:.4f}")
print(f"Component 2 Weight (pi2): {pi2:.4f}")

# Map means back to the original 784-dimensional space
# Reshape to (1, -1) to create a 2D array, then extract the first item [0] to get back a flat 784 array
mu1_orig = pca.inverse_transform(mu1.reshape(1, -1))[0]
mu2_orig = pca.inverse_transform(mu2.reshape(1, -1))[0]

# Plot 2: Reconstructed Means as Images
plt.figure(figsize=(4, 4))
plt.imshow(mu1_orig.reshape(28, 28, order='F'), cmap='gray')
plt.title('Reconstructed Mean 1')
plt.axis('off')
plt.show()

plt.figure(figsize=(4, 4))
plt.imshow(mu2_orig.reshape(28, 28, order='F'), cmap='gray')
plt.title('Reconstructed Mean 2')
plt.axis('off')
plt.show()

# Plot 3: 4x4 Covariance Matrix Heatmaps
plt.figure(figsize=(5, 4))
im1 = plt.imshow(cov1, cmap='viridis')
plt.title('Covariance Matrix 1 Intensity')
plt.colorbar(im1)
plt.show()

plt.figure(figsize=(5, 4))
im2 = plt.imshow(cov2, cmap='viridis')
plt.title('Covariance Matrix 2 Intensity')
plt.colorbar(im2)
plt.show()

# ==========================================
# 3. Label Inference & K-Means Comparison
# ==========================================

def map_clusters_to_labels(preds, y_true):
    """Maps unsupervised cluster IDs (0, 1) to true labels (2, 6) based on highest accuracy."""
    # Try mapping (0 -> 2, 1 -> 6)
    map_A = np.where(preds == 0, 2, 6)
    acc_A = np.mean(map_A == y_true)
    
    # Try mapping (0 -> 6, 1 -> 2)
    map_B = np.where(preds == 0, 6, 2)
    acc_B = np.mean(map_B == y_true)
    
    return map_A if acc_A > acc_B else map_B

# GMM Predictions
gmm_preds = np.argmax(np.column_stack((tau1, tau2)), axis=1)
gmm_mapped = map_clusters_to_labels(gmm_preds, y_true)

# K-Means Predictions
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans_preds = kmeans.fit_predict(X_pca)
kmeans_mapped = map_clusters_to_labels(kmeans_preds, y_true)

# Calculate Errors
mask_2 = (y_true == 2)
mask_6 = (y_true == 6)

gmm_err_2 = 1 - np.mean(gmm_mapped[mask_2] == y_true[mask_2])
gmm_err_6 = 1 - np.mean(gmm_mapped[mask_6] == y_true[mask_6])

kmeans_err_2 = 1 - np.mean(kmeans_mapped[mask_2] == y_true[mask_2])
kmeans_err_6 = 1 - np.mean(kmeans_mapped[mask_6] == y_true[mask_6])

print("\n--- Misclassification Rates (1 - Accuracy) ---")
print(f"{'Method':<10} | {'Digit 2':<10} | {'Digit 6':<10}")
print("-" * 35)
print(f"{'K-Means':<10} | {kmeans_err_2:<10.4f} | {kmeans_err_6:<10.4f}")
print(f"{'GMM':<10} | {gmm_err_2:<10.4f} | {gmm_err_6:<10.4f}")