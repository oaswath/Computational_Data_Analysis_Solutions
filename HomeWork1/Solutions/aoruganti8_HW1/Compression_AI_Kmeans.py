try:
    from PIL import Image
except ImportError as e:
    raise ImportError("Pillow is required to run this module. Install with `pip install pillow`.") from e

try:
    import numpy as np
except ImportError as e:
    raise ImportError("NumPy is required to run this module. Install with `pip install numpy`.") from e

import time
import os

try:
    import pandas as pd
except ImportError as e:
    raise ImportError("Pandas is required to run this module. Install with `pip install pandas`.") from e

import matplotlib.pyplot as plt
from os.path import abspath, exists
from sklearn.cluster import KMeans

data = np.load('AImodel.npy')
print(f"Data shape: {data.shape}")
N, D = data.shape
print(f"Number of samples (N): {N}")
print(f"Number of features (D): {D}")

# Original Memory in MB
# N * D elements, 4 bytes per float32
original_memory_mb = (N * D * 4) / (1024 * 1024)
print(f"Original Memory Usage: {original_memory_mb:.4f} MB")

#Normalization step
norm_data = np.linalg.norm(data, axis=1, keepdims=True)
norm_data[norm_data == 0] = 1e-10
normalized_data = data / norm_data

# K-means clustering
kmeans_output_256 = KMeans(n_clusters=256, random_state=42, n_init=20)
output_lables_256 = kmeans_output_256.fit_predict(normalized_data)
print(f"K-means with 256 clusters completed. Cluster centers shape: {kmeans_output_256.cluster_centers_.shape}")
centroids_256 = kmeans_output_256.cluster_centers_
print(f"Centroids shape: {centroids_256.shape}")

# Memory after K-means with 256 clusters
# 256 centroids, each with D features, 4 bytes per float32 and labels for N samples, 4 bytes per int32
compression_memory_mb_256 = (256 * D * 4) / (1024 * 1024) + (N * 1)/ (1024 * 1024)  # Centroids + Labels
print(f"Memory Usage after K-means with 256 clusters: {compression_memory_mb_256:.4f} MB")

# ----- 2. RECONSTRUCTION AND ERROR CALCULATION -----
reconstructed_data = centroids_256[output_lables_256]
print(f"Reconstructed data shape: {reconstructed_data.shape}")

# 3. Cosine similarity error calculation
dot_product = np.sum(data * reconstructed_data, axis=1)
original_norm = np.linalg.norm(data, axis=1)
reconstructed_norm = np.linalg.norm(reconstructed_data, axis=1)
denominator = original_norm * reconstructed_norm
denominator[denominator == 0] = 1e-10
cosine_similarity = dot_product / denominator
avg_cosine_similarity = np.mean(cosine_similarity)
print(f"Average Cosine Similarity: {avg_cosine_similarity:.4f}")
cosine_error = 1 - cosine_similarity
average_cosine_error = np.mean(cosine_error)
print(f"Average Cosine Error: {average_cosine_error:.4f}")

# 4. Model Selection and Tuning
k_values = [64, 128, 256, 512]
compression_results = []
reconstruction_similarities = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(normalized_data)
    centroids = kmeans.cluster_centers_

    # Memory usage calculation
    # K > 256 will required 2 bytes for labels instead of 1 byte
    index_bytes = 1 if k <= 256 else 2
    compression_bytes = (k * D * 4) + (N * index_bytes)
    compression_memory_mb = compression_bytes / (1024 * 1024)
    ratio = original_memory_mb / (compression_memory_mb)
    compression_results.append(ratio)

    # Smilarily errors
    reconstructed = centroids[labels]
    dot_product = np.sum(data * reconstructed, axis=1)
    original_norm = np.linalg.norm(data, axis=1)
    reconstructed_norm = np.linalg.norm(reconstructed, axis=1)
    denominator = original_norm * reconstructed_norm
    denominator[denominator == 0] = 1e-10
    cosine_similarity = dot_product / denominator
    avg_cosine_similarity = np.mean(cosine_similarity)
    reconstruction_similarities.append(avg_cosine_similarity)

    print(f"K={k}: Comp Ratio={ratio:.4f}x, Avg Cosine Sim={avg_cosine_similarity:.4f}")

# 5. Plotting the results
fig, ax1 = plt.subplots(figsize=(8, 5))

color = 'tab:blue'
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Compression Ratio', color=color)
ax1.plot(k_values, compression_results, marker='o', color=color, label='Compression Ratio')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Reconstruction Quality (Avg Cosine Sim)', color=color)  
ax2.plot(k_values, reconstruction_similarities, marker='s', color=color, label='Reconstruction Quality')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  
plt.title('Compression Ratio vs. Reconstruction Quality')
plt.show()