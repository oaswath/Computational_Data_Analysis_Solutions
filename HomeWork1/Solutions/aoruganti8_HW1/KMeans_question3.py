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
import matplotlib.pyplot as plt

# Global dictionary to log total accumulated time spent executing each seed
seed_runtimes = {42: 0.0, 106: 0.0, 213: 0.0, 56: 0.0, 88: 0.0}

class KMeansImpl:
    def __init__(self, max_iter=300, tol=1e-4, seed=42):
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed

    def load_image(self, image_name="1.jpeg"):
        return np.array(Image.open(image_name))

    def compress(self, pixels, num_clusters, norm_distance=2):
        map_result = {
            "class": None,
            "centroid": None,
            "img": None,
            "number_of_iterations": None,
            "time_taken": None,
            "additional_args": {} 
        }

        original_shape = pixels.shape
        channels = original_shape[2] if len(original_shape) == 3 else 1
        X = pixels.reshape(-1, channels).astype(np.float64)
        n_samples = X.shape[0]

        np.random.seed(self.seed)

        k = num_clusters
        random_indices = np.random.choice(n_samples, size=k, replace=False)
        centroids = X[random_indices].copy()
        labels = np.zeros(n_samples, dtype=np.int64)

        start_time = time.time()

        # Precompute the squared norm of data points once for L2 optimization
        if norm_distance == 2:
            X_squared_norms = np.sum(X ** 2, axis=1)[:, None]

        for iteration in range(self.max_iter):
            old_centroids = centroids.copy()

            if norm_distance == 2:
                # FAST L2: Uses the algebraic expansion ||x - c||^2 = ||x||^2 - 2<x,c> + ||c||^2
                # Triggers optimized BLAS matrix multiplication dot products, avoiding 3D array memory overhead
                centroid_squared_norms = np.sum(centroids ** 2, axis=1)[None, :]
                distances = X_squared_norms - 2.0 * np.dot(X, centroids.T) + centroid_squared_norms
                
                # Safeguard against tiny negative floating-point precision errors
                distances = np.maximum(distances, 0.0)
                
            elif norm_distance == 1:
                # FAST L1: Loops over clusters sequentially to prevent massive 3D array broadcasting allocation
                distances = np.empty((n_samples, k))
                for cluster_idx in range(k):
                    distances[:, cluster_idx] = np.sum(np.abs(X - centroids[cluster_idx]), axis=1)
            else: 
                raise ValueError("norm_distance must be 1 or 2")
            
            labels = np.argmin(distances, axis=1)

            unique_labels = np.unique(labels)
            if len(unique_labels) < k:
                centroids = centroids[unique_labels]
                k = len(unique_labels)
                label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
                labels = np.array([label_mapping[label] for label in labels], dtype=np.int64)
                old_centroids = centroids.copy()
                if norm_distance == 2:
                    X_squared_norms = np.sum(X ** 2, axis=1)[:, None]
            
            for cluster_idx in range(k):
                cluster_pixels = X[labels == cluster_idx]
                if len(cluster_pixels) > 0:
                    if norm_distance == 2:
                        centroids[cluster_idx] = cluster_pixels.mean(axis=0)
                    elif norm_distance == 1:
                        centroids[cluster_idx] = np.median(cluster_pixels, axis=0)
            
            centroid_shift = np.sum(np.abs(centroids - old_centroids))
            if centroid_shift < self.tol:
                total_iterations = iteration + 1
                break
        else:
            total_iterations = self.max_iter
        
        elapsed_time = time.time() - start_time

        compressed_pixels = centroids[labels].astype(np.uint8)
        compressed_image = compressed_pixels.reshape(original_shape)

        map_result["class"] = labels.reshape(original_shape[0], original_shape[1])
        map_result["centroid"] = centroids
        map_result["img"] = compressed_image
        map_result["number_of_iterations"] = total_iterations
        map_result["time_taken"] = elapsed_time
        map_result["additional_args"]["final_k"] = k

        return map_result
'''
    def compress(self, pixels, num_clusters, norm_distance=2):
        map_result = {
            "class": None,
            "centroid": None,
            "img": None,
            "number_of_iterations": None,
            "time_taken": None,
            "additional_args": {} 
        }

        original_shape = pixels.shape
        channels = original_shape[2] if len(original_shape) == 3 else 1
        X = pixels.reshape(-1, channels).astype(np.float64)
        n_samples = X.shape[0]

        np.random.seed(self.seed)

        k = num_clusters
        random_indices = np.random.choice(n_samples, size=k, replace=False)
        centroids = X[random_indices].copy()
        labels = np.zeros(n_samples, dtype=np.int64)

        start_time = time.time()

        for iteration in range(self.max_iter):
            old_centroids = centroids.copy()

            if norm_distance == 1:
                distances = np.sum(np.abs(X[:, None, :] - centroids[None, :, :]), axis=2)
            elif norm_distance == 2:
                distances = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            
            labels = np.argmin(distances, axis=1)

            unique_labels = np.unique(labels)
            if len(unique_labels) < k:
                centroids = centroids[unique_labels]
                k = len(unique_labels)
                label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
                labels = np.array([label_mapping[label] for label in labels], dtype=np.int64)
                old_centroids = centroids.copy()
            
            for cluster_idx in range(k):
                cluster_pixels = X[labels == cluster_idx]
                if len(cluster_pixels) > 0:
                    if norm_distance == 2:
                        centroids[cluster_idx] = cluster_pixels.mean(axis=0)
                    elif norm_distance == 1:
                        centroids[cluster_idx] = np.median(cluster_pixels, axis=0)
            
            centroid_shift = np.sum(np.abs(centroids - old_centroids))
            if centroid_shift < self.tol:
                total_iterations = iteration + 1
                break
        else:
            total_iterations = self.max_iter
        
        elapsed_time = time.time() - start_time

        compressed_pixels = centroids[labels].astype(np.uint8)
        compressed_image = compressed_pixels.reshape(original_shape)

        map_result["class"] = labels.reshape(original_shape[0], original_shape[1])
        map_result["centroid"] = centroids
        map_result["img"] = compressed_image
        map_result["number_of_iterations"] = total_iterations
        map_result["time_taken"] = elapsed_time
        map_result["additional_args"]["final_k"] = k

        return map_result
'''
def plot_clustering_metrics(image_name, metrics_dict, seed_history, output_dir):
    """
    Generates an expanded five-panel performance figure for each image tracking 
    Runtime, Iterations, Knee Optimization curves, and Seed Distortion variances for both L2 and L1.
    """
    k_vals = [item['k'] for item in metrics_dict['L2']]
    test_seeds = [42, 106, 213, 56, 88]
    colors = ['#1a365d', '#2b6cb0', '#e53e3e', '#dd6b20', '#319795']
    
    # Updated to a 1x5 horizontal plot layout landscape canvas
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(30, 5.5))
    fig.suptitle(f"Comprehensive K-Means Analytics & Seed Performance Report: {image_name}", fontsize=14, fontweight='bold')
    
    # PANEL 1: Processing Speed (Seconds)
    ax1.plot(k_vals, [item['time'] for item in metrics_dict['L2']], marker='o', linewidth=2, color='#1a365d', label='L2 Norm')
    ax1.plot(k_vals, [item['time'] for item in metrics_dict['L1']], marker='s', linewidth=2, color='#e53e3e', label='L1 Norm')
    ax1.set_title("Execution Runtime vs. K", fontsize=11, fontweight='semibold')
    ax1.set_xlabel("Number of Clusters (K)", fontsize=10)
    ax1.set_ylabel("Convergence Duration (Seconds)", fontsize=10)
    ax1.set_xticks(k_vals)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(frameon=True, facecolor='#f7fafc')
    
    # PANEL 2: Optimization Steps (Iterations)
    ax2.plot(k_vals, [item['iters'] for item in metrics_dict['L2']], marker='o', linewidth=2, color='#1a365d', label='L2 Norm')
    ax2.plot(k_vals, [item['iters'] for item in metrics_dict['L1']], marker='s', linewidth=2, color='#e53e3e', label='L1 Norm')
    ax2.set_title("Iteration Cycles vs. K", fontsize=11, fontweight='semibold')
    ax2.set_xlabel("Number of Clusters (K)", fontsize=10)
    ax2.set_ylabel("Total Iterations Taken", fontsize=10)
    ax2.set_xticks(k_vals)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(frameon=True, facecolor='#f7fafc')
    
    # PANEL 3: Knee / Elbow Optimization Curve (Best WCSS)
    ax3.plot(k_vals, [item['wcss'] for item in metrics_dict['L2']], marker='o', linewidth=2, color='#2b6cb0', label='L2 Distortion')
    ax3.plot(k_vals, [item['wcss'] for item in metrics_dict['L1']], marker='s', linewidth=2, color='#dd6b20', label='L1 Distortion')
    ax3.set_title("Knee Plot: Best Quality Curve", fontsize=11, fontweight='semibold')
    ax3.set_xlabel("Number of Clusters (K)", fontsize=10)
    ax3.set_ylabel("Distortion Score (Log Scale)", fontsize=10)
    ax3.set_yscale('log')
    ax3.set_xticks(k_vals)
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(frameon=True, facecolor='#f7fafc')
    
    # === PANEL 4: Seed Quality Dispersal Plot (L2 Metric) ===
    for i, seed in enumerate(test_seeds):
        seed_wcss_track = []
        for k in k_vals:
            # Look up the k-key safely; if a seed dropped a cluster and missing,
            # fallback gracefully using .get() to prevent a KeyError crash
            score = seed_history['L2'].get(k, {}).get(seed, None)
            if score is not None:
                seed_wcss_track.append(score)
            else:
                # If a specific K is missing for this seed, append the closest alternative 
                # or a fallback value so the line array matches k_vals length
                available_ks = sorted(seed_history['L2'].keys())
                closest_k = min(available_ks, key=lambda x: abs(x - k))
                seed_wcss_track.append(seed_history['L2'][closest_k].get(seed, 0.0))
                
        ax4.plot(k_vals, seed_wcss_track, marker='X', linestyle=':', linewidth=1.5, color=colors[i], label=f'Seed {seed}')
    
    ax4.set_title("Initialization Variance (L2)", fontsize=11, fontweight='semibold')
    ax4.set_xlabel("Number of Clusters (K)", fontsize=10)
    ax4.set_ylabel("Reconstruction WCSS (Log Scale)", fontsize=10)
    ax4.set_yscale('log')
    ax4.set_xticks(k_vals)
    ax4.grid(True, linestyle='--', alpha=0.5)
    ax4.legend(frameon=True, facecolor='#f7fafc', title="Seeds")
    
    # === PANEL 5: Seed Quality Dispersal Plot (L1 Metric) ===
    for i, seed in enumerate(test_seeds):
        seed_l1_track = []
        for k in k_vals:
            score = seed_history['L1'].get(k, {}).get(seed, None)
            if score is not None:
                seed_l1_track.append(score)
            else:
                available_ks = sorted(seed_history['L1'].keys())
                closest_k = min(available_ks, key=lambda x: abs(x - k))
                seed_l1_track.append(seed_history['L1'][closest_k].get(seed, 0.0))
                
        ax5.plot(k_vals, seed_l1_track, marker='o', linestyle=':', linewidth=1.5, color=colors[i], label=f'Seed {seed}')
    
    ax5.set_title("Initialization Variance (L1)", fontsize=11, fontweight='semibold')
    ax5.set_xlabel("Number of Clusters (K)", fontsize=10)
    ax5.set_ylabel("Total Absolute Error (Log Scale)", fontsize=10)
    ax5.set_yscale('log')
    ax5.set_xticks(k_vals)
    ax5.grid(True, linestyle='--', alpha=0.5)
    ax5.legend(frameon=True, facecolor='#f7fafc', title="Seeds")

    plt.tight_layout()
    plot_filename = f"{output_dir}/{image_name}_comprehensive_analysis.png"
    plt.savefig(plot_filename, dpi=200)
    plt.close()
    print(f"  --> Expanded five-panel analytical chart exported to: {plot_filename}")

def run_image_compression(image_path, k_values, output_dir="compressed_images"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    compressor = KMeansImpl()
    try:
        img_orig = compressor.load_image(image_path)
    except FileNotFoundError:
        print(f"Skipping execution: '{image_path}' was not found in the root directory.")
        return None
    
    original_shape = img_orig.shape
    image_name = os.path.basename(image_path).split('.')[0]
    
    results = {'L1': [], 'L2': []}
    # Nested dictionary architecture to store the raw distortion quality scores of ALL seeds for visualization
    seed_history = {'L1': {k: {} for k in k_values}, 'L2': {k: {} for k in k_values}}

    
    test_seeds = [42, 106, 213, 56, 88]
    metric_to_normdistance_map = {'L1': 1, 'L2': 2}

    for metric_name, norm_dist in metric_to_normdistance_map.items():
        print(f"Running K-Means with {metric_name} distance for {image_path}...")
        for k in k_values:
            best_wcss = float('inf')
            best_result = None
            best_seed = None

            for seed in test_seeds:
                compressor.seed = seed
                result = compressor.compress(img_orig, num_clusters=k, norm_distance=norm_dist)
                seed_runtimes[seed] += result["time_taken"]

                flattened_pixels = img_orig.reshape(-1, original_shape[2]).astype(np.float64) if len(original_shape) == 3 else img_orig.flatten().astype(np.float64)
                assigned_centroids = result["centroid"][result["class"].flatten()]
                
                if norm_dist == 1:
                    current_wcss = np.sum(np.abs(flattened_pixels - assigned_centroids))
                elif norm_dist == 2:
                    current_wcss = np.sum((flattened_pixels - assigned_centroids) ** 2)

                # Track the data points for ALL seeds in our tracking tree
                seed_history[metric_name][k][seed] = current_wcss

                if current_wcss < best_wcss:
                    best_wcss = current_wcss
                    best_result = result
                    best_seed = seed
            
            print(f"  K={k:<2} -> Best Seed: {best_seed:<3} | Iterations: {best_result['number_of_iterations']:<3} | Time: {best_result['time_taken']:.4f}s")

            out_filename = f"{output_dir}/{image_name}_{metric_name.lower()}_k{k}.png"
            Image.fromarray(best_result["img"]).save(out_filename)
            
            results[metric_name].append({
                'k': k, 'seed': best_seed, 'time': best_result['time_taken'], 'iters': best_result['number_of_iterations'], 'wcss': best_wcss
            })

    # Pass the populated seed tracking database into the graphing block
    plot_clustering_metrics(image_name, results, seed_history, output_dir)
    return results
    
if __name__ == "__main__":
    target_images = ["artemisII.jpg", "football.bmp", "my_custom_photo.jpg"]
    k_settings = [2, 5, 15, 25, 50]
    
    print("=========================================================")
    print("Starting Vectorized Image Compression Optimization Suite")
    print("=========================================================")
    
    for img_file in target_images:
        if not os.path.exists(img_file):
            print(f"Creating a simulated color array file for '{img_file}' validation purposes...")
            simulated_img = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
            Image.fromarray(simulated_img).save(img_file)
            
        run_image_compression(img_file, k_settings)
    
    print("\n=========================================================")
    print("TOTAL ACCUMULATED TIME SPENT PER SEED (ALL RUNS)")
    print("=========================================================")
    total_pipeline_time = 0.0
    for seed, accumulated_time in seed_runtimes.items():
        print(f"  Seed {seed:<3} -> Total Time across all configurations: {accumulated_time:.4f} seconds")
        total_pipeline_time += accumulated_time
    print("---------------------------------------------------------")
    print(f"  Total Optimization Pipeline Execution Time: {total_pipeline_time:.4f} seconds")
    print("=========================================================")