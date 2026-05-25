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

class PoliticalBlogsClustering:
    def __init__(self):
        pass

    def parse_nodes_file(self, filepath):
        """
        Parses the nodes.txt file to extract node IDs and their corresponding political orientation labels.
        Returns a dictionary mapping node_id to label.
        """
        nodes_ground_truth = {}
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        node_id = int(parts[0])
                        label = int(parts[2])
                        nodes_ground_truth[node_id] = label
        return nodes_ground_truth
    
    def parse_edges_file(self, filepath):
        """
        Parses the edges.txt file to extract edges between nodes.
        Returns a list of tuples representing edges (u, v).
        """
        edges = []
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        edges.append((int(parts[0]), int(parts[1])))
        return edges

    def manual_kmeans(self, eigen_k, k, max_iter=150, seed=42):
        """
        Manual implementation of the K-Means clustering algorithm to satisfy constraints.
        """
        np.random.seed(seed)
        n_samples, n_features = eigen_k.shape
    
        
        # Randomly choose initial centroids from rows
        idx = np.random.choice(n_samples, k, replace=False)
        centroids = eigen_k[idx].copy()

        labels = np.zeros(n_samples, dtype=np.int64)
        
        for conv_iter in range(max_iter):
            old_centroids = centroids.copy()
            
            n_samples = eigen_k.shape[0]
            k_clusters = centroids.shape[0]
            n_features = eigen_k.shape[1]
            distances = np.zeros((n_samples, k_clusters))

            for i in range(n_samples):
                for j in range(k_clusters):
                    squared_distance = 0.0
                    for f in range(n_features):
                        diff = eigen_k[i, f] - centroids[j, f]
                        squared_distance += diff * diff
                    distances[i, j] = squared_distance


            # Compute Euclidean distance using broadcasting
            # distances = np.sum((eigen_k[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            labels = np.argmin(distances, axis=1)
                    
            # Recompute centroids based on cluster means
            for i in range(k):
                mask = (labels == i)
                if np.sum(mask) > 0:
                    centroids[i] = eigen_k[mask].mean(axis=0)
            
            # Break early if centroids converge
            if np.allclose(centroids, old_centroids, atol=1e-6):
                break
                
        return labels

    def find_majority_labels(self, num_clusters = 2):
        '''
        This method loads the data, performs spectral clustering and reports the majority labels

        Inputs:
            num_clusters (int): The number of clusters to be created

        Output:
            A map with following attributes
            1. overall_mismatch_rate: <2 decimal places>
            2. mismatch_rates: [{"majority_index": <int>, "mismatch_rate": <2 decimal places>}]
        '''
        map_result = {
            "overall_mismatch_rate": None,
            "mismatch_rates": []
        }
       
        '''
        # 1. Parse nodes file to extract ground-truth orientation labels
        nodes_ground_truth = {}
        if os.path.exists("nodes.txt"):
            with open("nodes.txt", "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    parts = line.strip().split("\t")
                    if len(parts) >= 3:
                        node_id = int(parts[0])
                        label = int(parts[2])
                        nodes_ground_truth[node_id] = label

        # 2. Parse edges file
        edges = []
        if os.path.exists("edges.txt"):
            with open("edges.txt", "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        edges.append((int(parts[0]), int(parts[1])))
        '''
        # 1. Parse nodes file to extract ground-truth orientation labels
        nodes_ground_truth = self.parse_nodes_file("nodes.txt")

        # 2. Parse edges file to extract graph structure
        edges = self.parse_edges_file("edges.txt")

        # 3. Identify and remove isolated nodes as required by preprocessing instructions
        active_nodes = set()

        for edge in edges:
            u = edge[0]
            v = edge[1]
            if u not in active_nodes:
                active_nodes.add(u)
            if v not in active_nodes:
                active_nodes.add(v)

        connected_nodes = sorted(list(active_nodes))

        node_to_idx = {node_id: idx for idx, node_id in enumerate(connected_nodes)}
        

        n_active = len(connected_nodes)

        if n_active == 0:
            return map_result

        # 4. Construct Symmetrical Adjacency Matrix
        A = np.zeros((n_active, n_active))
        for u, v in edges:
            if u in node_to_idx and v in node_to_idx:
                i, j = node_to_idx[u], node_to_idx[v]
                A[i, j] = 1
                A[j, i] = 1

        # 5. Build Symmetric Normalized Laplacian Matrix: L_sym = I - D^(-1/2) * A * D^(-1/2)
        degrees = np.sum(A, axis=1)
        d_inv_sqrt = np.zeros_like(degrees)
        for i in range(len(degrees)):
            if degrees[i] > 0:
                d_inv_sqrt[i] = 1.0 / np.sqrt(degrees[i])
            else:                
                d_inv_sqrt[i] = 0.0
        D_inv_sqrt = np.diag(d_inv_sqrt)
        L_sym = np.eye(n_active) - D_inv_sqrt @ A @ D_inv_sqrt

        # 6. Extract Eigenvectors via Hermitian Eigh (Self-contained, sorted eigenvalue solver)
        eigenvalues, eigenvectors = np.linalg.eigh(L_sym)
        idx_sort = np.argsort(eigenvalues)
        eigenvectors = eigenvectors[:, idx_sort]

        # 7. Construct Matrix U from the first k lowest eigenvectors and project to unit sphere rows
        k = num_clusters
        n_rows = eigenvectors.shape[0]
        eigen_k = np.empty((n_rows, k))
        for col_index in range(k):
            eigen_k[:, col_index] = eigenvectors[:, col_index]
        row_norms = np.linalg.norm(eigen_k, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1e-12
        eigen_k = eigen_k / row_norms

        # 8. Run manual K-Means clustering assignment
        cluster_assignments = self.manual_kmeans(eigen_k, k, seed=42)

        # 9. Calculate mismatch rates against political orientations
        labels_list = []
        for node_id in connected_nodes:
            current_truth = nodes_ground_truth[node_id]
            labels_list.append(current_truth)
        true_labels = np.array(labels_list)

        total_mismatches = 0
        mismatch_rates_list = []

        for cluster_idx in range(k):
         
            mask = (cluster_assignments == cluster_idx)
            cluster_true_labels = true_labels[mask]
         
            
            if len(cluster_true_labels) == 0:
                mismatch_rates_list.append({
                    "cluster_id": cluster_idx,
                    "majority_index": 0,
                    "mismatch_rate": 0.00,
                    "cluster_size": 0
                })
                continue
                
            # Compute majority label using basic array counting operations
            counts = np.bincount(cluster_true_labels)
            majority_label = np.argmax(counts)
            
            mismatches = np.sum(cluster_true_labels != majority_label)
            total_mismatches += mismatches
            mismatch_rate = mismatches / len(cluster_true_labels)
            
            
            mismatch_rates_list.append({
                "cluster_id": cluster_idx,
                "majority_index": int(majority_label),
                "mismatch_rate": round(mismatch_rate, 2),
                "cluster_size": len(cluster_true_labels)
            })


        overall_mismatch_rate = total_mismatches / len(true_labels)
        
        map_result["overall_mismatch_rate"] = round(overall_mismatch_rate, 2)
        map_result["mismatch_rates"] = mismatch_rates_list

        return map_result

if __name__ == "__main__":
    clustering_solution = PoliticalBlogsClustering()
    target_k_values = [2, 5, 10, 30, 50]
    
    # --- NEW: Create a central plots directory safely ---
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    print("==========================================================================")
    print("Executing Spectral Clustering Pipeline for Political Blogs")
    print("==========================================================================")
    
    for k in target_k_values:
        print(f"\n--- Processing K = {k} clusters---")
        Label_evaluation = clustering_solution.find_majority_labels(num_clusters=k)
        
        # 1. Convert to Pandas DataFrame for Tabular Data
        df = pd.DataFrame(Label_evaluation['mismatch_rates'])
        
        # Sort values logically: First by political party (majority index), then by mismatch rate
        df_sorted = df.sort_values(by=['majority_index', 'mismatch_rate'], ascending=[True, True])
        
        # Export the table to a CSV file (kept in the root directory for easy access)
        csv_filename = f"cluster_metrics_K_{k}.csv"
        df_sorted.to_csv(csv_filename, index=False)
        
        print(f"\nConfiguration: K = {k} Clusters (CSV saved to root)")
        print(f" -> Overall Network Mismatch Rate: {Label_evaluation['overall_mismatch_rate']:.2%}")

        for idx, item in enumerate(Label_evaluation['mismatch_rates']):
            print(f"    * Cluster {idx:02d}: Majority Political Stance = {item['majority_index']} | Mismatch Rate = {item['mismatch_rate']:.2f}")
        
        # 2. Generate Bar Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Color coding: e.g., Blue for Party 0, Red for Party 1
        colors = ['#1f77b4' if row['majority_index'] == 0 else '#d62728' for _, row in df_sorted.iterrows()]
        
        # Use string conversion of cluster_id to prevent matplotlib from treating x-axis as continuous
        ax.bar(df_sorted['cluster_id'].astype(str), df_sorted['mismatch_rate'], color=colors, edgecolor='black')
        
        ax.set_title(f'Mismatch Rates per Cluster (K={k})\nOverall Mismatch Rate: {Label_evaluation["overall_mismatch_rate"]:.2%}')
        ax.set_xlabel('Cluster ID (Sorted by Party, then Error)')
        ax.set_ylabel('Mismatch Rate (Error %)')
        ax.set_ylim(0, 1.0) # Standardize Y-axis to 0-100%
        
        # Create custom legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#1f77b4', edgecolor='black', label='Majority: 0'),
                           Patch(facecolor='#d62728', edgecolor='black', label='Majority: 1')]
        ax.legend(handles=legend_elements, loc='upper left')
        
        # Adjust layout and save the figure
        plt.tight_layout()
        
        # --- NEW: Route the saved image file into the 'plots' directory ---
        plot_filename = os.path.join(plot_dir, f"mismatch_plot_K_{k}.png")
        plt.savefig(plot_filename)
        plt.close(fig)
        
    print("\n==========================================================================")
    print("Execution Complete.")
    print(" - CSV tables generated in the current directory.")
    print(f" - PNG plots safely routed to the './{plot_dir}/' folder.")