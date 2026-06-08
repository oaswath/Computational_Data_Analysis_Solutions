try:
    from pca import PCA
except ImportError:
    print("Could not import local PCA class. Falling back to sklearn PCA.")
import numpy as np
import scipy.io as sio
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import shortest_path, minimum_spanning_tree
from scipy.sparse import csr_matrix
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# -----------------------------------------------------------------------------
# NOTE: Do not change the parameters / return types for pre defined methods.
# -----------------------------------------------------------------------------
class OrderOfFaces:
    """
    This class handles loading and processing facial image data for dimensionality
    reduction using the ISOMAP algorithm, with PCA as an optional comparison.

    Attributes:
    ----------
    images_path : str
        Path to the .mat file containing the image dataset.

    Methods:
    -------
    get_adjacency_matrix(epsilon):
        Returns the adjacency matrix based on a given epsilon neighborhood.

    get_best_epsilon():
        Returns the best epsilon for the ISOMAP algorithm, likely based on
        graph connectivity or reconstruction error.

    isomap(epsilon):
        Computes a 2D embedding of the data using the ISOMAP algorithm.

    pca(num_dim):
        Returns a low-dimensional embedding of the data using PCA.
    """

    def __init__(self, images_path='data/isomap.mat'):
        """
        Initializes the OrderOfFaces object and loads image data from the given path.

        Parameters:
        ----------
        images_path : str
            Path to the .mat file containing the facial images dataset.
        """
        self.images_path = images_path

        if images_path.endswith('.mat'):
            data = sio.loadmat(images_path)
            for k in data.keys():
                if k.startswith('__'):
                    continue
                self.images = data[k]  # Assuming the first non-__ key contains the images
        else:
            self.images = np.loadtxt(images_path)  # For .txt files, if needed
        
        if self.images.shape[1] != 4096:
            self.images = self.images.T  # Transpose if images are in rows instead of columns. m samples x 4096 features
        print(f"Final images shape: {self.images.shape}")
    
    def get_adjacency_matrix(self, epsilon: float) -> np.ndarray:
        """
        Constructs the adjacency matrix using epsilon neighborhoods.

        Parameters:
        ----------
        epsilon : float
            The neighborhood radius within which points are considered connected.

        Returns:
        -------
        np.ndarray
            A 2D adjacency matrix (m x m) where each entry represents distance between
            neighbors within the epsilon threshold.
        """
        distance_matrix = cdist(self.images, self.images, metric='euclidean')
                
        # Create the adjacency matrix by keeping distances within epsilon and setting others to zero
        adjacency_matrix = np.where(distance_matrix <= epsilon, distance_matrix, 0.0)
       
        return adjacency_matrix


    def get_best_epsilon(self) -> float:
        """
        Heuristically determines the best epsilon value for graph connectivity in ISOMAP.

        Returns:
        -------
        float
            Optimal epsilon value ensuring a well-connected neighborhood graph.
        """
        distance_matrix = cdist(self.images, self.images, metric='euclidean')
        np.fill_diagonal(distance_matrix, np.inf)  # Ignore self-distances

        #compute MST on a fully connected graph to find the maximum edge weight in the MST, which can be a good heuristic for epsilon
        mst = minimum_spanning_tree(csr_matrix(distance_matrix))

        # Find the maximum edge weight in the MST
        max_edge_weight = mst.max()
        print(f"Maximum edge weight in MST: {max_edge_weight}")

        # Return a value slightly larger than the maximum edge weight to ensure connectivity
        return float(max_edge_weight * 1.01)  # Return a slightly larger value to ensure connectivity

    def isomap(self, epsilon: float) -> np.ndarray:
        """
        Applies the ISOMAP algorithm to compute a 2D low-dimensional embedding of the dataset.

        Parameters:
        ----------
        epsilon : float
            The neighborhood radius for building the adjacency graph.

        Returns:
        -------
        np.ndarray
            A (m x 2) array where each row is a 2D embedding of the original data point.
        """

        #step 1: Build a weighted graph using the adjacency matrix
        A_weighted = self.get_adjacency_matrix(epsilon)

        #Convert the adjacency matrix to a sparse format for efficient shortest path computation
        A_weighted_sparse = csr_matrix(A_weighted)

        #step 2: Compute the shortest path distances between all pairs of points in the graph
        distance_matrix = shortest_path(csgraph=A_weighted_sparse, method='auto', directed=False)

        # Check for disconnected components (represented as inf in D)
        if np.isinf(distance_matrix).any():
            raise ValueError(f"Graph is disconnected at epsilon={epsilon}. Try a larger value.")

        #step 3: compute centering matrix H and matrix C
        m = distance_matrix.shape[0]
        H = np.eye(m) - (1.0 / m) * np.ones((m, m))
      
        #Compute the matrix C using the formula C = -0.5 * H * D^2 * H
        C = -0.5 * H @ (distance_matrix ** 2) @ H
      
        #step 4: compute the top 2 eigenvectors of C corresponding to the largest eigenvalues
        eigenvalues, eigenvectors = eigh(C)
              
        # Sort the eigenvalues and corresponding eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        # Take the top 2 indices for eigenvectors corresponding to the largest eigenvalues
        top_eigenvalues = eigenvalues[:2]
        top_eigenvectors = eigenvectors[:, idx[:2]]
   
        #step 5: return the 2D embedding by scaling the top 2 eigenvectors with the square root of their corresponding eigenvalues
        embedding_Z = top_eigenvectors * np.sqrt(top_eigenvalues)
        print(f"Computed 2D embedding with shape: {embedding_Z.shape}")
        print(f"2D embedding sample:\n{embedding_Z[:5, :]}")

        return embedding_Z


    def pca(self, num_dim: int) -> np.ndarray:
        """
        Applies PCA to reduce the dataset to a specified number of dimensions.

        Parameters:
        ----------
        num_dim : int
            Number of principal components to project the data onto.

        Returns:
        -------
        np.ndarray
            A (m x num_dim) array representing the dataset in a reduced PCA space.
        """
        #Assuming the provided PCA implementation is correct, we can use it directly. If not, we can implement PCA using SVD or eigen decomposition.
        try:
            # Assuming the provided template's `pca` class has a standard scikit-learn like API
            pca_model = PCA(n_components=num_dim)
            embedding_Z = pca_model.fit_transform(self.images)
            return embedding_Z
        except Exception:
            # Fallback if the local `pca` class acts differently or isn't available
            from sklearn.decomposition import PCA as sklearn_PCA
            pca_model = sklearn_PCA(n_components=num_dim)
            return pca_model.fit_transform(self.images)
        
if __name__ == "__main__":
    """
    Main method to execute the ISOMAP and PCA algorithms, and visualize the results.
    """
    face_model = OrderOfFaces('data/isomap.mat')
    best_epsilon = face_model.get_best_epsilon()
    print(f"Best epsilon for ISOMAP: {best_epsilon}")

    # ---------------------------------------------------------
    # Part 1: Visualize Adjacency Matrix
    # ---------------------------------------------------------
    A = face_model.get_adjacency_matrix(epsilon=best_epsilon)  # Use the best epsilon value

    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.imshow(A > 0, cmap='binary', interpolation='none')
    ax1.set_title(f"Nearest Neighbor Adjacency Matrix (epsilon={best_epsilon:.2f})")
    ax1.set_xlabel("Image Index")
    ax1.set_ylabel("Image Index")

    # ---------------------------------------------------------
    # Part 2: ISOMAP Scatter Plot with Faces
    # ---------------------------------------------------------
    Z_isomap = face_model.isomap(epsilon=best_epsilon)

    fig2, ax2 = plt.subplots(figsize=(12, 10))
    ax2.scatter(Z_isomap[:, 0], Z_isomap[:, 1], c='blue', s=10, alpha=0.5)
    ax2.set_title("2D ISOMAP Embedding of Faces")

    # Overlay a random subset of faces (e.g., 20 faces) on the scatter plot
    num_faces_to_show = 20
    sample_indices = np.random.choice(range(len(face_model.images)), num_faces_to_show, replace=False)
    
    for idx in sample_indices:
    # Reshape the 4096 row vector back to 64x64
        img = face_model.images[idx].reshape(64, 64).T 
        imagebox = OffsetImage(img, cmap='gray', zoom=0.5)
        ab = AnnotationBbox(imagebox, (Z_isomap[idx, 0], Z_isomap[idx, 1]), frameon=False)
        ax2.add_artist(ab)

    # ---------------------------------------------------------
    # Part 3: PCA Scatter Plot with Faces
    # ---------------------------------------------------------
    Z_pca = face_model.pca(num_dim=2)

    fig3, ax3 = plt.subplots(figsize=(12, 10))
    ax3.scatter(Z_pca[:, 0], Z_pca[:, 1], c='red', s=10, alpha=0.5)
    ax3.set_title("2D PCA Embedding of Faces")

    for idx in sample_indices:
        img = face_model.images[idx].reshape(64, 64).T
        imagebox = OffsetImage(img, cmap='gray', zoom=0.5)
        ab = AnnotationBbox(imagebox, (Z_pca[idx, 0], Z_pca[idx, 1]), frameon=False)
        ax3.add_artist(ab)

    # ---------------------------------------------------------
    # Part 1b: Zoomed-in Nearest Neighbor Graph (Edges & Faces)
    # ---------------------------------------------------------
    # Create a new figure for this specific plot
    fig4, ax4 = plt.subplots(figsize=(8, 8))
    ax4.set_title("Local Nearest Neighbor Connections (Zoomed)")

    # 1. Pick a specific face index to be our "target" node
    target_idx = 150  # You can change this number to explore different parts of the manifold

    # 2. Find its direct neighbors from the Adjacency Matrix A
    # A[target_idx] > 0 returns the indices of images connected to the target
    connected_neighbors = np.where(A[target_idx] > 0)[0]

    # Limit to 3 or 4 neighbors so the plot isn't too cluttered
    neighbors = connected_neighbors[:4] 
    nodes_to_plot = [target_idx] + list(neighbors)

    # 3. Draw the edges (lines) between the target and its neighbors
    for neighbor in neighbors:
        x_coords = [Z_isomap[target_idx, 0], Z_isomap[neighbor, 0]]
        y_coords = [Z_isomap[target_idx, 1], Z_isomap[neighbor, 1]]
        # Draw a dashed line connecting the nodes
        ax4.plot(x_coords, y_coords, color='black', linestyle='--', linewidth=1.5, zorder=1)

    # 4. Overlay the face images on the nodes
    for idx in nodes_to_plot:
        img = face_model.images[idx].reshape(64, 64).T
        
        # Give the central target face a red border, and neighbors a blue border
        box_color = 'red' if idx == target_idx else 'blue'
        
        imagebox = OffsetImage(img, cmap='gray', zoom=0.6)
        ab = AnnotationBbox(imagebox, (Z_isomap[idx, 0], Z_isomap[idx, 1]), 
                            frameon=True, bboxprops=dict(edgecolor=box_color, linewidth=2))
        ax4.add_artist(ab)

    # 5. Crop/Zoom the axes to focus strictly on this local cluster
    # Get the min and max coordinates of our specific nodes
    min_x, max_x = Z_isomap[nodes_to_plot, 0].min(), Z_isomap[nodes_to_plot, 0].max()
    min_y, max_y = Z_isomap[nodes_to_plot, 1].min(), Z_isomap[nodes_to_plot, 1].max()

    # Add a buffer around the images so they don't get cut off by the edges of the plot
    buffer = 5.0 
    ax4.set_xlim(min_x - buffer, max_x + buffer)
    ax4.set_ylim(min_y - buffer, max_y + buffer)

    ax4.grid(True, linestyle=':', alpha=0.6)

    plt.show()