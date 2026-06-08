# -----------------------------------------------------------------------------
# NOTE: This file consists of 2 classes

# 1. EigenFacesResult - This class should not be modified. Gradescope will use the output of run() 
# method in this format.
# 2. EigenFaces - This is class which will implement the eigen faces algorithm and return the results.  
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# NOTE: This class should NOT be modified.
# Gradescope will depend on the structure of this class as defined. 
# -----------------------------------------------------------------------------
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


class EigenFacesResult:
    """    
    A structured container for storing the results of the EigenFaces computation.

    Attributes
    ----------
    subject_1_eigen_faces : np.ndarray
        A (6, a, b) array representing the top 6 eigenfaces for subject 1.
        A plt.imshow(map['subject_1_eigen_faces'][0]) should display first in a eigen face for subject 1

    subject_2_eigen_faces : np.ndarray
        A (6, a, b) array representing the top 6 eigenfaces for subject 2.
        A plt.imshow(map['subject_2_eigen_faces'][0]) should display first in a eigen face for subject 2

    s11 : float
        Projection residual of subject 1 test image on subject 1 eigenfaces.

    s12 : float
        Projection residual of subject 2 test image on subject 1 eigenfaces.

    s21 : float
        Projection residual of subject 1 test image on subject 2 eigenfaces.

    s22 : float
        Projection residual of subject 2 test image on subject 2 eigenfaces.
    """

    def __init__(
        self,
        subject_1_eigen_faces: np.ndarray,
        subject_2_eigen_faces: np.ndarray,
        s11: float,
        s12: float,
        s21: float,
        s22: float
    ):
        self.subject_1_eigen_faces = subject_1_eigen_faces
        self.subject_2_eigen_faces = subject_2_eigen_faces
        self.s11 = s11
        self.s12 = s12
        self.s21 = s21
        self.s22 = s22
        
# -----------------------------------------------------------------------------
# NOTE: Do not change the parameters / return types for pre defined methods.
# -----------------------------------------------------------------------------
class EigenFaces:
    """
    This class handles loading facial images for two subjects, computing eigenfaces
    via PCA, and evaluating projection residuals for test images.

    Methods
    -------
    run():
        Computes the eigenfaces for each subject and the projection residuals for test images.
    """

    def __init__(self, images_root_directory="data/yalefaces"):
        """
        Initializes the EigenFaces object and loads all relevant facial images from the specified directory.

        Parameters
        ----------
        images_root_directory : str
            The path to the root directory containing subject images.
        """
        self.images_root_directory = images_root_directory
        self.image_shape = None  # To be set after loading images

        s1_train_list = []
        s2_train_list = []
        self.s1_test = None
        self.s2_test = None

        # Traverse the directory and load images for both subjects
        for filename in sorted(os.listdir(images_root_directory)):
            filepath = os.path.join(images_root_directory, filename)
            if not os.path.isfile(filepath):
                continue
            try:
                # Load the image in grayscale mode
                img = Image.open(filepath).convert('L')

                # Downsample by a factor of 4 (e.g. reduce a 16x16 iamge to 4x4)
                img = img.resize((img.width // 4, img.height // 4), Image.Resampling.BILINEAR)

                # convert the reduce image into numpy array
                img_array = np.array(img, dtype=np.float64)
                
                # Store the reduced geometric shape of the image for later use
                if self.image_shape is None:
                    self.image_shape = img_array.shape
                
                #Flatten the image into a 1D array for PCA processing
                img_vector = img_array.flatten()

                #Identify subject and whether it's a test or train image based on filename and store accordingly
                if filename.startswith("subject01"):
                    if "test" in filename:
                        self.s1_test = img_vector
                    else:
                        s1_train_list.append(img_vector)
                elif filename.startswith("subject02"):
                    if "test" in filename:
                        self.s2_test = img_vector
                    else:
                        s2_train_list.append(img_vector)
            except Exception as e:
                # Silently skip files that cannot be opened as images
                continue
        
        self.s1_train = np.array(s1_train_list)
        self.s2_train = np.array(s2_train_list)

        #Fall mechanism if no explicitly named "test" files are found, we will assume the last image for each subject is the test image
        if self.s1_test is None and len(s1_train_list) > 0:
            self.s1_test = s1_train_list[-1]  # Use the last image as test if no explicit test image is found
            self.s1_train = self.s1_train[:-1]  # Remove the last image from training set
        if self.s2_test is None and len(s2_train_list) > 0:
            self.s2_test = s2_train_list[-1]  # Use the last image as test if no explicit test image is found
            self.s2_train = self.s2_train[:-1]  # Remove the last image from training set

    def compute_pca(self, train_data: np.ndarray, num_components: int = 6):
        """
        Computes the top eigenfaces using PCA for the given training data.

        Parameters
        ----------
        train_data : np.ndarray
            A (n_samples, n_features) array of training images, where each row is a flattened image.
        num_components : int
            The number of top eigenfaces to compute.

        Returns
        -------
        Returns the top num_components eigenvectors and the mean of the training data
        """
        # Calculate the mean face from the training data
        mean_face = np.mean(train_data, axis=0)

        #Mean center the data points by subtracting the mean face from each training image
        centered_data = train_data - mean_face

        # Compute covariance matrix and perform eigen decomposition
        covariance_matrix = np.cov(centered_data, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Sort eigenvalues and corresponding eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        
        # Extract top k eigenvectors and immediately transpose them.
        # Now, each row is a proper Eigenface. Shape becomes (k, features).
        top_eigenvectors = eigenvectors[:, sorted_indices[:num_components]].T
 
        return top_eigenvectors, mean_face

    def compute_residual(self, test_image: np.ndarray, eigen_vecs: np.ndarray, mean_face: np.ndarray) -> float:
        """
        Computes the projection residual of a test image onto the space defined by the eigenfaces.

        Parameters
        ----------
        test_image : np.ndarray
            A flattened test image vector.
        mean_face : np.ndarray
            The mean face vector computed from the training data.
        eigen_vecs : np.ndarray
            The matrix of eigen vectors (each column is an eigen vector).

        Returns
        -------
        float
            The projection residual of the test image onto the eigenface space.
        """
        # Mean center the test image by subtracting the mean face
        centered_test_image = test_image - mean_face

        # Project into the eigenspace (calculate specific coordinate weights)
        weights = eigen_vecs @ centered_test_image
        
        # Reconstruct the face image strictly from the eigenspace limits
        # Shape math: (4800, 6) @ (6,) = (4800,)
        reconstruction = eigen_vecs.T @ weights
        
        # Calculate the residual vector (the difference between original centered image and reconstruction)
        residual_vec = centered_test_image - reconstruction
        
        # Return the squared Euclidean norm of the residual
        return float(np.sum(residual_vec ** 2))
    
    def run(self) -> EigenFacesResult:
        """
        Computes eigenfaces for both subjects and projection residuals
        for test images using those eigenfaces.

        Returns
        -------
        EigenFacesResult
            Object containing eigenfaces and residuals for both subjects.
        """
        # Compute eigenfaces and mean face for subject 1
        eigenfaces_1_vecs, mean_face_1 = self.compute_pca(self.s1_train, num_components=6)
        # Compute eigenfaces and mean face for subject 2
        eigenfaces_2_vecs, mean_face_2 = self.compute_pca(self.s2_train, num_components=6)

        # Reshape eigenfaces to the original image dimensions for output
        eigenfaces_1 = eigenfaces_1_vecs.reshape((6, *self.image_shape))
        eigenfaces_2 = eigenfaces_2_vecs.reshape((6, *self.image_shape))

        # Compute projection residuals for subject 1 test image
        projection_residual_s11 = self.compute_residual(self.s1_test, eigenfaces_1_vecs, mean_face_1)
        projection_residual_s12 = self.compute_residual(self.s2_test, eigenfaces_1_vecs, mean_face_1)
        projection_residual_s21 = self.compute_residual(self.s1_test, eigenfaces_2_vecs, mean_face_2)
        projection_residual_s22 = self.compute_residual(self.s2_test, eigenfaces_2_vecs, mean_face_2)

        return EigenFacesResult(
            subject_1_eigen_faces=eigenfaces_1,
            subject_2_eigen_faces=eigenfaces_2,
            s11=projection_residual_s11,
            s12=projection_residual_s12,
            s21=projection_residual_s21,
            s22=projection_residual_s22
        )

if __name__ == "__main__":
    engine = EigenFaces(images_root_directory="data/yalefaces")
    
    # Execute the computation
    results = engine.run()
    
    # 1. Report the four projection residual scores
    print("--- Projection Residual Scores ---")
    print(f"s11 (Subject 1 Test on Subject 1 Eigenfaces): {results.s11:.4f}")
    print(f"s12 (Subject 2 Test on Subject 1 Eigenfaces): {results.s12:.4f}")
    print(f"s21 (Subject 1 Test on Subject 2 Eigenfaces): {results.s21:.4f}")
    print(f"s22 (Subject 2 Test on Subject 2 Eigenfaces): {results.s22:.4f}")
    
    # 2. Recognition Logic
    print("\n--- Recognition Results ---")
    subject1_pred = "Subject 1" if results.s11 < results.s21 else "Subject 2"
    subject2_pred = "Subject 2" if results.s22 < results.s12 else "Subject 1"
    
    print(f"Test Image 1 classified as: {subject1_pred}")
    print(f"Test Image 2 classified as: {subject2_pred}")

    # 3. Plotting the Eigenfaces
    def plot_eigenfaces(eigenfaces, title):
        fig, axes = plt.subplots(2, 3, figsize=(10, 6))
        fig.suptitle(title, fontsize=16)
        for i, ax in enumerate(axes.flatten()):
            # Using 'bicubic' interpolation to upscale the 4x4 image back to a visually smooth state
            ax.imshow(eigenfaces[i], cmap='gray', interpolation='bicubic')
            ax.set_title(f"Eigenface {i+1}")
            ax.axis('off')
        plt.tight_layout()

    plot_eigenfaces(results.subject_1_eigen_faces, "Top 6 Eigenfaces: Subject 1")
    plot_eigenfaces(results.subject_2_eigen_faces, "Top 6 Eigenfaces: Subject 2")
    
    plt.show()