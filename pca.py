import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# Start timer
start_time = time.time()
print(start_time)

# Load images and convert to grayscale
def load_images(folder_path, size=(64, 64)):
    images = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is not None:
            img = cv2.resize(img, size)  # Resize image
            images.append(img)
    return images

# Function to flatten images to vectors
def flatten_images(images):
    flattened_images = []
    for img in images:
        flattened_images.append(img.flatten())
    return np.array(flattened_images)

# Load the dataset of images (cats, dogs, wild animals)
train_folder_path = r"KostasEdition/afhq/train/cat"
images = load_images(train_folder_path)

# Convert images to grayscale
grayscale_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

# Flatten and convert images to numpy array (vectors)
flattened_resized_images = flatten_images(grayscale_images)

# Convert images to float64 data type
flattened_resized_images = flattened_resized_images.astype('float64')


# Implement PCA
class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None #The Q Matix
        self.mean = None


    def fit(self, X):
        # Convert to float for calculations
        X = X.astype('float64')

        # Mean centering
        self.mean = np.mean(X, axis=0)
        X -= self.mean

        # Calculate covariance matrix
        cov_matrix = np.cov(X.T)

        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Sort eigenvalues and eigenvectors
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        inv_2_eigenvalues = np.sqrt(np.linalg.inv(np.diag(eigenvalues)))

        # Store first n_components eigenvectors
        self.components = inv_2_eigenvalues @ eigenvectors.T [:self.n_components]

    def transform(self, X):
        # Mean centering
        X -= self.mean

        # Project data onto the components
        return np.dot(X, self.components.T)


    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# The number of components
n_components = 100 

# Apply PCA to images
pca = PCA(n_components=n_components)
transformed_images = pca.fit_transform(flattened_resized_images)
print("Applied PCA to images")

# Reconstruct images using the transformed data
reconstructed_images = np.dot(transformed_images, pca.components) + pca.mean
print("Reconstructed images using the transformed data")

# Reshape reconstructed images to their original shapes
reconstructed_images = reconstructed_images.reshape(len(images), *grayscale_images[0].shape)
print("Reshaped reconstructed images to their original shapes")

# Ensure reconstructed images are of appropriate data type (e.g., convert to uint8)
reconstructed_images = np.real(reconstructed_images).astype('uint8')
print("Ensure reconstructed images are of appropriate data type (e.g., convert to uint8)")

# Plot original and reconstructed images
fig, axes = plt.subplots(2, 5, figsize=(10, 6))
for i in range(5):
    axes[0, i].imshow(grayscale_images[i], cmap='gray')
    axes[0, i].axis('off')
    axes[0, i].set_title('Original')

    axes[1, i].imshow(reconstructed_images[i], cmap='gray')
    axes[1, i].axis('off')
    axes[1, i].set_title('Reconstructed')

plt.tight_layout()
plt.show()
