#fetch data from MNIST dataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from tqdm import tqdm  # Import tqdm for progress bar

# Function to wrap around the fetch_openml method to show progress
def fetch_mnist_with_progress():
    # You can use tqdm to show a wait message while the dataset is being fetched
    print("Downloading MNIST dataset... Please wait.")
    
    # Fetch dataset with progress using a custom method
    with tqdm(total=1, desc="Downloading MNIST", ncols=100) as pbar:
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, parser="auto") # 784 features in the MNIST dataset
        pbar.update(1)  # Mark the download as complete
    return X, y

# Fetch data with progress indication
X, y = fetch_mnist_with_progress()

# Convert to numpy arrays and scale
X = np.array(X) / 255.0  # Scale the pixel values to [0, 1]
y = np.array(y, dtype=np.int8)  # Convert labels to int8

# Split into train and test
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# Check dataset size
print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Show the first 3 images
plt.figure(figsize=(20, 4))
for index, (image, label) in enumerate(zip(X_train[0:3], y_train[0:3])):
    plt.subplot(1, 3, index + 1)
    plt.imshow(image.reshape(28, 28), cmap=plt.cm.gray)  # Reshape to 28x28 for visualization
    plt.title(f"Label: {label}", fontsize=20)

plt.show()
