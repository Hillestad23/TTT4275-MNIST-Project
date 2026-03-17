import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = sio.loadmat('data_all.mat')
train_images = data['trainv']
train_labels = data['trainlab'].flatten()
test_images = data['testv']
test_labels = data['testlab'].flatten()

# Example: Display a sample image
plt.imshow(train_images[0].reshape(28, 28), cmap='gray')
plt.title(f'Label: {train_labels[0]}')
plt.show()

print(f"Training set: {train_images.shape}")
print(f"Test set: {test_images.shape}")
