# Part 1: Data Preprocessing
# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image
import random

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Step 1: Load the dataset (using CIFAR-10)
def load_cifar10_data():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Convert class vectors to binary class matrices (one-hot encoding)
    num_classes = 10
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    
    # Get class names for CIFAR-10
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    return x_train, y_train, x_test, y_test, class_names

# Load the data
x_train, y_train, x_test, y_test, class_names = load_cifar10_data()

# Step 2: Exploratory Data Analysis (EDA)
def perform_eda(x_train, y_train, class_names):
    print("Dataset Information:")
    print(f"Training data shape: {x_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    
    # Display data types and range
    print(f"Data type: {x_train.dtype}")
    print(f"Min pixel value: {x_train.min()}")
    print(f"Max pixel value: {x_train.max()}")
    
    # Calculate class distribution
    class_counts = np.sum(y_train, axis=0)
    
    # Plot class distribution
    plt.figure(figsize=(12, 5))
    plt.bar(class_names, class_counts)
    plt.title('Class Distribution in Training Set')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    
    # Display sample images
    plt.figure(figsize=(12, 10))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        
        # Find an example of this class
        class_indices = np.where(np.argmax(y_train, axis=1) == i)[0]
        sample_idx = np.random.choice(class_indices)
        
        plt.imshow(x_train[sample_idx])
        plt.xlabel(class_names[i])
    plt.tight_layout()
    plt.savefig('sample_images.png')
    
    # Display image statistics
    pixel_means = np.mean(x_train, axis=(0, 1, 2))
    pixel_stds = np.std(x_train, axis=(0, 1, 2))
    
    print("\nPixel Statistics:")
    print(f"Mean values per channel: {pixel_means}")
    print(f"Standard deviation per channel: {pixel_stds}")
    
    # Plot pixel value distribution
    plt.figure(figsize=(15, 5))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        channel_data = x_train[:, :, :, i].flatten()
        plt.hist(channel_data, bins=50, alpha=0.7)
        plt.title(f'Channel {i} Pixel Distribution')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('pixel_distribution.png')
    
    return pixel_means, pixel_stds

# Perform EDA
pixel_means, pixel_stds = perform_eda(x_train, y_train, class_names)

# Step 3: Preprocess images (resize, normalize, etc.)
def preprocess_data(x_train, y_train, x_test, y_test, pixel_means, pixel_stds):
    # Create a validation set from the training data
    x_train_final, x_val, y_train_final, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42
    )
    
    # Normalize pixel values to the range [0, 1]
    x_train_norm = x_train_final.astype('float32') / 255.0
    x_val_norm = x_val.astype('float32') / 255.0
    x_test_norm = x_test.astype('float32') / 255.0
    
    # Alternative: Standardize (zero mean, unit variance)
    # x_train_std = (x_train_final.astype('float32') - pixel_means) / pixel_stds
    # x_val_std = (x_val.astype('float32') - pixel_means) / pixel_stds
    # x_test_std = (x_test.astype('float32') - pixel_means) / pixel_stds
    
    print("\nPreprocessed Data Shapes:")
    print(f"Training set: {x_train_norm.shape}")
    print(f"Validation set: {x_val_norm.shape}")
    print(f"Test set: {x_test_norm.shape}")
    
    return x_train_norm, y_train_final, x_val_norm, y_val, x_test_norm, y_test

# Preprocess the data
x_train_norm, y_train_final, x_val_norm, y_val, x_test_norm, y_test = preprocess_data(
    x_train, y_train, x_test, y_test, pixel_means, pixel_stds
)

# Step 4: Setup data augmentation
def setup_data_augmentation():
    # Create an ImageDataGenerator for data augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=15,  # randomly rotate images by up to 15 degrees
        width_shift_range=0.1,  # randomly shift images horizontally by up to 10%
        height_shift_range=0.1,  # randomly shift images vertically by up to 10%
        horizontal_flip=True,  # randomly flip images horizontally
        zoom_range=0.1,  # randomly zoom images by up to 10%
        fill_mode='nearest'  # strategy for filling in newly created pixels
    )
    
    # Display examples of augmented images
    plt.figure(figsize=(12, 8))
    
    # Take a sample image
    sample_idx = np.random.randint(0, x_train_norm.shape[0])
    sample_image = x_train_norm[sample_idx].reshape(1, 32, 32, 3)
    sample_label = np.argmax(y_train_final[sample_idx])
    
    # Generate augmented images
    augmented_iterator = train_datagen.flow(sample_image, batch_size=1)
    
    # Plot original and augmented images
    plt.subplot(3, 3, 1)
    plt.imshow(sample_image[0])
    plt.title(f'Original: {class_names[sample_label]}')
    plt.axis('off')
    
    for i in range(8):
        plt.subplot(3, 3, i+2)
        batch = next(augmented_iterator)
        plt.imshow(batch[0])
        plt.title(f'Augmented {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('augmented_images.png')
    
    return train_datagen

# Setup data augmentation
train_datagen = setup_data_augmentation()

# Step 5: Convert to TensorFlow datasets for efficient training
def create_tf_datasets(x_train_norm, y_train_final, x_val_norm, y_val, x_test_norm, y_test, train_datagen):
    # Batch size for training
    batch_size = 64
    
    # Create training dataset with data augmentation
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train_norm, y_train_final))
    train_dataset = train_dataset.shuffle(buffer_size=1000)
    train_dataset = train_dataset.batch(batch_size)
    
    # Create validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val_norm, y_val))
    val_dataset = val_dataset.batch(batch_size)
    
    # Create test dataset
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test_norm, y_test))
    test_dataset = test_dataset.batch(batch_size)
    
    print("\nTensorFlow Datasets Created:")
    print(f"Training steps per epoch: {len(x_train_norm) // batch_size}")
    print(f"Validation steps per epoch: {len(x_val_norm) // batch_size}")
    print(f"Test steps: {len(x_test_norm) // batch_size}")
    
    return train_dataset, val_dataset, test_dataset, batch_size

# Create TensorFlow datasets
train_dataset, val_dataset, test_dataset, batch_size = create_tf_datasets(
    x_train_norm, y_train_final, x_val_norm, y_val, x_test_norm, y_test, train_datagen
)

# Save preprocessed data for model training
np.save('x_train_norm.npy', x_train_norm)
np.save('y_train_final.npy', y_train_final)
np.save('x_val_norm.npy', x_val_norm)
np.save('y_val.npy', y_val)
np.save('x_test_norm.npy', x_test_norm)
np.save('y_test.npy', y_test)

# Save class names
with open('class_names.txt', 'w') as f:
    for class_name in class_names:
        f.write(f"{class_name}\n")

print("\nData preprocessing complete. Files saved for model training.")

# Function to run the entire preprocessing pipeline
def run_preprocessing_pipeline():
    print("Running complete preprocessing pipeline...")
    
    # Step 1: Load dataset
    x_train, y_train, x_test, y_test, class_names = load_cifar10_data()
    
    # Step 2: Perform EDA
    pixel_means, pixel_stds = perform_eda(x_train, y_train, class_names)
    
    # Step 3: Preprocess data
    x_train_norm, y_train_final, x_val_norm, y_val, x_test_norm, y_test = preprocess_data(
        x_train, y_train, x_test, y_test, pixel_means, pixel_stds
    )
    
    # Step 4: Setup data augmentation
    train_datagen = setup_data_augmentation()
    
    # Step 5: Create TensorFlow datasets
    train_dataset, val_dataset, test_dataset, batch_size = create_tf_datasets(
        x_train_norm, y_train_final, x_val_norm, y_val, x_test_norm, y_test, train_datagen
    )
    
    return {
        'x_train': x_train_norm,
        'y_train': y_train_final,
        'x_val': x_val_norm,
        'y_val': y_val,
        'x_test': x_test_norm,
        'y_test': y_test,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'batch_size': batch_size,
        'class_names': class_names,
        'datagen': train_datagen
    }

# Run the preprocessing pipeline if this file is executed directly
if __name__ == "__main__":
    processed_data = run_preprocessing_pipeline()
    print("Preprocessing complete!")