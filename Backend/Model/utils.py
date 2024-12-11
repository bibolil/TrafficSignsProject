import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Data Preparation
def load_data(image_dir, label_dir):
    data = []
    labels = []
    
    # Get all image filenames
    image_files = os.listdir(image_dir)
    
    for img_name in image_files:
        try:
            # Full image path
            img_path = os.path.join(image_dir, img_name)
            
            # Full label path (replace .jpg/.png with .txt)
            label_name = os.path.splitext(img_name)[0] + ".txt"
            label_path = os.path.join(label_dir, label_name)
            
            # Ensure the label file exists
            if not os.path.exists(label_path):
                print(f"Label file {label_name} not found for image {img_name}. Skipping.")
                continue
            
            # Load image
            image = Image.open(img_path).resize((30, 30))
            data.append(np.array(image))
            
            # Load label
            with open(label_path, "r") as f:
                label = int(f.read().strip())  # Assume single-line label files
            labels.append(label)
        
        except Exception as e:
            print(f"Error loading image {img_name} or label: {e}")
    
    return np.array(data), np.array(labels)

# Plotting Function
def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.show()