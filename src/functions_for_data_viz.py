# Here, I will save some useful functions for the data visualization

import os
import numpy as np
import random

import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

# Function to collect file paths
def collect_file_paths(folder_path):

    file_paths = []

    for root, dirs, files in os.walk(folder_path):

        for file in files:

            if file.endswith('.jpeg') or file.endswith('.jpg'):
                file_paths.append(os.path.join(root, file))

    return file_paths

# Function to visualize the total numbers of the two classes
def plot_total_images(train_NORMAL_paths, train_PNEUMONIA_paths, 
                            val_NORMAL_paths, val_PNEUMONIA_paths, 
                            test_NORMAL_paths, test_PNEUMONIA_paths):
    
    total_NORMAL = sum([len(train_NORMAL_paths), len(val_NORMAL_paths), len(test_NORMAL_paths)])
    total_PNEUMONIA = sum([len(train_PNEUMONIA_paths), len(val_PNEUMONIA_paths), len(test_PNEUMONIA_paths)])

    percent_NORMAL = (total_NORMAL / (total_NORMAL + total_PNEUMONIA)) * 100
    percent_PNEUMONIA = (total_PNEUMONIA / (total_NORMAL + total_PNEUMONIA)) * 100

    classes = ['NORMAL', 'PNEUMONIA']
    counts = [total_NORMAL, total_PNEUMONIA]

    # Create bar chart
    plt.figure(figsize=(8, 6))
    bars = plt.bar(classes, counts, color=['skyblue', 'salmon'])

    plt.xlabel('Category')
    plt.ylabel('Total Number of Images')
    plt.title('Total Number of NORMAL and PNEUMONIA Images Across All Folders')
    legend_handles = [Patch(color='skyblue', label=f'NORMAL: {percent_NORMAL:.2f}%'),
                    Patch(color='salmon', label=f'PNEUMONIA: {percent_PNEUMONIA:.2f}%')]
    plt.legend(handles=legend_handles, loc='upper left')
    plt.show()


# Function to visualize the ditrubution of images in train, val and test
def plot_image_distribution(train_NORMAL_paths, train_PNEUMONIA_paths, 
                            val_NORMAL_paths, val_PNEUMONIA_paths, 
                            test_NORMAL_paths, test_PNEUMONIA_paths):
    
    light_blue = 'lightblue'
    light_orange = 'lightcoral'
    light_green = 'lightgreen'

    train_counts = [len(train_NORMAL_paths), len(train_PNEUMONIA_paths)]
    val_counts = [len(val_NORMAL_paths), len(val_PNEUMONIA_paths)]
    test_counts = [len(test_NORMAL_paths), len(test_PNEUMONIA_paths)]

    classes = ['NORMAL', 'PNEUMONIA']

    all_counts = np.array([train_counts, val_counts, test_counts])

    # Create a horizontal bar plot
    plt.figure(figsize=(10, 6))
    bar_width = 0.2
    bar_offsets = np.arange(len(classes))

    for i, dataset in enumerate(['Train', 'Validation', 'Test']):

        max_normal = all_counts[i, 0]
        max_pneumonia = all_counts[i, 1]
        legend_label = f'(N: {max_normal}, P: {max_pneumonia})'
        color = [light_blue, light_orange, light_green][i] 

        plt.barh(bar_offsets + i * bar_width, all_counts[i], height=bar_width, label=f'{dataset} {legend_label}',
                 color=color)

    plt.xlabel('Number of Images (log scale)')
    plt.ylabel('Labels')
    plt.title('Distribution of Images in Train, Validation, and Test Datasets')
    plt.yticks(bar_offsets + bar_width, classes)
    plt.xscale('log')
    plt.legend()
    plt.show()

# Function to visualize 9 random images from a chosen path
def visualize_random_images(paths, folder_name, num_images=9):

    random_paths = random.sample(paths, min(len(paths), num_images))

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    for i, path in enumerate(random_paths):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        ax = axes[i // 3, i % 3]
        ax.imshow(img)
        ax.axis('off')

    plt.suptitle(f'Random Images from {folder_name} Folder', fontsize=16) 
    plt.tight_layout()
    plt.show()


# Function to visualize 9 random images with Canny's filter from a chosen path
def visualize_random_images_with_canny(paths, folder_name, num_images=9):

    random_paths = random.sample(paths, min(len(paths), num_images))

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    # Plot the images with Canny filter applied
    for i, path in enumerate(random_paths):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        edges = cv2.Canny(img, 60, 70)  
        ax = axes[i // 3, i % 3]
        ax.imshow(edges, cmap='viridis')
        ax.axis('off')

    plt.suptitle(f'Random Images with Canny Filter from {folder_name} Folder', fontsize=16)  
    plt.tight_layout()
    plt.show()




# Function to visualize histograms for how widths and heights are distributed for each class in a choosen folder:
def visualize_image_sizes_per_class(paths_NORMAL, paths_PNEUMONIA, folder_name):

    dimensions_NORMAL = []
    for path in paths_NORMAL:
        img = cv2.imread(path)
        if img is not None:
            height, width, _ = img.shape
            dimensions_NORMAL.append((width, height))

    dimensions_PNEUMONIA = []
    for path in paths_PNEUMONIA:
        img = cv2.imread(path)
        if img is not None:
            height, width, _ = img.shape
            dimensions_PNEUMONIA.append((width,height))
    
    min_width_NORMAL = min(dim[0] for dim in dimensions_NORMAL)
    max_width_NORMAL = max(dim[0] for dim in dimensions_NORMAL)
    min_height_NORMAL = min(dim[1] for dim in dimensions_NORMAL)
    max_height_NORMAL = max(dim[1] for dim in dimensions_NORMAL)

    min_width_PNEUMONIA = min(dim[0] for dim in dimensions_PNEUMONIA)
    max_width_PNEUMONIA = max(dim[0] for dim in dimensions_PNEUMONIA)
    min_height_PNEUMONIA = min(dim[1] for dim in dimensions_PNEUMONIA)
    max_height_PNEUMONIA = max(dim[1] for dim in dimensions_PNEUMONIA)

    plt.figure(figsize=(12, 6))
    # Histogram for NORMAL class
    if dimensions_NORMAL:
        widths_NORMAL, heights_NORMAL = zip(*dimensions_NORMAL)
        plt.hist(widths_NORMAL, bins=30, alpha=0.5, color='blue', label=f'NORMAL Width (min: {min_width_NORMAL}, max: {max_width_NORMAL})')
        plt.hist(heights_NORMAL, bins=30, alpha=0.5, color='green', label=f'NORMAL Height (min: {min_height_NORMAL}, max: {max_height_NORMAL})')

    # Histogram for PNEUMONIA class
    if dimensions_PNEUMONIA:
        widths_PNEUMONIA, heights_PNEUMONIA = zip(*dimensions_PNEUMONIA)
        plt.hist(widths_PNEUMONIA, bins=30, alpha=0.5, color='red', label=f'PNEUMONIA Width (min: {min_width_PNEUMONIA}, max: {max_width_PNEUMONIA})')
        plt.hist(heights_PNEUMONIA, bins=30, alpha=0.5, color='orange', label=f'PNEUMONIA Height (min: {min_height_PNEUMONIA}, max: {max_height_PNEUMONIA})')

    plt.xlabel('Dimension')
    plt.ylabel('Frequency')
    plt.title(f'Image Sizes in {folder_name} Folder')
    plt.legend()
    plt.show()
