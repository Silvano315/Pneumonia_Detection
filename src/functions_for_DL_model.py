import os
import platform
import sys
import numpy as np
from matplotlib import pyplot as plt
from src.constants import SIZE, batch_size

import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout


# Data preprocessing steps for each image folders: train, val, test
def image_preprocessing(folder_path):

    datagen = ImageDataGenerator(rescale=1./255)

    if folder_path == 'images/train':

        generator = datagen.flow_from_directory(
        folder_path,
        target_size = (SIZE, SIZE),
        batch_size = batch_size,
        class_mode = 'binary',
        shuffle = True
        )

    else:

        generator = datagen.flow_from_directory(
        folder_path,
        target_size = (SIZE, SIZE),
        batch_size = 1,
        class_mode = 'binary',
        shuffle = False
        )

    return generator


def scratch_model():

    model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(SIZE, SIZE, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
    ])
    
    return model


def plot_history(history):

    # Calculate mean and standard deviation for accuracy and loss
    train_acc_mean = np.mean(history.history['accuracy'])
    train_acc_std = np.std(history.history['accuracy'])
    val_acc_mean = np.mean(history.history['val_accuracy'])
    val_acc_std = np.std(history.history['val_accuracy'])
    train_loss_mean = np.mean(history.history['loss'])
    train_loss_std = np.std(history.history['loss'])
    val_loss_mean = np.mean(history.history['val_loss'])
    val_loss_std = np.std(history.history['val_loss'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))

    # Plot training and validation accuracy
    ax1.plot(history.history['accuracy'], label=f'Training Accuracy ({train_acc_mean:.2f} ± {train_acc_std:.2f})')
    ax1.plot(history.history['val_accuracy'], label=f'Validation Accuracy ({val_acc_mean:.2f} ± {val_acc_std:.2f})')
    ax1.set_title('Training and Validation Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    # Plot training and validation loss
    ax2.plot(history.history['loss'], label=f'Training Loss ({train_loss_mean:.2f} ± {train_loss_std:.2f})')
    ax2.plot(history.history['val_loss'], label=f'Validation Loss ({val_loss_mean:.2f} ± {val_loss_std:.2f})')
    ax2.set_title('Training and Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.show()