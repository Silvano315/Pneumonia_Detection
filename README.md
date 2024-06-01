# Pneumonia_Detection

1. [Introduction](#introduction)
2. [Dataset](#dataset)
   2.1. [Kaggle](#kaggle)
   2.2. [Description](#description)
3. [Methods](#methods)
   3.1. [Data Visualization](#data-visualization)
   3.2. [Deep Learning models](#deep-learning-models)
       3.2.1. [Model from scratch](#model-from-scratch)
       3.2.2. [Transfer Learning](#transfer-learning)
   3.3. [Grad-Cam](#grad-cam)
4. [Main Results](#main-results)
5. [Docker App](#docker-app)
6. [References](#references)

## 1. Introduction

This project aims to study a public dataset on pneumonia detection based on a binary classification problem. I will perform a preprocessing phase, create a deep learning model from scratch, compare it with a transfer learning technique and complete this study with a grad cam integration. At the end of this case study, I created a docker app to upload a test function and receive prediction with heatmap from grad-cam.

## 2. Dataset

### 2.1 Kaggle

Dataset taken from: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
It is composed by three main folders: train, test, val. Each of them contains two folders: NORMAL and PNEUMONIA. In [images](images/) there are just few examples to see the images I used. You can find the entire dataset at the Kaggle link.

### 2.2 Description

Detailed description of the dataset, including its features, labels, and any preprocessing steps.

## 3. Methods

### 3.1 Data Visualization

Explanation of data visualization techniques used to explore the dataset.

### 3.2 Deep Learning models

Overview of the deep learning models employed in the project.

#### 3.2.1 Model from scratch

Description of the custom deep learning model built from scratch.

#### 3.2.2 Transfer Learning

Explanation of the transfer learning approach adopted in the project.

### 3.3 Grad-Cam

Overall, this function takes an input image array, computes the gradient of the predicted class score with respect to the activations of the last convolutional layer, and generates a Grad-CAM heatmap highlighting important regions in the image for the predicted class.

In the context of Grad-CAM, the gradient of the predicted class score with respect to the output of the last convolutional layer tells us how much each activation in that layer influences the predicted class score. This information is used to weigh the activations, highlighting the parts of the image that are most important for the prediction.

## 4. Main Results

Summary of the main results obtained from the project, including model performance metrics and visualizations.

## 5. Docker App

Details about the Docker app created for deploying the project, including how to run it locally and any dependencies.

## References