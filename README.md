# Martian Frost Detection Using CNN and Transfer Learning in HiRISE Images

## Overview
This project aims to develop classifiers to identify frost in Martian terrain images using the HiRISE dataset. It explores the effectiveness of a custom-built CNN+MLP model and compares it with transfer learning models (EfficientNetB0, ResNet50, and VGG16). The focus is on classifying images into 'frost' and 'background' categories and evaluating models based on precision, recall, and F1 score. 

## Dataset
The dataset contains 214 subframes and 119920 tiles, each labeled as 'frost' or 'background'. It is designed to study Mars' frost cycle and its impact on the planet's climate and surface evolution.

## Project Sections

### Data Exploration and Pre-processing
- Images and labels are provided in separate files, with images organized as 299x299 pixel tiles.
- Dataset is split into training, testing, and validation sets.

### Training CNN + MLP
- Image augmentation techniques are used for empirical regularization.
- A three-layer CNN followed by a dense MLP layer is trained with various hyperparameters and techniques like ReLU, softmax, batch normalization, dropout, and ADAM optimizer.
- The model is trained for at least 20 epochs with early stopping.

### Transfer Learning
- Pre-trained models (EfficientNetB0, ResNet50, VGG16) are used, training only the last fully connected layer.
- Similar image augmentation techniques as CNN+MLP.
- Model configurations include ReLU activation, softmax layer, batch normalization, dropout, and ADAM optimizer.
- Training is conducted for at least 10 epochs with early stopping.

## Files Description
- `Martian_Frost_Detection_Cici_Chang.ipynb`: Main Jupyter notebook containing the implementation of data processing, model training, evaluation, and result visualization.

## Requirements
- Python 3.x
- Libraries: Keras, OpenCV, sklearn, pandas, numpy, matplotlib (refer to the notebook for detailed requirements and installation commands)

## Usage
To run the notebook:
1. Ensure Python 3.x is installed on your system.
2. Install the required libraries.
3. Open `Martian_Frost_Detection_Cici_Chang.ipynb` in a Jupyter environment.
4. Execute the notebook cells sequentially to view the analysis and results.

## Conclusions
The project provides insights into the effectiveness of CNNs and transfer learning methods in processing extraterrestrial terrain images, demonstrating advanced machine learning techniques in planetary science.
