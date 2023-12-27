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

# Modeling Insights
- <strong>For precision, transfer learning models and the CNN model are the same. This indicates that the ability of each model to correctly identify the positive class is similar.</strong>
- <strong>For recall, the CNN model has a significantly lower recall score compared to the transfer learning models and transfer learning models achieved perfect recall scores, which indicates that transfer learning models were able to identify all true positive cases in the daset.</strong>
- <strong>Similarly, the transfer learning models have much higher f1 score, which means the transfer learning models are more balanced in terms of both precision and recall.</strong>


## Explanation
- <strong>Pre-trained models such as transfer learning models often generalize better, especially when the amount of training data is limited. However, CNN model is more customized, leading to being outperformed by transfer learning model in recall and F1 score.</strong>
- <strong>The architectures of transfer learning models (ResNet50, EfficientNetB0, and VGG16) are more sophisticated compared to a typical CNN + MLP model, which allows them to capture more nuanced patterns in the data, contributing to a better performance.</strong>

## Model Performance Metrics

| Model           | Precision | Recall   | F1 Score |
|-----------------|-----------|----------|----------|
| CNN             | 0.656     | 0.391    | 0.490    |
| ResNet50        | 0.655     | 1.0      | 0.792    |
| EfficientNetB0  | 0.655     | 1.0      | 0.792    |
| VGG16           | 0.655     | 1.0      | 0.792    |
