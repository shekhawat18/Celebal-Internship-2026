# Week 4 - CIFAR-10 Image Classification using ANN and CNN

## Objective
Build and compare Artificial Neural Network (ANN) and Convolutional Neural Network (CNN) models on the CIFAR-10 dataset and analyze their performance.

## Dataset
CIFAR-10 is a popular image classification dataset containing 60,000 color images of size 32×32 belonging to 10 classes.

- Training Images: 50,000
- Test Images: 10,000
- Classes: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

## Models Implemented

### 1. Artificial Neural Network (ANN)
- Fully Connected Dense Layers
- ReLU Activation
- Dropout Regularization
- Adam Optimizer

### 2. Convolutional Neural Network (CNN)
- Convolution Layers
- Max Pooling Layers
- Dropout Regularization
- Adam Optimizer

## Results

| Model | Test Accuracy |
|---------|------------|
| ANN | 43.09% |
| CNN | 69.13% |

## Key Observations

- ANN treats images as flattened vectors and cannot effectively capture spatial features.
- CNN preserves spatial information through convolution operations.
- CNN significantly outperformed ANN on the CIFAR-10 dataset.
- CNN achieved approximately 26% higher accuracy than ANN.

## Conclusion

This project demonstrates that Convolutional Neural Networks are more effective for image classification tasks than traditional Artificial Neural Networks. CNNs extract meaningful image features and achieve better classification performance on the CIFAR-10 dataset.
