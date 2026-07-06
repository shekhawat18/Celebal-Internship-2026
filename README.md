# Autoencoder for Image Denoising using CNN (MNIST)

## Overview

This project demonstrates image denoising using a Convolutional Neural Network (CNN) Autoencoder on the MNIST handwritten digit dataset. Artificial Gaussian noise is added to the images, and the autoencoder is trained to reconstruct the original clean images.

This project was completed as part of the **Celebal Technologies Data Science Internship - Week 6 Assignment**.

---

## Objective

- Load and preprocess the MNIST dataset.
- Add Gaussian noise to clean images.
- Design and train a CNN Autoencoder.
- Reconstruct clean images from noisy inputs.
- Evaluate reconstruction quality using loss curves and image comparison.

---

## Dataset

- Dataset: MNIST Handwritten Digits
- Training Images: 60,000
- Test Images: 10,000
- Image Size: 28 × 28 pixels
- Channels: Grayscale

Dataset Source:
TensorFlow/Keras Datasets

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn

---

## CNN Autoencoder Architecture

### Encoder

- Conv2D (32 Filters)
- MaxPooling2D
- Conv2D (64 Filters)
- MaxPooling2D

### Bottleneck

- Conv2D (64 Filters)

### Decoder

- UpSampling2D
- Conv2D (64 Filters)
- UpSampling2D
- Conv2D (32 Filters)
- Conv2D (1 Filter, Sigmoid)

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Binary Crossentropy |
| Batch Size | 128 |
| Epochs | 30 |
| Early Stopping | Enabled |

---

## Results
### Training Performance

The training and validation loss decreased consistently throughout training, indicating that the CNN Autoencoder successfully learned to reconstruct clean images from noisy inputs without significant overfitting.

![Training Loss](outputs/training_loss.png)

The CNN Autoencoder successfully learns to reconstruct noisy MNIST images.

Outputs include:

- Original Images
- Noisy Images
- Reconstructed Images
- Training & Validation Loss Curve

---

## Folder Structure

```
Week6/
│
├── Autoencoder_Image_Denoising.ipynb
├── README.md
├── requirements.txt
├── models/
│     autoencoder.keras
├── outputs/
│     loss_curve.png
│     denoised_images.png
```

---

## Future Improvements

- Train deeper Autoencoders
- Use Skip Connections
- Apply to CIFAR-10 dataset
- Train on real-world noisy images

---

## Author

Harsh Vardhan Shekhawat

Celebal Technologies Data Science Internship
