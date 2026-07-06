"""
Configuration file for Autoencoder Image Denoising Project
"""

# Dataset Parameters
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
CHANNELS = 1

INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)

# Training Parameters
EPOCHS = 20
BATCH_SIZE = 64

LEARNING_RATE = 0.001

# Noise
NOISE_FACTOR = 0.3

# Output Paths
MODEL_PATH = "models/autoencoder.keras"

LOSS_PLOT = "outputs/training_loss.png"

ORIGINAL_IMAGE = "outputs/original_images.png"

NOISY_IMAGE = "outputs/noisy_images.png"

DENOISED_IMAGE = "outputs/denoised_images.png"

RANDOM_SEED = 42