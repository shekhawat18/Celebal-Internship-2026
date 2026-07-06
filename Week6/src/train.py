"""
Training Script for CNN Autoencoder
"""

import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

from src.data_loader import get_dataset
from src.noise import add_gaussian_noise
from src.model import build_autoencoder
from src.config import *


def train():

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("Loading Dataset...")
    x_train, x_test = get_dataset()

    print("Adding Gaussian Noise...")
    x_train_noisy = add_gaussian_noise(x_train)
    x_test_noisy = add_gaussian_noise(x_test)

    print("Building Model...")
    autoencoder = build_autoencoder()

    autoencoder.summary()

    early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

    print("\nTraining Started...\n")

    history = autoencoder.fit(
    x_train_noisy,
    x_train,
    epochs=15,
    batch_size=128,
    shuffle=True,
    validation_data=(x_test_noisy, x_test),
    verbose=1
)

    autoencoder.save(MODEL_PATH)

    print("\nModel Saved Successfully!")

    plt.figure(figsize=(8,5))

    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")

    plt.legend()

    plt.savefig(LOSS_PLOT)
    plt.show()

    decoded_imgs = autoencoder.predict(x_test_noisy)

    plt.figure(figsize=(18,6))

    n = 10

    for i in range(n):

        # Original

        ax = plt.subplot(3,n,i+1)
        plt.imshow(x_test[i].reshape(28,28), cmap="gray")
        plt.axis("off")

        # Noisy

        ax = plt.subplot(3,n,i+1+n)
        plt.imshow(x_test_noisy[i].reshape(28,28), cmap="gray")
        plt.axis("off")

        # Denoised

        ax = plt.subplot(3,n,i+1+2*n)
        plt.imshow(decoded_imgs[i].reshape(28,28), cmap="gray")
        plt.axis("off")

    plt.savefig(DENOISED_IMAGE)

    plt.show()

    print("\nTraining Completed Successfully!")


if __name__ == "__main__":
    train()