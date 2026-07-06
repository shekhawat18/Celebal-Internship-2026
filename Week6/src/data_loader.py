"""
Data Loader Module

Loads and preprocesses the MNIST dataset.
"""

import numpy as np
from tensorflow.keras.datasets import mnist

from .config import INPUT_SHAPE


def load_data():
    """
    Load MNIST dataset.
    """

    (x_train, _), (x_test, _) = mnist.load_data()

    return x_train, x_test


def preprocess_data(x_train, x_test):
    """
    Normalize and reshape images.
    """

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.reshape(x_train, (-1, *INPUT_SHAPE))
    x_test = np.reshape(x_test, (-1, *INPUT_SHAPE))

    return x_train, x_test


def get_dataset():
    """
    Complete preprocessing pipeline.
    """

    x_train, x_test = load_data()

    x_train, x_test = preprocess_data(
        x_train,
        x_test
    )

    return x_train, x_test


if __name__ == "__main__":

    train, test = get_dataset()

    print(f"Training Images : {train.shape}")
    print(f"Testing Images  : {test.shape}")