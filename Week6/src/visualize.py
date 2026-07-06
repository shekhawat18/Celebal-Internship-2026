"""
Visualization Utilities
"""

import matplotlib.pyplot as plt


def plot_results(original, noisy, reconstructed, save_path="outputs/denoised_images.png"):
    """
    Display Original, Noisy and Reconstructed images.
    """

    n = 10

    plt.figure(figsize=(18, 6))

    for i in range(n):

        # Original
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(original[i].reshape(28, 28), cmap="gray")
        plt.title("Original")
        plt.axis("off")

        # Noisy
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(noisy[i].reshape(28, 28), cmap="gray")
        plt.title("Noisy")
        plt.axis("off")

        # Reconstructed
        ax = plt.subplot(3, n, i + 1 + (2 * n))
        plt.imshow(reconstructed[i].reshape(28, 28), cmap="gray")
        plt.title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()