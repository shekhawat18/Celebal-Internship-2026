"""
CNN Autoencoder Model
"""

from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D
)

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from .config import INPUT_SHAPE, LEARNING_RATE


def build_autoencoder():

    input_img = Input(shape=INPUT_SHAPE)

    # ---------------- Encoder ---------------- #

    x = Conv2D(
        32,
        (3,3),
        activation="relu",
        padding="same"
    )(input_img)

    x = MaxPooling2D(
        (2,2),
        padding="same"
    )(x)

    x = Conv2D(
        16,
        (3,3),
        activation="relu",
        padding="same"
    )(x)

    encoded = MaxPooling2D(
        (2,2),
        padding="same"
    )(x)

    # ---------------- Decoder ---------------- #

    x = Conv2D(
        16,
        (3,3),
        activation="relu",
        padding="same"
    )(encoded)

    x = UpSampling2D((2,2))(x)

    x = Conv2D(
        32,
        (3,3),
        activation="relu",
        padding="same"
    )(x)

    x = UpSampling2D((2,2))(x)

    decoded = Conv2D(
        1,
        (3,3),
        activation="sigmoid",
        padding="same"
    )(x)

    autoencoder = Model(
        input_img,
        decoded
    )

    autoencoder.compile(
        optimizer=Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="mse"
    )

    return autoencoder


if __name__ == "__main__":

    model = build_autoencoder()

    model.summary()