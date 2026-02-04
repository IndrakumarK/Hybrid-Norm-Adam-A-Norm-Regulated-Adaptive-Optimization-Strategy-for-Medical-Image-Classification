import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    BatchNormalization
)
from tensorflow.keras.models import Model


def build_vgg19(num_classes, input_shape=(240, 240, 3)):
    """
    Modified VGG19 for Brain Tumor Classification.

    Modifications over standard VGG19:
    1. Pre-trained ImageNet weights
    2. Early convolutional layers frozen
    3. Custom lightweight classification head
    4. Dropout + BatchNorm for improved generalization
    """

    # ------------------------------------------------
    # Base VGG19 Backbone
    # ------------------------------------------------
    base_model = VGG19(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )

    # ------------------------------------------------
    # Freeze Early Layers (Transfer Learning)
    # ------------------------------------------------
    for layer in base_model.layers[:15]:
        layer.trainable = False

    # ------------------------------------------------
    # Modified Classification Head
    # ------------------------------------------------
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)

    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)

    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)

    # ------------------------------------------------
    # Output Layer
    # ------------------------------------------------
    if num_classes == 1:
        outputs = Dense(1, activation="sigmoid")(x)
    else:
        outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(
        inputs=base_model.input,
        outputs=outputs,
        name="Modified_VGG19"
    )

    return model
