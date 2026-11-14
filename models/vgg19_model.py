"""
VGG19-based binary classifier for brain tumor MRI classification.
"""

import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model


def build_vgg19(
    num_classes=1,
    input_shape=(240, 240, 3),
    weights="imagenet",
    freeze_backbone=True,
):
    """
    Build a VGG19 model for binary brain tumor classification.

    Args:
        num_classes: Number of output units. Must be 1 for binary classification.
        input_shape: Input MRI image dimensions.
        weights: Initial weights ('imagenet' or None).
        freeze_backbone: Whether to freeze the pretrained VGG19 layers.

    Returns:
        A compiled-independent Keras Model.
    """

    if num_classes != 1:
        raise ValueError(
            "Binary classification requires num_classes=1."
        )

    base_model = VGG19(
        weights=weights,
        include_top=False,
        input_shape=input_shape,
    )

    for layer in base_model.layers:
        layer.trainable = not freeze_backbone

    x = Flatten(name="flatten_features")(base_model.output)

    x = Dense(
        64,
        activation="relu",
        name="classifier_dense",
    )(x)

    output = Dense(
        1,
        activation="sigmoid",
        name="binary_output",
    )(x)

    model = Model(
        inputs=base_model.input,
        outputs=output,
        name="HNAdam_VGG19",
    )

    return model