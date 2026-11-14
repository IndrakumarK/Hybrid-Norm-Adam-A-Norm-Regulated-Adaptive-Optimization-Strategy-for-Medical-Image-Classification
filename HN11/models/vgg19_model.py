
import tensorflow as tf

from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    BatchNormalization
)
from tensorflow.keras.models import Model


def build_vgg19(
    num_classes=1,
    input_shape=(240, 240, 3),
    weights="imagenet",
    freeze_until=15
):
    """
    Modified VGG19 for Binary Brain Tumor MRI Classification.

    Architecture:
        1. ImageNet-pretrained VGG19 backbone.
        2. First 15 backbone layers frozen.
        3. Global Average Pooling.
        4. Batch Normalization.
        5. Dense layer with 256 neurons and ReLU activation.
        6. Dropout (0.5).
        7. Dense layer with 128 neurons and ReLU activation.
        8. Dropout (0.3).
        9. Single sigmoid output neuron.

    Classification:
        Tumor     -> 1
        No Tumor  -> 0

    Loss:
        Binary Cross-Entropy.

    Args:
        num_classes (int): Must be 1 for binary classification.
        input_shape (tuple): Input image dimensions.
        weights (str): Pretrained weights ("imagenet" or None).
        freeze_until (int): Number of initial backbone layers to freeze.

    Returns:
        tensorflow.keras.Model: Modified VGG19 classification model.
    """

    # --------------------------------------------------
    # Validate configuration
    # --------------------------------------------------

    if num_classes != 1:
        raise ValueError(
            "This experimental configuration supports "
            "binary classification only. Set num_classes=1."
        )

    if len(input_shape) != 3 or input_shape[-1] != 3:
        raise ValueError(
            "Input shape must have three channels, "
            "for example (240, 240, 3)."
        )

    # --------------------------------------------------
    # Load pretrained VGG19 backbone
    # --------------------------------------------------

    base_model = VGG19(
        weights=weights,
        include_top=False,
        input_shape=input_shape
    )

    # --------------------------------------------------
    # Freeze initial VGG19 layers
    # --------------------------------------------------

    if not 0 <= freeze_until <= len(base_model.layers):
        raise ValueError(
            f"freeze_until must be between 0 and "
            f"{len(base_model.layers)}."
        )

    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False

    # --------------------------------------------------
    # Custom classification head
    # --------------------------------------------------

    x = base_model.output

    # Global feature aggregation
    x = GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    # Feature normalization
    x = BatchNormalization(
        name="feature_batch_normalization"
    )(x)

    # First dense layer
    x = Dense(
        256,
        activation="relu",
        name="dense_256"
    )(x)

    x = Dropout(
        0.5,
        name="dropout_50"
    )(x)

    # Second dense layer
    x = Dense(
        128,
        activation="relu",
        name="dense_128"
    )(x)

    x = Dropout(
        0.3,
        name="dropout_30"
    )(x)

    # --------------------------------------------------
    # Binary classification output
    # --------------------------------------------------

    outputs = Dense(
        1,
        activation="sigmoid",
        name="binary_classification_output"
    )(x)

    # --------------------------------------------------
    # Construct final model
    # --------------------------------------------------

    model = Model(
        inputs=base_model.input,
        outputs=outputs,
        name="Modified_VGG19_HNAdam"
    )

    return model


# ------------------------------------------------------
# Model summary and parameter verification
# ------------------------------------------------------

if __name__ == "__main__":

    model = build_vgg19(
        num_classes=1,
        input_shape=(240, 240, 3),
        weights="imagenet",
        freeze_until=15
    )

    model.summary()

    total_params = model.count_params()

    trainable_params = sum(
        tf.keras.backend.count_params(weight)
        for weight in model.trainable_weights
    )

    non_trainable_params = sum(
        tf.keras.backend.count_params(weight)
        for weight in model.non_trainable_weights
    )

    print("\n========== MODEL PARAMETERS ==========")
    print(f"Total parameters:       {total_params:,}")
    print(f"Trainable parameters:   {trainable_params:,}")
    print(f"Non-trainable params:   {non_trainable_params:,}")

    print("\n========== MODEL CONFIGURATION ==========")
    print("Architecture: Modified VGG19")
    print("Input shape: (240, 240, 3)")
    print("Pretrained weights: ImageNet")
    print("Frozen backbone layers: First 15")
    print("Classification: Binary")
    print("Output activation: Sigmoid")
    print("Recommended loss: Binary Cross-Entropy")
