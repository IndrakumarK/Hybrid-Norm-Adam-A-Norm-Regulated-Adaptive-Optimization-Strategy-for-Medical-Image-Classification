"""
Grad-CAM++ visualization for a binary VGG19 classifier.
"""

import cv2
import numpy as np
import tensorflow as tf


def find_last_conv_layer(model):
    """Find the last convolutional layer in the model."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name

        if isinstance(layer, tf.keras.Model):
            try:
                return find_last_conv_layer(layer)
            except ValueError:
                continue

    raise ValueError("No Conv2D layer found in the model.")


def gradcam_plus_plus(model, image_batch, layer_name=None):
    """
    Generate Grad-CAM++ heatmaps for a sigmoid-output classifier.

    Args:
        model: Trained binary classification model.
        image_batch: Input images with shape (batch, height, width, channels).
        layer_name: Name of the convolutional layer.

    Returns:
        heatmaps: Normalized heatmaps with shape (batch, height, width).
        predictions: Model predictions.
    """
    if layer_name is None:
        layer_name = find_last_conv_layer(model)

    target_layer = model.get_layer(layer_name)

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output],
    )

    images = tf.convert_to_tensor(image_batch, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(images)

        with tf.GradientTape() as second_tape:
            second_tape.watch(images)

            with tf.GradientTape() as first_tape:
                first_tape.watch(images)

                conv_outputs, predictions = grad_model(
                    images,
                    training=False,
                )

                class_scores = predictions[:, 0]

            first_gradients = first_tape.gradient(
                class_scores,
                conv_outputs,
            )

        second_gradients = second_tape.gradient(
            first_gradients,
            conv_outputs,
        )

    third_gradients = tape.gradient(
        second_gradients,
        conv_outputs,
    )

    if (
        first_gradients is None
        or second_gradients is None
        or third_gradients is None
    ):
        raise RuntimeError(
            "Unable to calculate the required Grad-CAM++ derivatives."
        )

    activation_sum = tf.reduce_sum(
        conv_outputs,
        axis=[1, 2],
        keepdims=True,
    )

    epsilon = tf.keras.backend.epsilon()

    alpha_numerator = second_gradients

    alpha_denominator = (
        2.0 * second_gradients
        + third_gradients * activation_sum
    )

    alphas = alpha_numerator / (
        alpha_denominator + epsilon
    )

    positive_gradients = tf.nn.relu(first_gradients)

    weights = tf.reduce_sum(
        alphas * positive_gradients,
        axis=[1, 2],
    )

    heatmaps = tf.reduce_sum(
        weights[:, tf.newaxis, tf.newaxis, :]
        * conv_outputs,
        axis=-1,
    )

    heatmaps = tf.nn.relu(heatmaps)

    heatmaps = heatmaps / (
        tf.reduce_max(
            heatmaps,
            axis=[1, 2],
            keepdims=True,
        ) + epsilon
    )

    return heatmaps.numpy(), predictions.numpy()


def overlay_heatmap(image, heatmap, alpha=0.4):
    """
    Overlay a Grad-CAM++ heatmap on an MRI image.

    Args:
        image: Original RGB image.
        heatmap: Normalized two-dimensional heatmap.
        alpha: Heatmap transparency.

    Returns:
        RGB image containing the heatmap overlay.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    image = np.asarray(image)

    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Expected an RGB image with three channels.")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255)

        if image.max() <= 1.0:
            image = image * 255.0

        image = image.astype(np.uint8)

    heatmap = np.asarray(heatmap, dtype=np.float32)

    heatmap = cv2.resize(
        heatmap,
        (image.shape[1], image.shape[0]),
    )

    heatmap = np.uint8(
        255 * np.clip(heatmap, 0.0, 1.0)
    )

    colored_heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET,
    )

    colored_heatmap = cv2.cvtColor(
        colored_heatmap,
        cv2.COLOR_BGR2RGB,
    )

    return cv2.addWeighted(
        image,
        1.0 - alpha,
        colored_heatmap,
        alpha,
        0,
    )