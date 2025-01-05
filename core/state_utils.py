import cv2
import numpy as np

def preprocess_state(raw_screen):
    """
    Preprocesses the raw screen for input to the neural network.
    Args:
        raw_screen (np.ndarray): Raw screen capture (height, width, 3).
    Returns:
        np.ndarray: Preprocessed state (1, height, width).
    """
    # Convert to grayscale
    gray_screen = cv2.cvtColor(raw_screen, cv2.COLOR_RGB2GRAY)

    # Resize to the required input size for the model
    resized_screen = cv2.resize(gray_screen, (128, 128))  # Example size

    # Normalize pixel values to [0, 1]
    normalized_screen = resized_screen / 255.0

    # Add channel dimension (1, height, width)
    state = np.expand_dims(normalized_screen, axis=0)
    return state
