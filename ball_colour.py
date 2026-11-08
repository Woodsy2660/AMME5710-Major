"""
  Label pixels by colour: white, dark, red, yellow, or other.

  ball_detection.py and tracking_helpers.py build one mask per colour family from these labels,
  and ball_detection.colour_at decides each ball's colour from the labels in a disc round its centre.
"""

import numpy as np

import config 

OTHER = 0
WHITE = 1 
DARK = 2 
RED = 3
YELLOW = 4 

def hue_distance(hue, reference):
    """
    Distance between two hues, going the short way round the 0-180 circle
    """
    difference = np.abs(np.asarray(hue, dtype=float) - reference) % 180
    return np.minimum(difference, 180 - difference)

def label_pixels(hsv_pixels):
    """
    Input
    - hsv_pixels : N x 3 array of pixels in HSV colour space (the pixels of one blob)
    Output 
    - labels : N array of integers, one for each pixel, saying what colour it is
    """ 
    hue = hsv_pixels[:, 0].astype(float)
    saturation = hsv_pixels[:, 1].astype(float)
    value = hsv_pixels[:, 2].astype(float)

    labels = np.zeros(len(hsv_pixels), dtype=int)

    saturated = saturation > config.MIN_COLOUR_SATURATION
    labels[(hue_distance(hue, config.RED_HUE) < config.RED_HUE_TOLERANCE) & saturated] = RED
    labels[(hue_distance(hue, config.YELLOW_HUE) < config.YELLOW_HUE_TOLERANCE) & saturated] = YELLOW

    # Dark and white are decided last so they override a hue-based label.
    # In HSV, white means "bright AND weakly saturated". Brightness alone is not enough,
    # because a pure red or yellow also has a value near 255.
    labels[value < config.DARK_MAX_VALUE] = DARK
    is_white = (saturation < config.WHITE_MAX_SATURATION) & (value > config.WHITE_MIN_VALUE)
    labels[is_white] = WHITE
    return labels
