"""
Step 5: decidee what colour a ball is: cue, black, red, yellow. 

- Label every pixel as white, dark, red, yellow, or other.
- White dominance -> cue ball. black dominance -> black ball 
- otherwise average the hue of the coloured pixels and pick nearest of red or yellow. 

"""

import numpy as np

import config 

OTHER = 0
WHITE = 1 
DARK = 2 
RED = 3
YELLOW = 4 

LABEL_NAMES = {WHITE: 'cue', DARK: 'black', RED: 'red', YELLOW: 'yellow'}

def hue_distance(hue, reference):
    """
    Distnace between two hues, going the shorrt way round the 0-180 circle
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

def classify_ball(hsv_pixels):
    """
    Input
    - hsv_pixels : stated above
    Output 
    - colour : one of WHITE, DARK, RED, YELLOW
    """
    labels = label_pixels(hsv_pixels)
    counts = np.bincount(labels, minlength=5)

    if counts[WHITE] > config.WHITE_FRACTION_FOR_CUE * len(hsv_pixels):
        return 'cue'
    if counts[DARK] > config.DARK_FRACTION_FOR_BLACK * len(hsv_pixels):
        return 'black'

    # Otherwise it's a coloured ball. Average the hue of the coloured pixels and pick nearest.
    coloured_hues = hsv_pixels[(labels == RED) | (labels == YELLOW), 0]
    if len(coloured_hues) == 0:
        return 'unknown'
    average_hue = np.mean(coloured_hues)
    if hue_distance(average_hue, config.RED_HUE) < hue_distance(average_hue, config.YELLOW_HUE):
        return 'red'
    else:
        return 'yellow'