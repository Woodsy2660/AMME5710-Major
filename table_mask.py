"""
Step 1: find the blue cloth, then the outline of the table.

- Threshold cloth colour (blue) in HSV 
- Clean up with opening and closing morphological operations
- Keep largest colour and take its convex hull 

"""

import cv2 
import numpy as np 

import config 

def cloth_colour_at(image_bgr, click_x, click_y):
    """
    Returns the median HSV colour of a 40 x 40 pixel patch around the click.
    """
    image_hsv = cv2.cvtColor(cv2.GaussianBlur(image_bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
    patch = image_hsv[click_y - 20:click_y + 20, click_x - 20:click_x + 20]
    patch = patch.reshape(-1, 3).astype(float)
    return (float(np.median(patch[:, 0])), float(np.median(patch[:, 1])), float(np.median(patch[:, 2])))

def find_cloth_center(image_bgr):
    """
    Automatically finds a point on the cloth by looking for the dominant blue/green region.
    Returns (click_x, click_y) for a point likely to be on the table cloth.
    """
    height, width = image_bgr.shape[:2]

    # Convert to HSV and look for typical cloth colors (blue/green hues)
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Pool table cloths are typically blue or green with high saturation
    hue = image_hsv[:, :, 0]
    saturation = image_hsv[:, :, 1]

    # Mask for blue-green hues (roughly 80-130 in OpenCV's 0-180 scale) with good saturation
    cloth_like = ((hue > 80) & (hue < 130) & (saturation > 50)).astype(np.uint8) * 255

    # Find the largest region
    contours, _ = cv2.findContours(cloth_like, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return cx, cy

    # Fallback to image center
    return width // 2, height // 2

def find_cloth_mask(image_bgr, click_x, click_y, cloth_colour=None):
    """
    inputs
    - image_bgr : the photo as loaded by cv2.imread()
    - click_x, click_y one pixel that is definitely cloth (from a mouse click)
    - cloth_colour : optional (hue, saturation, value) from cloth_colour_at(), 
                     click is not sampled again if given. 
    outputs
    - image_hsv : the photo converted to HSV 
    - cloth_mask : 255 where the pixel looks like cloth, 0 elsewhere    
    """
    # A light blur removes pixel noise before thresholding.
    blurred = cv2.GaussianBlur(image_bgr, (5, 5), 0)

    # HSV separates colour (hue) from brightness (value).
    # A shadow on the cloth lowers value but barely changes hue,
    # which is why a hue-based threshold is more reliable than RGB.
    # Channel order is H, S, V  (HLS was H, L, S: saturation has moved from index 2 to index 1).
    image_hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    if cloth_colour is None:
        cloth_colour = cloth_colour_at(image_bgr, click_x, click_y)
    cloth_hue, cloth_saturation, cloth_value = cloth_colour

    # hue = image_hsv[:, :, 0].astype(float)
    # saturation = image_hsv[:, :, 1].astype(float)
    # value = image_hsv[:, :, 2].astype(float)

    # # Hue is an angle that wraps round (179 is next to 0), so the difference
    # # between two hues is the shorter way round the circle.
    # hue_difference = np.abs(hue - cloth_hue)
    # hue_difference = np.minimum(hue_difference, 180 - hue_difference)

    # is_cloth = (hue_difference < config.CLOTH_HUE_TOLERANCE) \
    #     & (np.abs(value - cloth_value) < config.CLOTH_VALUE_TOLERANCE) \
    #     & (saturation > cloth_saturation - config.CLOTH_SATURATION_DROP)

    # cloth_mask = (is_cloth * 255).astype(np.uint8)

    levels = np.arange(256, dtype=float)
    hue_difference = np.abs(levels - cloth_hue)
    hue_difference = np.minimum(hue_difference, 180 - hue_difference)

    hue_ok   = ((hue_difference < config.CLOTH_HUE_TOLERANCE) * 255).astype(np.uint8)
    sat_ok   = ((levels > cloth_saturation - config.CLOTH_SATURATION_DROP) * 255).astype(np.uint8)
    value_ok = ((np.abs(levels - cloth_value) < config.CLOTH_VALUE_TOLERANCE) * 255).astype(np.uint8)

    h, s, v = cv2.split(image_hsv)
    cloth_mask = cv2.bitwise_and(cv2.LUT(h, hue_ok),
                                cv2.bitwise_and(cv2.LUT(s, sat_ok), cv2.LUT(v, value_ok)))

    return image_hsv, cloth_mask

def find_table_outline(cloth_mask, click_x, click_y):
    """
    Inputs
    - cloth_mask : from find_cloth_mask()
    - click_x, click_y : same cloth click
    Outputs 
    - table_contour : outline of the cloth region (follows every bump, including pockets)
    - table_hull : the convex hull of that outline (forms aa tight rectangle of the pool table)
    """

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    # Opening (erode then dilate) deletes small white specks outside the table.
    cleaned = cv2.morphologyEx(cloth_mask, cv2.MORPH_OPEN, kernel)
    
    # Closing (dilate then erode) fills small black gaps inside the table.
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Go through the contours from largest to smallest and take the first one
    # that contains the click.
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    table_contour = contours[0]
    for contour in contours:
        inside = cv2.pointPolygonTest(contour, (float(click_x), float(click_y)), False)
        if inside >= 0:
            table_contour = contour
            break

    # The pockets are bites out of the cloth outline. The convex hull bridges
    # across them, which leaves four nearly straight sides.
    table_hull = cv2.convexHull(table_contour)
    return table_contour, table_hull
