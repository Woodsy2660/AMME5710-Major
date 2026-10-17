"""
Step 6: pictures of the result. Nothing here affects detection. Built to be modula r
"""

import cv2
import numpy as np

import archive.config as config

# Drawing colours in BGR order.
DRAW_COLOURS = {'cue': (235, 235, 235), 'black': (40, 40, 40),
                'red': (40, 40, 230), 'yellow': (0, 200, 255)}

def draw_overlay(image_bgr, table_hull, corners, balls):
    """
    Draws the table outline, corners and balls on a copy of the photo.
    Inputs
    - image_bgr : the photo as loaded by cv2.imread()
    - table_hull : from find_table_outline()
    - corners : from find_corners()
    - balls : list of ball records from detect_balls()
    Output
    - image_bgr : a copy of the photo with the overlay drawn on it
    """

    picture = image_bgr.copy()
    image_height, image_width = picture.shape[:2]

    cv2.drawContours(picture, [table_hull], -1, (0, 255, 0), 2)

    for corner_x, corner_y in corners:
        # A corner can fall just outside the photo, so clamp it to the edge before drawing.
        x = int(min(max(corner_x, 0), image_width - 1))
        y = int(min(max(corner_y, 0), image_height - 1))
        cv2.circle(picture, (x, y), 10, (0, 0, 255), -1)

    for ball in balls:
        centre = (int(ball['photo_x']), int(ball['photo_y']))
        radius = int(ball['radius']) + 3
        colour = DRAW_COLOURS[ball['colour']]
        letter = ball['colour'][0].upper()
        label_position = (centre[0] + radius + 2, centre[1] - 3)

        # Draw each mark twice, thick black then thin colour, so it shows on any background.
        cv2.circle(picture, centre, radius, (0, 0, 0), 4)
        cv2.circle(picture, centre, radius, colour, 2)
        cv2.putText(picture, letter, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(picture, letter, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

    return picture

def draw_map(balls):
    """
    Draws a top-down view of the table with the balls on it.
    Inputs
    - balls : list of ball records from detect_balls()
    Output
    - map_image : a new image showing the table and balls from above
    """

    width = config.MAP_WIDTH
    height = config.MAP_HEIGHT
    border = 36

    picture = np.full((height + 2 * border, width + 2 * border, 3), (60, 60, 60), dtype=np.uint8)
    cv2.rectangle(picture, (border, border), (border + width, border + height), (200, 120, 30), -1)
    cv2.rectangle(picture, (border, border), (border + width, border + height), (0, 255, 0), 2)

    pockets = [(0, 0), (width / 2, 0), (width, 0), (0, height), (width / 2, height), (width, height)]
    for pocket_x, pocket_y in pockets:
        cv2.circle(picture, (int(border + pocket_x), int(border + pocket_y)), 20, (0, 0, 0), -1)

    ball_size = int(round(config.BALL_RADIUS_ON_MAP)) + 1
    for ball in balls:
        centre = (int(border + ball['map_x']), int(border + ball['map_y']))
        cv2.circle(picture, centre, ball_size, DRAW_COLOURS[ball['colour']], -1)
        cv2.circle(picture, centre, ball_size, (0, 0, 0), 1)

    return picture
