"""
Step 3: map between the photo and the top-down view 

- perspective transform (homography) from the four table corners to a 2:1 rectangle 
- applied to ball positions, not the whole image 

the homography H is a 3x3 matrix, To move a pixel (x, y): 

    [X, Y, W] = H @ [x, y, 1]
    new pixel = (X / W, Y / W)

division by W produces perspective distortion (further away more shrinkage)

limitations - far side corners are off by roughly a ball radius 
due to the corners sitting of top of the cushions 
"""

import cv2
import numpy as np

import config

def make_map_transform(corners):
    """
    Input
    - corners : four table corners, long side first 
    Outputs
    - to_map : 3 x 3 matrix that moves photo pixels onto the map 
    - to_photo : its inverse, which moves map positions back onto the photo 
    """

    width = config.MAP_WIDTH
    height = config.MAP_HEIGHT
    map_corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

    # Four point pairs give eight equations, exactly enough for the eight unknowns of H.
    to_map = cv2.getPerspectiveTransform(corners, map_corners)
    to_photo = np.linalg.inv(to_map)
    return to_map, to_photo

def move_point(transform, x, y):
    """
    Moves one point through a homograhy
    Returns (new_x, new_y)
    """
    moved = transform @ np.array([x, y, 1.0])
    return moved[0] / moved[2], moved[1] / moved[2]

def ball_radius_in_photo(to_map, to_photo, x, y):
    """
    How many pixels across a ball's radius should be at position (x, y)
    Balls near the camera look bigger than balls far away, changing across the table 

    Method: find where (x, y) is on the map, step one map pixel left/right and 
    up/down, move those steps back to the photo, and see how far apart they land.
    """
    map_x, map_y = move_point(to_map, x, y)

    left = move_point(to_photo, map_x - 1, map_y)
    right = move_point(to_photo, map_x + 1, map_y)
    up = move_point(to_photo, map_x, map_y - 1)
    down = move_point(to_photo, map_x, map_y + 1)

    photo_pixels_per_map_pixel_x = np.hypot(right[0] - left[0], right[1] - left[1]) / 2
    photo_pixels_per_map_pixel_y = np.hypot(down[0] - up[0], down[1] - up[1]) / 2

    # A ball is a sphere, so it looks about as wide as the less squashed direction.
    scale = max(photo_pixels_per_map_pixel_x, photo_pixels_per_map_pixel_y)
    return config.BALL_RADIUS_ON_MAP * scale