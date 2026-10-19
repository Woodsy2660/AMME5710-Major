"""
Run ball detection and the game-state tracker on an mp4 video.

How to run: python run_video.py "path/to/video.mp4"

Output:<name>_tracked.mp4       every frame with the last committed state drawn, plus a status line

"""

import cv2
import numpy as np
import config, table_mask, table_corners, top_down, ball_detection, draw
from shot_detection import shrink_to_gray, moving_pixels, view_is_blocked, update_detector
from ball_tracking import initialise_tracking, track_frame, closest_pocket

def make_reference(first_frame, click_x, click_y):
    """
    Everything that is worked out ONCE from the first frame and reused for the whole video.
    Output: a dictionary with
    - cloth_colour: (h, s, v) of the cloth
    - table_hull, corners: outline and four corners (long side first), full-frame pixels
    - to_map, to_photo: photo <-> map matrices
    - inner_small: filled hull at SIGNAL_SCALE, trimmed by 15 px so the cushion edge does not flicker
    - edge_small: a thin ring just inside the trimmed hull (for the "touches the edge" test)
    - table_area_small: number of pixels in the hull at SIGNAL_SCALE (the fractions in config are of this)
    """
    cloth_colour = table_mask.cloth_colour_at(first_frame, click_x, click_y)
    image_hsv, cloth_mask = table_mask.find_cloth_mask(first_frame, click_x, click_y, cloth_colour)
    table_contour, table_hull = table_mask.find_table_outline(cloth_mask, click_x, click_y)
    corners = table_corners.put_long_side_first(table_corners.find_corners(table_hull), table_contour)
    to_map, to_photo = top_down.make_map_transform(corners)

    scale = config.SIGNAL_SCALE
    small_height = int(round(first_frame.shape[0] * scale))
    small_width = int(round(first_frame.shape[1] * scale))
    hull_small = np.zeros((small_height, small_width), dtype=np.uint8)
    cv2.fillConvexPoly(hull_small, np.round(table_hull * scale).astype(np.int32), 255)
    inner_small = cv2.erode(hull_small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    edge_small = cv2.subtract(inner_small, cv2.erode(inner_small, np.ones((7, 7), np.uint8)))
    hull_mask = np.zeros(first_frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(hull_mask, table_hull, 255)    
    return {'cloth_colour': cloth_colour, 'table_hull': table_hull, 'corners': corners,
            'to_map': to_map, 'to_photo': to_photo, 'inner_small': inner_small, 'edge_small': edge_small,
            'table_area_small': int(np.count_nonzero(hull_small)), 'hull_mask': hull_mask}

def read_board(frame_bgr, reference, click_x, click_y):
    """
    The ball list for one full-size frame, using the corners stored in the reference.
    """
    image_hsv, cloth_mask = table_mask.find_cloth_mask(frame_bgr, click_x, click_y, reference['cloth_colour'])
    table_contour, table_hull = table_mask.find_table_outline(cloth_mask, click_x, click_y)
    return ball_detection.find_balls(image_hsv, cloth_mask, table_contour, table_hull, reference['to_map'], reference['to_photo'])

def count_colours(balls):
    """
    {'cue': n, 'black': n, 'red': n, 'yellow': n}
    """
    counts = {'cue': 0, 'black': 0, 'red': 0, 'yellow': 0}
    for ball in balls:
        counts[ball['colour']] = counts[ball['colour']] + 1
    return counts

def match_balls(balls_a, balls_b, tolerance):
    """
    Pairs each ball in balls_a with the nearest unused ball of the same colour in balls_b,
    nearest pairs first. Returns the balls of a that found no partner within tolerance (map px).
    """
    candidates = []
    for i, ball_a in enumerate(balls_a):
        for j, ball_b in enumerate(balls_b):
            if ball_a['colour'] == ball_b['colour']:
                distance = np.hypot(ball_a['map_x'] - ball_b['map_x'], ball_a['map_y'] - ball_b['map_y'])
                if distance <= tolerance:
                    candidates.append((distance, i, j))
    candidates.sort()
    used_a = set()
    used_b = set()
    for distance, i, j in candidates:
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
    return [balls_a[i] for i in range(len(balls_a)) if i not in used_a]

def same_reading(balls_a, balls_b):
    """ True if two readings show the same balls in the same places (within AGREE_TOLERANCE radii). """
    if count_colours(balls_a) != count_colours(balls_b):
        return False
    tolerance = config.AGREE_TOLERANCE * config.BALL_RADIUS_ON_MAP
    return len(match_balls(balls_a, balls_b, tolerance)) == 0
