
import cv2

import config
import numpy as np
import config, ball_colour, ball_detection, table_mask, table_corners, top_down, draw
# Tracking numbers (move to config.py once they are tuned). Distances are map px unless stated.
WINDOW_BASE_RADII = 3.0     # search window half-size for a found ball, in ball radii
WINDOW_SPEED_FACTOR = 1.5   # extra half-size per map px the ball is expected to move this frame
WINDOW_MAX_RADII = 15.0     # the window doubles every frame a ball is unseen, up to this size
VELOCITY_BLEND = 0.6        # weight of a new velocity measurement (the rest is the old velocity)
MAX_COAST_FRAMES = 8        # a moving ball unseen for more frames than this is dropped
DROP_MIN_SPEED = 20.0       # map px/s; slower than this counts as still, and is never dropped
POCKET_ZONE_RADII = 3.0     # dropped within this many radii of a pocket centre = potted there
MAP_BORDER = 36             # must match the border in draw.draw_map
NO_MATCH = 1e6              # cost for a pairing outside a ball's window

FAMILY = {'cue': ball_colour.WHITE, 'black': ball_colour.DARK,
          'red': ball_colour.RED, 'yellow': ball_colour.YELLOW}
POCKETS = {'top-left': (0, 0), 'top-middle': (config.MAP_WIDTH / 2, 0), 'top-right': (config.MAP_WIDTH, 0),
           'bottom-left': (0, config.MAP_HEIGHT), 'bottom-middle': (config.MAP_WIDTH / 2, config.MAP_HEIGHT),
           'bottom-right': (config.MAP_WIDTH, config.MAP_HEIGHT)}


def window_half_size(track, frame_dt):
    """
    Half-size of the search square on the map. Grows with speed, doubles per unseen frame.
    """
    r = config.BALL_RADIUS_ON_MAP
    speed = np.hypot(track['vx'], track['vy'])
    half = WINDOW_BASE_RADII * r + WINDOW_SPEED_FACTOR * speed * frame_dt
    half = half * 2 ** min(track['missed'], 10)
    return min(half, WINDOW_MAX_RADII * r)

def photo_window(to_photo, map_x, map_y, half, frame_shape):
    """
    The map square round (map_x, map_y) moved into the photo, as a clipped bounding box
    (x0, y0, x1, y1), or None if nothing of it is in the frame.
    """
    corners = [top_down.move_point(to_photo, map_x + dx, map_y + dy) for dx in (-half, half) for dy in (-half, half)]
    xs, ys = zip(*corners)
    height, width = frame_shape[:2]
    x0, x1 = max(0, int(min(xs))), min(width, int(max(xs)) + 1)
    y0, y1 = max(0, int(min(ys))), min(height, int(max(ys)) + 1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    return x0, y0, x1, y1

def find_candidates(frame_bgr, window, colour, reference):
    """
    Map positions of balls of one colour inside one photo window.
    Same idea as ball_detection.find_balls, on a crop: not cloth, on the table, the right colour,
    closed to fill the highlight, then distance-transform peaks (which also split touching balls).
    """
    x0, y0, x1, y1 = window
    crop = frame_bgr[y0:y1, x0:x1]
    crop_hsv, cloth = table_mask.find_cloth_mask(crop, 0, 0, reference['cloth_colour'])
    labels = ball_colour.label_pixels(crop_hsv.reshape(-1, 3)).reshape(cloth.shape)
    on_table = reference['hull_mask'][y0:y1, x0:x1] > 0
    mask = ((labels == FAMILY[colour]) & (cloth == 0) & on_table).astype(np.uint8)

    radius = top_down.ball_radius_in_photo(reference['to_map'], reference['to_photo'], (x0 + x1) / 2, (y0 + y1) / 2)
    size = max(3, int(config.CLOSE_KERNEL_RATIO * radius)) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))

    found = []
    for x, y, _ in ball_detection.find_centres(mask, radius, reference['to_map'], x0, y0):
        map_x, map_y = top_down.move_point(reference['to_map'], x, y)
        if colour == 'black' and ball_detection.is_at_pocket_mouth(map_x, map_y):
            continue                                  # a dark pocket opening, not the black ball
        found.append((map_x, map_y))
    return found


def merge_close(points):
    """
    Windows overlap, so the same ball can be found twice. Keep one of any two closer than half a radius.
    """
    kept = []
    for x, y in points:
        if all(np.hypot(x - kept_x, y - kept_y) > 0.5 * config.BALL_RADIUS_ON_MAP for kept_x, kept_y in kept):
            kept.append((x, y))
    return kept
