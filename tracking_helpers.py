
import cv2
import numpy as np
import config, ball_colour, ball_detection, table_mask, top_down
from config import WINDOW_BASE_RADII, WINDOW_SPEED_FACTOR, WINDOW_MAX_RADII

FAMILY = {'cue': ball_colour.WHITE, 'black': ball_colour.DARK,
          'red': ball_colour.RED, 'yellow': ball_colour.YELLOW}


def window_half_size(track, frame_dt):
    """
    Half-size of the search square on the map. Grows with speed, doubles per unseen frame.
    """
    r = config.BALL_RADIUS_ON_MAP
    if not track['free']:
        # Still and not hit yet: it can only be where it is. A small window stops it being
        # matched to a hand, the cue or another ball nearby (see ball_tracking.free_hit_balls).
        # It grows a little while the ball is not found, to allow for a handheld camera shaking.
        half = config.STILL_WINDOW_RADII * r * 2 ** min(track['missed'], 4)
        return min(half, config.STILL_WINDOW_MAX_RADII * r)
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
    mask = (labels == FAMILY[colour]) & (cloth == 0) & on_table
    if colour == 'cue':
        # The cue ball is a much purer white than skin (a hand on the rail is the usual imposter
        # just after a shot), so only very weakly saturated pixels count for it.
        mask = mask & (crop_hsv[:, :, 1] < config.CUE_MAX_SATURATION)
    mask = mask.astype(np.uint8)

    radius = top_down.ball_radius_in_photo(reference['to_map'], reference['to_photo'], (x0 + x1) / 2, (y0 + y1) / 2)
    size = max(3, int(config.CLOSE_KERNEL_RATIO * radius)) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))

    found = []
    for x, y, height in ball_detection.find_centres(mask, radius, reference['to_map'], x0, y0):
        # A ball's centre is about one radius from the edge of its colour patch, even when it is
        # blurred (blur makes it longer, not narrower). A much thinner patch is the cue stick or
        # its white tip; a much fatter one is a hand or arm (skin reads as yellow).
        if height < config.MIN_TRACK_PEAK_RATIO * radius or height > config.MAX_TRACK_PEAK_RATIO * radius:
            continue
        map_x, map_y = top_down.move_point(reference['to_map'], x, y)
        if colour == 'black' and ball_detection.is_at_pocket_mouth(map_x, map_y):
            continue                                  # a dark pocket opening, not the black ball
        # Verify dark/white candidates through colour_at to reject shadows
        if colour in ('cue', 'black'):
            expected_radius = top_down.ball_radius_in_photo(reference['to_map'], reference['to_photo'], x, y)
            # x, y are in full-frame coords; need crop-relative coords for crop_hsv
            crop_x, crop_y = x - x0, y - y0
            detected = ball_detection.colour_at(crop_hsv, crop_x, crop_y, 0.55 * expected_radius, 'darkwhite')
            if detected is None or detected != colour:
                continue
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
