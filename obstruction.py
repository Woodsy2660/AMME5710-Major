"""
Obstruction detection module for pool table video analysis.

Replaces scattered functions (view_is_blocked, covers_balls, camera_moved) with a unified
approach that measures non-cloth lumps by their shape in ball units.

See INTEGRATION.md for the full specification of what each function should do.

Config parameters used (add to config.py):
- THICK_RADII: thickness threshold for obstruction detection
- THIN_MIN_BALLS: thin part area threshold
- EDGE_CONTACT_RADII: edge contact length threshold
- EDGE_MIN_BALLS: edge contact area threshold
- HUGE_BALLS: area threshold for huge obstructions
- HIDDEN_DILATE_PX: dilation for occluder mask
- CLEAR_FRAMES: frames required for blocked->clear transition
- CAMERA_SHIFT_PX: pixel shift threshold for camera movement
- CAMERA_SHIFT_FRAMES: frame count for camera movement detection
"""

import cv2
import numpy as np
import config
import table_mask
import top_down


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def to_hsv(bgr_image):
    """Convert BGR image to HSV."""
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)


def cloth_mask_for_colour(hsv, cloth_colour):
    """
    Create a mask of cloth-coloured pixels.
    Uses the cloth colour tolerances from config.
    cloth_colour is (hue, saturation, value) in HSV format.
    """
    h, s, v = cloth_colour
    lower = np.array([max(0, h - config.CLOTH_HUE_TOLERANCE),
                      max(0, s - config.CLOTH_SATURATION_DROP),
                      max(0, v - config.CLOTH_VALUE_TOLERANCE)])
    upper = np.array([min(180, h + config.CLOTH_HUE_TOLERANCE),
                      255,
                      min(255, v + config.CLOTH_VALUE_TOLERANCE)])
    return cv2.inRange(hsv, lower, upper)


def ball_radius_in_photo(to_map, to_photo, photo_x, photo_y):
    """
    The ball radius expected at a photo position, in photo pixels. Uses top_down's measurement
    through the table's perspective, so it is right for any camera distance and angle (a rough
    fixed size made every ball look 'thick' when the camera is close to the table).
    """
    return top_down.ball_radius_in_photo(to_map, to_photo, photo_x, photo_y)


def move_point(matrix, x, y):
    """
    Apply a 3x3 homography matrix to a point (x, y).
    Returns the transformed (x', y') coordinates.
    """
    pt = np.array([x, y, 1.0])
    result = matrix @ pt
    if result[2] != 0:
        return result[0] / result[2], result[1] / result[2]
    return x, y


# -----------------------------------------------------------------------------
# Lump detection
# -----------------------------------------------------------------------------

def unexplained_lumps(frame_bgr, reference):
    """
    Steps: 
    1. cloth mask on the small frame, shadows on the cloth (cloth hue, still saturated) 
    count as cloth.
    2. what is left inside the trimmed table outline, opened to drop specks, is split into lumps. 
    3. each lump is measured in units of the ball expected at that spot 
    4. ball shaped lumps are dropped, other non-ball identifying lumps (thick, long-thin, spread along the table(hand))

    Inputs
    - frame_bgr : the current full-size frame
    - reference : from board_reading.make_reference (cloth_colour, inner_small, to_map, to_photo)
    Outputs
    - lump_mask : small-frame image, 255 where an obstruction lump is (balls are NOT in it)
    - lumps : one dictionary per obstruction lump: area_balls, thickness_radii, thin_balls, contact_radii,
            kind ('thick', 'thin' or 'edge')
    """

    scale = config.SIGNAL_SCALE
    small = cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    hsv = to_hsv(small)
    cloth = cloth_mask_for_colour(hsv, reference['cloth_colour'])

    # A shadow on the cloth keeps the cloth's hue but loses brightness. It is cloth, not an obstruction.
    cloth_hue = reference['cloth_colour'][0]
    hue_difference = np.abs(hsv[:, :, 0].astype(float) - cloth_hue)
    hue_difference = np.minimum(hue_difference, 180 - hue_difference)
    shadow = (hue_difference < config.CLOTH_HUE_TOLERANCE) & (hsv[:, :, 1] > config.SHADOW_MIN_SATURATION)
    cloth[shadow] = 255

    not_cloth = cv2.bitwise_and(reference['inner_small'], cv2.bitwise_not(cloth))
    not_cloth = cv2.morphologyEx(not_cloth, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    lump_count, lump_image, lump_stats, lump_centres = cv2.connectedComponentsWithStats(not_cloth)
    lumps = []
    lump_mask = np.zeros_like(not_cloth)
    for lump in range(1, lump_count):
        # the ball radius expected where this lump is (near balls look bigger than far ones)
        centre_x, centre_y = lump_centres[lump] / scale
        radius = ball_radius_in_photo(reference['to_map'], reference['to_photo'], centre_x, centre_y) * scale
        one_ball = np.pi * radius * radius

        area_balls = lump_stats[lump, 4] / one_ball
        if area_balls < 0.25:
            continue                                      # a speck
        pixels = lump_image == lump
        lump_pixels = pixels.astype(np.uint8)
        thickness_radii = cv2.distanceTransform(lump_pixels, cv2.DIST_L2, 3).max() / radius

        # the thin part of the lump: what an opening with a disc THIN_RADII wide removes.
        # A cue lying across the cloth is long and thin even where it touches balls.
        disc = max(3, int(round(2 * config.THIN_RADII * radius))) | 1
        opened = cv2.morphologyEx(lump_pixels, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (disc, disc)))
        thin_balls = (lump_stats[lump, 4] - int(opened.sum())) / one_ball

        # how much of the lump lies along the table edge (an arm or hand always reaches in from the side)
        contact_radii = np.count_nonzero(pixels & (reference['edge_small'] > 0)) / config.EDGE_RING_WIDTH / radius

        if thickness_radii > config.THICK_RADII or area_balls > config.HUGE_BALLS:
            kind = 'thick'                                # arm, hand, body, phone
        elif thin_balls > config.THIN_MIN_BALLS:
            kind = 'thin'                                 # a cue resting on the cloth
        elif contact_radii > config.EDGE_CONTACT_RADII and area_balls > config.EDGE_MIN_BALLS:
            kind = 'edge'                                 # a hand or arm over the cushion, reaching in
        else:
            continue                                      # ball-shaped: a ball, a cluster, a blurred moving ball
        lumps.append({'area_balls': float(area_balls), 'thickness_radii': float(thickness_radii),
                      'thin_balls': float(thin_balls), 'contact_radii': float(contact_radii), 'kind': kind})
        lump_mask[pixels] = 255
    return lump_mask, lumps


def occluder_mask(lump_mask):
    """
    Grows the obstruction lumps a little, thus ball whose edge is under an arm counts as hidden. 
    """
    kernel = np.ones((config.HIDDEN_DILATE_PX * 2 + 1,) * 2, np.uint8)
    return cv2.dilate(lump_mask, kernel)


# -----------------------------------------------------------------------------
# View blocked detection
# -----------------------------------------------------------------------------

def view_is_blocked(lumps):
    """
    True if there is any obstruction lump at all (already filtered by shape)
    """
    return len(lumps) > 0


def new_blocked_filter():
    return {'blocked': False, 'clear_count': 0}


def update_blocked(filter_state, blocked_now):
    """
    Blocked: when switches ON at once, and OFF only after CLEAR_FRAMES in a row.
    A hand that flickers in and out of the cloth mask cannot make the reader flicker with it.
    Returns the smoothed value.
    """
    if blocked_now:
        filter_state['blocked'] = True
        filter_state['clear_count'] = 0
    else:
        filter_state['clear_count'] = filter_state['clear_count'] + 1
        if filter_state['clear_count'] >= config.CLEAR_FRAMES:
            filter_state['blocked'] = False
    return filter_state['blocked']


# -----------------------------------------------------------------------------
# Per-ball hidden detection
# -----------------------------------------------------------------------------

def hidden_balls(balls, occluder, reference):
    """
    Inputs
    - balls : records with map_x, map_y (tracked positions during a shot, or a committed state)
    - occluder : from occluder_mask
    - reference : for to_photo
    Output
    - a list of True / False, one per ball: True if its centre is under an obstruction
    The tracker uses this per ball: a hidden ball keeps coasting and is never called potted.
    """
    scale = config.SIGNAL_SCALE
    height, width = occluder.shape
    result = []
    for ball in balls:
        photo_x, photo_y = move_point(reference['to_photo'], ball['map_x'], ball['map_y'])
        x = int(round(photo_x * scale))
        y = int(round(photo_y * scale))
        inside = 0 <= x < width and 0 <= y < height
        result.append(bool(inside and occluder[y, x] > 0))
    return result
# -----------------------------------------------------------------------------
# Camera movement detection
# -----------------------------------------------------------------------------

def new_camera_filter():
    return {'moved': False, 'moved_count': 0, 'shift': 0.0}


def camera_moved(gray_small, reference, filter_state):
    """
    Phase correlation between the current small grey frame and the REFERENCE frame (the first
    frame, stored as reference['first_gray_small']). Compared with the first frame, not the previous
    one, so a slow drift is caught as well as a bump. A person walking through can fake a shift
    for a frame or two, so the answer only flips after CAMERA_SHIFT_FRAMES frames in a row.
    Returns True while the camera is considered moved (the table must be re-found).
    """
    (shift_x, shift_y), _ = cv2.phaseCorrelate(np.float32(reference['first_gray_small']), np.float32(gray_small))
    shift = float(np.hypot(shift_x, shift_y))
    filter_state['shift'] = shift
    if shift > config.CAMERA_SHIFT_PX:
        filter_state['moved_count'] = filter_state['moved_count'] + 1
        if filter_state['moved_count'] >= config.CAMERA_SHIFT_FRAMES:
            filter_state['moved'] = True
    else:
        filter_state['moved_count'] = 0
        filter_state['moved'] = False
    return filter_state['moved']
