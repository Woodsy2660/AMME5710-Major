"""
Shot detection logic for identifying when a shot is taken in the pool game.
State machine with states SHOT and SETTLED, determined based on the number of moving pixels in the video frames.
The shot is considered to be in progress when the number of moving pixels exceeds a certain threshold 
"""

import cv2
import config
import numpy as np
import table_mask

def shrink_to_gray(frame_bgr):
    """
    The frame at SIGNAL_SCALE, in grey, lightly blurred. Used for the moving-pixels test.
    """
    small = cv2.resize(frame_bgr, None, fx=config.SIGNAL_SCALE, fy=config.SIGNAL_SCALE, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)

def moving_pixels(gray_small, previous_gray_small, reference):
    """ How many pixels inside the table changed since the last frame (balls rolling, cue, arms). """
    if previous_gray_small is None:
        return 0
    difference = cv2.absdiff(gray_small, previous_gray_small)
    return int(np.count_nonzero((difference > config.MOVING_PIXEL_DIFF) & (reference['inner_small'] > 0)))


def view_is_blocked(frame_bgr, reference):
    """
    True if something other than balls is over the table.
    The biggest non-cloth lump inside the table is measured: a ball cluster is small and sits in
    the middle; a person, an arm or a cue is big, and always comes in from the table edge.
    """
    small = cv2.resize(frame_bgr, None, fx=config.SIGNAL_SCALE, fy=config.SIGNAL_SCALE, interpolation=cv2.INTER_AREA)
    _, cloth_small = table_mask.find_cloth_mask(small, 0, 0, reference['cloth_colour'])
    not_cloth = cv2.bitwise_and(reference['inner_small'], cv2.bitwise_not(cloth_small))
    not_cloth = cv2.morphologyEx(not_cloth, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    lump_count, lump_image, lump_stats, _ = cv2.connectedComponentsWithStats(not_cloth)
    if lump_count <= 1:
        return False
    biggest = 1 + int(np.argmax(lump_stats[1:, 4]))
    area = int(lump_stats[biggest, 4])
    touches_edge = bool(np.any((lump_image == biggest) & (reference['edge_small'] > 0)))

    big = config.OCCLUDER_BIG_FRACTION * reference['table_area_small']
    edge = config.OCCLUDER_EDGE_FRACTION * reference['table_area_small']
    return area > big or (area > edge and touches_edge)


def update_detector(detector, moving, blocked, reference):
    """
    The SHOT / SETTLED state machine, kept in a plain dictionary.
        moving pixels above MOVE_HIGH  -> SHOT, still-counter restarts
        moving pixels below MOVE_LOW   -> still-counter + 1
        in between                     -> still-counter restarts (a ball slowing down)
        SHOT becomes SETTLED once the still-counter reaches HOLD_FRAMES
    Returns True if the board may be read from this frame.
    Two thresholds instead of one stop the state flickering while a ball slows down.
    """
    move_high = config.MOVE_HIGH_FRACTION * reference['table_area_small']
    move_low = config.MOVE_LOW_FRACTION * reference['table_area_small']

    if moving > move_high:
        detector['state'] = 'SHOT'
        detector['still_count'] = 0
    elif moving < move_low:
        detector['still_count'] = detector['still_count'] + 1
    else:
        detector['still_count'] = 0

    if detector['state'] == 'SHOT' and detector['still_count'] >= config.HOLD_FRAMES:
        detector['state'] = 'SETTLED'

    detector['blocked'] = blocked
    detector['readable'] = detector['state'] == 'SETTLED' and detector['still_count'] >= config.HOLD_FRAMES and not blocked
    return detector['readable']
