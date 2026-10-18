"""
Run ball detection and the game-state tracker on an mp4 video.

How to run: python run_video.py "path/to/video.mp4"

Output:<name>_tracked.mp4       every frame with the last committed state drawn, plus a status line

"""

import argparse
import os
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

# TO DO: pocket name comes from the ball's position BEFORE the shot, so it's often wrong.
#   Take the pocket from the tracker's coast() event instead; use this only to confirm which balls are gone.
def events_between(balls_before, balls_after):
    """
    What happened between two committed states, as a list of strings.
        a colour has fewer balls  -> potted (the ball with no partner in the new state names the pocket)
        a colour has more balls   -> appeared (the cue ball put back after a scratch, or a ball that was hidden)
    Matching is generous (the whole table), so a ball that was merely hit still pairs with itself.
    """
    events = []
    for ball in match_balls(balls_before, balls_after, 10 * config.MAP_WIDTH):
        events.append('%s ball potted (last seen near the %s pocket)' % (ball['colour'], closest_pocket([(ball['map_x'], ball['map_y'])])[0]))
    for ball in match_balls(balls_after, balls_before, 10 * config.MAP_WIDTH):
        events.append('%s ball appeared at map (%.0f, %.0f)' % (ball['colour'], ball['map_x'], ball['map_y']))
    return events

# DISPLAY AND VIDEO OUTPUT
def draw_status(image, lines):
    """
    Writes the status lines along the top of an image.
    """
    for row, line in enumerate(lines):
        cv2.putText(image, line, (10, 30 + 28 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(image, line, (10, 30 + 28 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return image


def side_by_side(overlay, balls):
    """
    The overlay (left) and the top-down map (right) in one picture, 540 px high.
    """
    height = 540
    overlay = cv2.resize(overlay, (int(overlay.shape[1] * height / overlay.shape[0]), height))
    table_map = draw.draw_map(balls)
    table_map = cv2.resize(table_map, (520, int(table_map.shape[0] * 520 / table_map.shape[1])))
    picture = np.full((height, overlay.shape[1] + 15 + 520, 3), 255, dtype=np.uint8)
    picture[:, :overlay.shape[1]] = overlay
    top = (height - table_map.shape[0]) // 2
    picture[top:top + table_map.shape[0], overlay.shape[1] + 15:] = table_map
    return picture

def counts_text(counts):
    return 'cue %d  black %d  red %d  yellow %d' % (counts['cue'], counts['black'], counts['red'], counts['yellow'])

def main():
    parser = argparse.ArgumentParser(description='Board states and events of a pool game from a fixed camera.')
    parser.add_argument('video', help='path to the video file')
    parser.add_argument('--click', nargs=2, type=int, metavar=('X', 'Y'), help='a pixel on bare cloth in the first frame')
    parser.add_argument('--start', type=int, default=0, help='first frame to process')
    parser.add_argument('--end', type=int, default=None, help='last frame to process')
    parser.add_argument('--no-video', action='store_true', help='do not write the output video')
    parser.add_argument('--show', action='store_true', help='live preview (press q to stop)')
    arguments = parser.parse_args()

    video = cv2.VideoCapture(arguments.video)
    fps = video.get(cv2.CAP_PROP_FPS) or 30.0
    if arguments.start > 0:
        video.set(cv2.CAP_PROP_POS_FRAMES, arguments.start)
    ok, frame = video.read()
    if not ok:
        print('Could not read', arguments.video)
        return

    # One time set up 
    click_x, click_y = table_mask.find_cloth_center(frame)
    reference = make_reference(frame, click_x, click_y)
    print('thresholds: low %d  high %d' % (config.MOVE_LOW_FRACTION * reference['table_area_small'],
                                       config.MOVE_HIGH_FRACTION * reference['table_area_small']))

    os.makedirs('results', exist_ok=True)
    name = os.path.splitext(os.path.basename(arguments.video))[0]
    output_path = os.path.join('results', name + '_tracked.mp4')
    writer = None

    # What the loop remember from one frame to the next
    detector = {'state': 'SETTLED', 'still_count': 0, 'blocked': False, 'readable': False}
    previous_gray = None
    previous_reading = None          # the last reading in the current readable window
    board = None
    tracks = None
    first_hit = None
    frame_index = arguments.start

    while ok:
        if arguments.end is not None and frame_index > arguments.end:
            break
        t = frame_index / fps

        # Checks if the table still or if a shot is occuring
        # Counts pixels that changed since the last frame and update SHOT / SETTLED state machine
        gray = shrink_to_gray(frame)
        moving = moving_pixels(gray, previous_gray, reference)
        previous_gray = gray

        was_shot = detector['state'] == 'SHOT'
        readable = update_detector(detector, moving, False, reference)
        is_shot = detector['state'] == 'SHOT'

        # If shot has just started - give every ball from the last committed state a track
        if is_shot and not was_shot and board is not None:
            tracks = initialise_tracking(board, t)
            first_hit = None
            print('%6.1f s  shot started' % t)

        # During a shot - follow every ball frame by frame
        if is_shot and tracks is not None:
            _, events = track_frame(tracks, frame, t, 1.0 / fps, reference)
            for event in events:
                print('%6.1f s    %s' % (t, event))
            # if first_hit is None:
            #     first_hit = first_contact(tracks)

        # Shot ends - stop tracking
        if was_shot and not is_shot and tracks is not None:
            print('%6.1f s  shot over, first ball hit: %s' % (t, first_hit))
            tracks = None

        # When the table is still, read the board every READ_EVERY frames
        # Commit a reading if it agrees with the previous reading in the same readable window 
        # and something has changed since the last committed state
        if not readable:
            previous_reading = None
        elif frame_index % config.READ_EVERY == 0 and not view_is_blocked(frame, reference):
            reading = read_board(frame, reference, click_x, click_y)
            trusted = previous_reading is not None and same_reading(previous_reading, reading)
            changed = board is None or not same_reading(board, reading)
            if trusted and changed:
                if board is not None:
                    for event in events_between(board, reading):
                        print('%6.1f s    %s' % (t, event))
                board = reading
                print('%6.1f s  board committed: %s' % (t, counts_text(count_colours(board))))
            previous_reading = reading
 
        if tracks is not None:
            map_balls = [{'colour': track['colour'], 'map_x': track['x'], 'map_y': track['y']}
                         for track in tracks.values() if not track['lost']]
        else:
            map_balls = board or []
        overlay = draw.draw_overlay(frame, reference['table_hull'], reference['corners'], board or [])
        picture = draw_status(side_by_side(overlay, map_balls), ['t = %.1f s   %s' % (t, detector['state'])])
 
        if writer is None:
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                     (picture.shape[1], picture.shape[0]))
        writer.write(picture)

        ok, frame = video.read()
        if not ok:
            break
        frame_index = frame_index + 1
        if frame_index % 300 == 0:
            print('  ... %.0f s' % t)

    video.release()
    if writer is not None:
        writer.release()

if __name__ == '__main__':
    main()
