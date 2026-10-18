import numpy as np
import os
import cv2
import argparse
import config, ball_colour, ball_detection, table_mask, table_corners, top_down, draw
from window_helpers import *
from scipy.optimize import linear_sum_assignment
# from ball_detection import is_near_pocket
# Compare the ball info (received from ball_detector.py) at every frame 
# Output game events
# Output which player's turn it is

ACTIVE, SUNK, LOST = 'active', 'sunk', 'lost'   # define possible ball states
                                                # active means on the table, sunkis in a pocket, lost is unknown (eg occluded)

N_BALLS = 16    # there are 16 balls (1 cue, 1 8-ball, then 14 (2x7) normal)
BALL_STATE = [ACTIVE, SUNK, LOST]   # all balls start ACTIVE
MATCH_GATE = 80        # max map distance to count as the same ball
SUNK_AFTER = 10        # frames missing near a pocket before calling it sunk

def initialise_tracking(first_frame_balls, t):
    """Give every ball in a clean starting frame a persistent ID."""
    tracked_balls, counts = {}, {}
    for b in first_frame_balls:
        colour = b['colour']
        counts[colour] = counts.get(colour, 0) + 1
        if (colour == 'cue' or colour == 'black'):
            id = colour
        else:
            id = colour + str(counts[colour])

        tracked_balls[id] = {'colour': colour, 'x': b['map_x'], 'y': b['map_y'],
                           'vx': 0.0, 'vy': 0.0,
                           'seen_x': b['map_x'], 'seen_y': b['map_y'], 'seen_t': t,
                           'missed': 0, 'vanish': None, 'lost': False,
                           'path': [(b['map_x'], b['map_y'], True)]}
    return tracked_balls

def see(track, position, t):
    """
    The ball was found at `position` (map) at time t.
    """
    x, y = position
    dt = t - track['seen_t']
    if dt > 0:
        track['vx'] = VELOCITY_BLEND * (x - track['seen_x']) / dt + (1 - VELOCITY_BLEND) * track['vx']
        track['vy'] = VELOCITY_BLEND * (y - track['seen_y']) / dt + (1 - VELOCITY_BLEND) * track['vy']
    track.update(x=x, y=y, seen_x=x, seen_y=y, seen_t=t, missed=0, vanish=None)
    track['path'].append((x, y, True))


def closest_pocket(points):
    """
    (pocket name, distance) of the pocket closest to any of the given map points.
    """
    best = None
    for x, y in points:
        for name, (pocket_x, pocket_y) in POCKETS.items():
            distance = np.hypot(x - pocket_x, y - pocket_y)
            if best is None or distance < best[1]:
                best = (name, distance)
    return best

def coast(ball_id, track, predicted):
    """
    The ball was not found this frame. Move it to its prediction; drop it if it is moving and
    has been unseen too long. Returns an event string when it is dropped, else None.
    """
    track['missed'] += 1
    if track['missed'] == 1:
        track['vanish'] = predicted                   # first place it should have been, but wasn't
    track['x'], track['y'] = predicted
    track['path'].append((predicted[0], predicted[1], False))

    speed = np.hypot(track['vx'], track['vy'])
    if track['missed'] <= MAX_COAST_FRAMES or speed < DROP_MIN_SPEED:
        return None

    track['lost'] = True
    pocket, distance = closest_pocket([(track['seen_x'], track['seen_y']), track['vanish']])
    if distance < POCKET_ZONE_RADII * config.BALL_RADIUS_ON_MAP:
        return '%s potted in the %s pocket' % (ball_id, pocket)
    return '%s lost (last seen at map %.0f, %.0f)' % (ball_id, track['seen_x'], track['seen_y'])

def first_contact(tracks):
    """
    Returns the first ball that was hit by the cue ball
    """
    # first one the move from its origin
    return None


def track_frame(tracks, frame_bgr, t, frame_dt, reference):
    """
    Updates every track from one frame.
    Output
    - windows : {ball_id: (predicted_x, predicted_y, half_size, photo_window)} for drawing
    - events : list of strings (pots, lost balls)
    """
    windows = {}
    candidates = {colour: [] for colour in FAMILY}
    for ball_id, track in tracks.items():
        if track['lost']:
            continue
        predicted_x =  track['seen_x'] + track['vx'] * (t - track['seen_t'])
        predicted_y = track['seen_y'] + track['vy'] * (t - track['seen_t'])
        half = window_half_size(track, frame_dt)
        window = photo_window(reference['to_photo'], predicted_x, predicted_y, half, frame_bgr.shape)
        windows[ball_id] = (predicted_x, predicted_y, half, window)
        if window is not None:
            candidates[track['colour']] += find_candidates(frame_bgr, window, track['colour'], reference)

    events = []
    for colour in FAMILY:
        found = merge_close(candidates[colour])
        ids = [ball_id for ball_id in windows if tracks[ball_id]['colour'] == colour]
        matched = set()

        if ids and found:
            cost = np.array([[np.hypot(windows[i][0] - x, windows[i][1] - y) for x, y in found] for i in ids])
            for row, ball_id in enumerate(ids):
                cost[row, cost[row] > windows[ball_id][2]] = NO_MATCH    # outside its window
            rows, cols = linear_sum_assignment(cost)
            for row, col in zip(rows, cols):
                if cost[row, col] < NO_MATCH:
                    see(tracks[ids[row]], found[col], t)
                    matched.add(ids[row])

        for ball_id in ids:
            if ball_id not in matched:
                event = coast(ball_id, tracks[ball_id], windows[ball_id][:2])
                if event:
                    events.append(event)
    return windows, events

    
def draw_windows(frame_bgr, tracks, windows):
    """
    The photo with every search window: green if the ball was found this frame, red if coasting.
    """
    picture = frame_bgr.copy()
    for ball_id, (_, _, _, window) in windows.items():
        if window is None:
            continue
        colour = (0, 200, 0) if tracks[ball_id]['missed'] == 0 else (0, 0, 255)
        # cv2.rectangle(picture, (window[0], window[1]), (window[2], window[3]), colour, 2)
        # cv2.putText(picture, ball_id, (window[0], window[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    return picture


def draw_paths(tracks):
    """
    The top-down map with every ball's path (in its colour where seen, grey where coasted).
    """
    on_table = [{'colour': track['colour'], 'map_x': track['x'], 'map_y': track['y']}
                for track in tracks.values() if not track['lost']]
    picture = draw.draw_map(on_table)
    for track in tracks.values():
        colour = draw.DRAW_COLOURS[track['colour']]
        path = track['path']
        for (xa, ya, _), (xb, yb, seen) in zip(path[:-1], path[1:]):
            a = (int(MAP_BORDER + xa), int(MAP_BORDER + ya))
            b = (int(MAP_BORDER + xb), int(MAP_BORDER + yb))
            # cv2.line(picture, a, b, colour if seen else (150, 150, 150), 2 if seen else 1)
    return picture


def side_by_side(left, right, height=540):
    left = cv2.resize(left, (int(left.shape[1] * height / left.shape[0]), height))
    right = cv2.resize(right, (int(right.shape[1] * height / right.shape[0]), height))
    return np.hstack([left, right])

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

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# def main():
#     parser = argparse.ArgumentParser(description='Minimal per-frame ball tracker for a short clip.')
#     parser.add_argument('video', help='path to the video file')
#     parser.add_argument('--click', nargs=2, type=int, metavar=('X', 'Y'), help='a pixel on bare cloth in the first frame')
#     parser.add_argument('--start', type=int, default=0, help='first frame (must show still balls, nothing over the table)')
#     parser.add_argument('--end', type=int, default=None, help='last frame to process')
#     parser.add_argument('--show', action='store_true', help='live preview (press q to stop)')
#     arguments = parser.parse_args()

#     os.makedirs('results', exist_ok=True)
#     name = os.path.splitext(os.path.basename(arguments.video))[0]
#     output_path = os.path.join('results', name + '_paths.mp4')

#     video = cv2.VideoCapture(arguments.video)
#     fps = video.get(cv2.CAP_PROP_FPS) or 30.0
#     if arguments.start > 0:
#         video.set(cv2.CAP_PROP_POS_FRAMES, arguments.start)
#     ok, frame = video.read()
#     if not ok:
#         print('Could not read', arguments.video)
#         return

#     # Set-up from the first frame: table, homography, and the starting balls.
#     # click_x, click_y = run_video.find_click(frame, arguments.click)
#     click_x, click_y = find_cloth_center(frame)
#     reference = run_video.make_reference(frame, click_x, click_y)
#     hull_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
#     cv2.fillConvexPoly(hull_mask, reference['table_hull'], 255)
#     reference['hull_mask'] = hull_mask

#     frame_index = arguments.start
#     t = frame_index / fps
#     tracks = initialise_tracking(run_video.read_board(frame, reference, click_x, click_y), t)
#     print('Cloth click:', click_x, click_y)
#     print('Tracking %d balls: %s' % (len(tracks), ', '.join(sorted(tracks))))

#     writer = None
#     previous_t = t
#     while True:
#         ok, frame = video.read()
#         if not ok:
#             break
#         frame_index += 1
#         if arguments.end is not None and frame_index > arguments.end:
#             break
#         t = frame_index / fps

#         windows, events = track_frame(tracks, frame, t, t - previous_t, reference)
#         previous_t = t
#         for event in events:
#             print('%6.2f s  %s' % (t, event))

#         picture = side_by_side(draw_windows(frame, tracks, windows), draw_paths(tracks))
#         if writer is None:
#             writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
#                                      (picture.shape[1], picture.shape[0]))
#         writer.write(picture)
#         if arguments.show:
#             cv2.imshow('ball tracking', picture)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#     video.release()
#     if writer is not None:
#         writer.release()
#     if arguments.show:
#         cv2.destroyAllWindows()

#     # Net movement from start to end (not path length, which would add up detection jitter).
#     print('\nSummary:')
#     r = config.BALL_RADIUS_ON_MAP
#     for ball_id in sorted(tracks):
#         track = tracks[ball_id]
#         start_x, start_y, _ = track['path'][0]
#         moved = np.hypot(track['x'] - start_x, track['y'] - start_y) / r
#         if track['lost']:
#             status = 'dropped'
#         elif track['missed']:
#             status = 'unseen for the last %d frames' % track['missed']
#         else:
#             status = 'on table'
#         print('  %-8s moved %5.1f radii   %s' % (ball_id, moved, status))
#     print('Video:', output_path)


# if __name__ == '__main__':
#     main()
