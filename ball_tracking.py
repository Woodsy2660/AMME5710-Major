import numpy as np
import config
from config import VELOCITY_BLEND, MAX_COAST_FRAMES, DROP_MIN_SPEED, POCKET_ZONE_RADII, NO_MATCH, POCKETS
from tracking_helpers import *
from scipy.optimize import linear_sum_assignment

def initialise_tracking(frame, t):
    """Give every ball in a clean starting frame a persistent ID."""
    tracked_balls, counts = {}, {}
    for ball in frame:
        colour = ball['colour']
        counts[colour] = counts.get(colour, 0) + 1
        if (colour == 'cue' or colour == 'black'):
            id = colour     # cue and black are unique, so no number is needed
        else:
            id = colour + str(counts[colour])   # e.g. red1, red2, yellow1, yellow2 etc

        tracked_balls[id] = {'colour': colour, 'x': ball['map_x'], 'y': ball['map_y'],
                           'vx': 0.0, 'vy': 0.0,
                           'seen_x': ball['map_x'], 'seen_y': ball['map_y'], 'seen_t': t,
                           'missed': 0, 'vanish': None, 'lost': False,
                           'path': [(ball['map_x'], ball['map_y'], True)]}
    return tracked_balls

def see(track, position, t):
    """ Called when a ball is found in its search window. Updates its position, velocity and last-seen time. """
    x, y = position
    dt = t - track['seen_t']
    if dt > 0:
        track['vx'] = VELOCITY_BLEND * (x - track['seen_x']) / dt + (1 - VELOCITY_BLEND) * track['vx']
        track['vy'] = VELOCITY_BLEND * (y - track['seen_y']) / dt + (1 - VELOCITY_BLEND) * track['vy']

    track.update(x=x, y=y, seen_x=x, seen_y=y, seen_t=t, missed=0, vanish=None)
    track['path'].append((x, y, True))

def closest_pocket(points):
    """ Name and distance of the pocket closest to any of the given map points. """
    best = None
    for x, y in points:
        for name, (pocket_x, pocket_y) in POCKETS.items():
            distance = np.hypot(x - pocket_x, y - pocket_y)
            if best is None or distance < best[1]:
                best = (name, distance)
    return best

def coast(ball_id, track, predicted):
    """ The ball was not found this frame. Move it to its prediction; drop it if it is moving and
    has been unseen too long. Returns an event string when it is dropped.
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
