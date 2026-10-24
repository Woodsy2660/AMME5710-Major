import cv2
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
                           'free': False,   # may move away from where it is (see free_hit_balls)
                           'hidden': False, # ball is under an obstruction (arm, cue, body)
                           'path': [(ball['map_x'], ball['map_y'], True)]}
    return tracked_balls

def see(track, position, t):
    """ Called when a ball is found in its search window. Updates its position, velocity and last-seen time. """
    if not track['free']:
        # Still and not hit: found where it should be. Its position is not moved at all, so a
        # hand or the cue sliding past cannot drag it along a little every frame.
        track.update(seen_t=t, missed=0, vanish=None, vx=0.0, vy=0.0)
        track['path'].append((track['x'], track['y'], True))
        return
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

    if not track['free']:
        return None              # still and not hit: it is hidden (arm, cue, person), not gone
    speed = np.hypot(track['vx'], track['vy'])
    if track['missed'] <= MAX_COAST_FRAMES or speed < DROP_MIN_SPEED:
        return None

    pocket, distance = closest_pocket([(track['seen_x'], track['seen_y']), track['vanish']])
    near_pocket = distance < POCKET_ZONE_RADII * config.BALL_RADIUS_ON_MAP
    if track['colour'] == 'cue' and not near_pocket:
        # There is only one cue ball, so rather than give up on it (a fast cue ball is blurred and
        # often missed), stop predicting and keep searching round where it was last seen: its
        # window keeps growing (window_half_size) until it is found again.
        track['vx'] = track['vy'] = 0.0
        return None

    track['lost'] = True
    if near_pocket:
        return '%s potted in the %s pocket' % (ball_id, pocket)
    return '%s lost (last seen at map %.0f, %.0f)' % (ball_id, track['seen_x'], track['seen_y'])

def first_contact(tracks):
    """
    Returns the first ball that was hit by the cue ball
    """
    # first one the move from its origin
    return None

def free_hit_balls(tracks):
    """
    A still ball can only start moving when something hits it. Until a ball is 'free' it keeps
    a small search window round where it is, so an arm, a hand or the cue passing over it
    cannot drag its track away.
    - the cue ball is freed when it disappears from its small window for CUE_RELEASE_FRAMES
      frames in a row: it has been struck away (the cue tip sliding beside it does not do that)
    - any other ball is freed when a moving free ball's path since the last frame passes within
      HIT_RADII of it (it was hit). The whole path is checked, as a fast ball can travel
      several radii between two frames.
    """
    reach = config.HIT_RADII * config.BALL_RADIUS_ON_MAP
    for track in tracks.values():
        if not track['free'] and not track['lost'] and track['colour'] == 'cue' \
                and track['missed'] >= config.CUE_RELEASE_FRAMES:
            track['free'] = True
            track['missed'] = 0                      # start searching from a fresh, growing window
    movers = [track for track in tracks.values() if track['free'] and not track['lost'] and len(track['path']) > 1]
    for track in tracks.values():
        if track['free'] or track['lost']:
            continue
        for mover in movers:
            (x0, y0, _), (x1, y1, _) = mover['path'][-2], mover['path'][-1]
            if distance_to_segment(track['x'], track['y'], x0, y0, x1, y1) < reach:
                track['free'] = True
                break


def distance_to_segment(px, py, x0, y0, x1, y1):
    """Distance from point (px, py) to the line segment (x0, y0) - (x1, y1)."""
    dx, dy = x1 - x0, y1 - y0
    length2 = dx * dx + dy * dy
    if length2 == 0:
        return np.hypot(px - x0, py - y0)
    along = min(1.0, max(0.0, ((px - x0) * dx + (py - y0) * dy) / length2))
    return np.hypot(px - (x0 + along * dx), py - (y0 + along * dy))


def track_frame(tracks, frame_bgr, t, frame_dt, reference):
    """
    Updates every track from one frame.
    Output
    - windows : {ball_id: (predicted_x, predicted_y, half_size, photo_window)} for drawing
    - events : list of strings (pots, lost balls)
    """
    free_hit_balls(tracks)
    windows = {}
    candidates = {colour: [] for colour in FAMILY}
    for ball_id, track in tracks.items():
        if track['lost']:
            continue
        if track['free']:
            predicted_x = track['seen_x'] + track['vx'] * (t - track['seen_t'])
            predicted_y = track['seen_y'] + track['vy'] * (t - track['seen_t'])
        else:
            predicted_x, predicted_y = track['seen_x'], track['seen_y']    # still: stays put
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
                # Further from where it was last seen than any ball could roll in the time.
                track = tracks[ball_id]
                reach = config.TRACK_MAX_SPEED * max(t - track['seen_t'], frame_dt) + config.BALL_RADIUS_ON_MAP
                for col, (x, y) in enumerate(found):
                    if np.hypot(x - track['seen_x'], y - track['seen_y']) > reach:
                        cost[row, col] = NO_MATCH
            rows, cols = linear_sum_assignment(cost)
            for row, col in zip(rows, cols):
                if cost[row, col] < NO_MATCH:
                    see(tracks[ids[row]], found[col], t)
                    matched.add(ids[row])

        for ball_id in ids:
            if ball_id not in matched:
                track = tracks[ball_id]
                # A hidden ball coasts without being dropped - obstruction is temporary.
                # A moving (free) ball still counts as missed, so its search window keeps growing
                # (window_half_size). A still ball does not: it is just under a hand or the cue, and
                # counting it as missed would release the cue ball as if it had been struck.
                if track.get('hidden', False):
                    predicted = windows[ball_id][:2]
                    if track['free']:
                        track['missed'] += 1
                    if track['vanish'] is None:
                        track['vanish'] = predicted   # where it was when it went out of sight (used by coast)
                    track['x'], track['y'] = predicted
                    track['path'].append((predicted[0], predicted[1], False))
                    track['vx'] *= 0.95  # Velocity decays while hidden
                    track['vy'] *= 0.95
                    continue
                event = coast(ball_id, track, windows[ball_id][:2])
                if event:
                    events.append(event)
    find_cue_anywhere(tracks, frame_bgr, t, reference)
    return windows, events


def find_cue_anywhere(tracks, frame_bgr, t, reference):
    """
    There is only one cue ball, so when it has been missing for CUE_RELEASE_FRAMES frames it
    is searched for over the whole table, not just round where it was last seen. This finds it
    again after it was moved by hand since the last board reading (placed for the break, or after
    a scratch), and after a fast shot that left its window. A cue ball that is already rolling
    must still be within reach of where it was seen (TRACK_MAX_SPEED).
    """
    cue = tracks.get('cue')
    if cue is None or cue['lost'] or cue['missed'] < config.CUE_RELEASE_FRAMES:
        return
    x, y, width, height = cv2.boundingRect(reference['table_hull'])
    found = find_candidates(frame_bgr, (x, y, x + width, y + height), 'cue', reference)
    if cue['free']:
        reach = config.TRACK_MAX_SPEED * max(t - cue['seen_t'], 1e-3) + config.BALL_RADIUS_ON_MAP
        found = [p for p in found if np.hypot(p[0] - cue['seen_x'], p[1] - cue['seen_y']) <= reach]
    if not found:
        return
    best_x, best_y = min(found, key=lambda p: np.hypot(p[0] - cue['x'], p[1] - cue['y']))
    confirmed = any(seen for _, _, seen in cue['path'][1:])
    if not cue['free'] and confirmed:
        # It was seen sitting where it should be this shot, and now it is somewhere else: it has
        # been struck. Follow it as a rolling ball from here.
        cue['free'] = True
    if not cue['free']:
        # It was never where the last board reading put it (moved by hand since then):
        # start again from here, still.
        cue.update(x=best_x, y=best_y, seen_x=best_x, seen_y=best_y, seen_t=t, missed=0, vanish=None,
                   vx=0.0, vy=0.0, path=[(best_x, best_y, True)])
        return
    see(cue, (best_x, best_y), t)
