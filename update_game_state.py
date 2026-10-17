import numpy as np
from scipy.optimize import linear_sum_assignment
from archive.ball_detection import is_near_pocket
# Compare the ball info (received from ball_detector.py) at every frame 
# Output game events
# Output which player's turn it is

ACTIVE, SUNK, LOST = 'active', 'sunk', 'lost'   # define possible ball states
                                                # active means on the table, sunkis in a pocket, lost is unknown (eg occluded)

N_BALLS = 16    # there are 16 balls (1 cue, 1 8-ball, then 14 (2x7) normal)
BALL_STATE = [ACTIVE, SUNK, LOST]   # all balls start ACTIVE
MATCH_GATE = 80        # max map distance to count as the same ball
SUNK_AFTER = 10        # frames missing near a pocket before calling it sunk

def initialise_tracking(first_frame_balls):
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
                            'state': ACTIVE, 'frames_missing': 0}
    return tracked_balls

def match_detections(tracked_balls, current_balls):
    matches = {}
    for colour in ('cue', 'black', 'red', 'yellow'):
        ids = []
        dets = []

        # create a list of balls which haven't been sunk from the initial list
        for ball_id, info in tracked_balls.items():
            if info['colour'] == colour and info['state'] != SUNK:
                ids.append(ball_id)

        # list the current ball detections
        for b in current_balls:
            if b['colour'] == colour:
                dets.append(b)

        # if not in these lists, continue
        if not ids or not dets: 
            continue

        # match current ball detections to initial ids 
        cost = np.array([[np.hypot(tracked_balls[i]['x'] - d['map_x'], tracked_balls[i]['y'] - d['map_y']) for d in dets] for i in ids])
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            if cost[r, c] < MATCH_GATE:
                matches[ids[r]] = dets[c]

    return matches


def compare_frames(tracked_balls, matches):
    events = []

    for ball_id, last_seen in tracked_balls.items():
        if last_seen['state'] == SUNK:
            continue
        
        if ball_id in matches:                       # seen this frame
            current = matches[ball_id]
            last_seen['x'], last_seen['y'] = current['map_x'], current['map_y']
            last_seen['frames_missing'] = 0
            if last_seen['state'] == LOST:
                events.append(('reappeared', ball_id))
            last_seen['state'] = ACTIVE

        else:                                         # not seen this frame
            last_seen['frames_missing'] += 1
            if last_seen['frames_missing'] >= SUNK_AFTER and is_near_pocket(last_seen['x'], last_seen['y']):
                last_seen['state'] = SUNK
                events.append(('sunk', ball_id))
            else:
                last_seen['state'] = LOST
    
    return events

    
