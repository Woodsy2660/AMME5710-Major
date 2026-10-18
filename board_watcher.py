"""
Step 7a: turn a video into shots for the scoring rules (game_state.py).

Uses run_video.py's tests to get trusted board states:
- is the table still (moving pixels), is the view clear (no arm or cue over the table)
- two readings half a second apart must agree before a state is kept

On top of that, this file decides whether a SHOT happened between two kept states,
and what went down. Kept states with no shot in between (someone leaned over the
table and stepped back, or leaned in to aim) are not shots, so they do not hand the
turn over.
"""

import config
import run_video

POCKET_SHORT_NAMES = {'top-left': 'TL', 'top-middle': 'TM', 'top-right': 'TR',
                      'bottom-left': 'BL', 'bottom-middle': 'BM', 'bottom-right': 'BR'}


def capped_counts(balls):
    """
    Ball counts per colour, capped at a full rack, so a ball detected twice
    cannot make it look like an extra ball went down later.
    """
    counts = run_video.count_colours(balls)
    counts['red'] = min(counts['red'], config.BALLS_PER_COLOUR)
    counts['yellow'] = min(counts['yellow'], config.BALLS_PER_COLOUR)
    counts['black'] = min(counts['black'], 1)
    counts['cue'] = min(counts['cue'], 1)
    return counts


def shot_result(balls_before, balls_after):
    """
    Inputs
    - balls_before, balls_after : two kept board states (ball lists from find_balls())
    Outputs
    - potted : record in the format of game_state.empty_potted()
    - note : which pockets, e.g. 'red TL, yellow BM' (empty if nothing went down)
    """
    before = capped_counts(balls_before)
    after = capped_counts(balls_after)
    potted = {'red': max(0, before['red'] - after['red']),
              'yellow': max(0, before['yellow'] - after['yellow']),
              'black': before['black'] > 0 and after['black'] == 0,
              'cue': before['cue'] > 0 and after['cue'] == 0}

    # Balls in the old state with no partner in the new one were potted;
    # the pocket nearest to where they were last seen is the one they went in.
    # Only name as many balls of each colour as were counted as potted.
    names_left = {'red': potted['red'], 'yellow': potted['yellow'],
                  'black': int(potted['black']), 'cue': int(potted['cue'])}
    gone = run_video.match_balls(balls_before, balls_after, 10 * config.MAP_WIDTH)
    pockets = []
    for ball in gone:
        if names_left[ball['colour']] == 0:
            continue
        names_left[ball['colour']] = names_left[ball['colour']] - 1
        pocket = run_video.nearest_pocket(ball['map_x'], ball['map_y'])
        pockets.append(f"{ball['colour']} {POCKET_SHORT_NAMES[pocket]}")
    return potted, ', '.join(pockets)


class BoardWatcher:

    def __init__(self, first_frame=None, click=None):
        """
        The table is found from the first frame (camera on a tripod, table clear).
        first_frame can be left out to test commit() on its own.
        """
        self.committed_balls = None          # the last kept board state
        self.shot_since_commit = False       # has the table seen a shot since then?
        self.status = 'settling'
        if first_frame is not None:
            self.calibrate(first_frame, click)

    def calibrate(self, frame, click=None):
        """Find the table (again). Keeps the game's last board state."""
        self.click_x, self.click_y = run_video.find_click(frame, click)
        self.reference = run_video.make_reference(frame, self.click_x, self.click_y)
        self.detector = {'state': 'SETTLED', 'still_count': 0, 'blocked': False, 'readable': False}
        self.previous_gray = None
        self.previous_reading = None
        self.frames_since_reading = config.READ_EVERY
        self.window_committed = False
        self.status = 'settling'

    def read(self, frame):
        """The ball list for one frame."""
        return run_video.read_board(frame, self.reference, self.click_x, self.click_y)

    def update(self, frame):
        """
        Call once per video frame. Follows run_video.main(), steps 1-4.
        Output: None, or a dict {'potted', 'note'} when a shot has just finished.
        """
        gray = run_video.shrink_to_gray(frame)
        moving = run_video.moving_pixels(gray, self.previous_gray, self.reference)
        self.previous_gray = gray
        readable = run_video.update_detector(self.detector, moving, False, self.reference)

        if self.detector['state'] == 'SHOT':
            self.shot_since_commit = True

        result = None
        if not readable:
            self.previous_reading = None
            self.window_committed = False
            self.frames_since_reading = config.READ_EVERY
        elif not self.window_committed:
            self.frames_since_reading = self.frames_since_reading + 1
            if self.frames_since_reading >= config.READ_EVERY:
                self.frames_since_reading = 0
                self.detector['blocked'] = run_video.view_is_blocked(frame, self.reference)
                if self.detector['blocked']:
                    self.previous_reading = None
                else:
                    reading = self.read(frame)
                    if self.previous_reading is not None and run_video.same_reading(self.previous_reading, reading):
                        self.window_committed = True
                        result = self.commit(reading)
                    self.previous_reading = reading

        if self.detector['state'] == 'SHOT':
            self.status = 'SHOT'
        elif self.detector['blocked']:
            self.status = 'view blocked'
        elif self.window_committed:
            self.status = 'settled'
        elif readable:
            self.status = 'reading'
        else:
            self.status = 'settling'
        return result

    def commit(self, balls, shot_happened=None):
        """
        Keep a new board state.
        Inputs
        - balls : the agreed ball list
        - shot_happened : force True for still photos (no motion to measure)
        Output: None, or {'potted', 'note'} if this state ends a shot
        """
        if shot_happened is not None:
            self.shot_since_commit = shot_happened

        before = self.committed_balls
        self.committed_balls = balls
        if before is None:
            return None                       # first view of the table
        if not self.shot_since_commit:
            return None                       # view was blocked then cleared: not a shot
        self.shot_since_commit = False

        # Movement over the table (a player leaning in to aim, then stepping back) is
        # only a shot if the balls ended up somewhere else.
        # Limitation: a complete miss of the cue ball also leaves every ball in place;
        # the swap-turn key in the GUI covers that.
        if run_video.same_reading(before, balls):
            return None

        potted, note = shot_result(before, balls)

        # After a scratch the next player puts the cue ball back by hand. The hand
        # movement looks like a shot, but only the cue ball came back: not a shot.
        cue_placed_back = capped_counts(before)['cue'] == 0 and capped_counts(balls)['cue'] > 0
        nothing_potted = potted['red'] == 0 and potted['yellow'] == 0 and not potted['black']
        if cue_placed_back and nothing_potted:
            return None

        return {'potted': potted, 'note': note}
