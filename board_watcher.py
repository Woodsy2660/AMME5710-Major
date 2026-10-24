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

import cv2
import numpy as np

import ball_tracking
from ball_tracking import initialise_tracking, track_frame, first_contact
import config
import board_reading
import table_mask
import top_down
from shot_detection import shrink_to_gray, moving_pixels, update_detector, view_is_blocked
import obstruction


POCKET_SHORT_NAMES = {'top-left': 'TL', 'top-middle': 'TM', 'top-right': 'TR',
                      'bottom-left': 'BL', 'bottom-middle': 'BM', 'bottom-right': 'BR'}


def capped_counts(balls):
    """
    Ball counts per colour, capped at a full rack, so a ball detected twice
    cannot make it look like an extra ball went down later.
    """
    counts = board_reading.count_colours(balls)
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
    gone = board_reading.match_balls(balls_before, balls_after, 10 * config.MAP_WIDTH)
    pockets = []
    for ball in gone:
        if names_left[ball['colour']] == 0:
            continue
        names_left[ball['colour']] = names_left[ball['colour']] - 1
        pocket = ball_tracking.closest_pocket([(ball['map_x'], ball['map_y'])])[0]
        pockets.append(f"{ball['colour']} {POCKET_SHORT_NAMES[pocket]}")
    return potted, ', '.join(pockets)


class BoardWatcher:

    def __init__(self, first_frame=None, click=None, fps = 30.0, moving_camera=False):
        """
        The table is found from the first frame (camera on a tripod, table clear).
        moving_camera: the camera may be moved or bumped during the game, so the table is
        found again before every board reading (see refind_table).
        first_frame can be left out to test commit() on its own.
        """
        self.moving_camera = moving_camera
        self.committed_balls = None          # the last kept board state
        self.status_note = None              # e.g. 'player over table' while tracking is paused
        self.shot_since_commit = False       # has the table seen a shot since then?
        self.status = 'settling'
        if first_frame is not None:
            self.calibrate(first_frame, click)
        self.fps = fps
        self.frame_index = 0
        self.tracks = None
        self.last_tracks = None   # where the tracker left the balls when the shot ended
        self.pot_events = []      # (colour, pocket) pairs seen by the tracker since the last kept state

    def calibrate(self, frame, click=None):
        """Find the table (again). Keeps the game's last board state."""
        self.click_x, self.click_y = table_mask.find_cloth_center(frame)
        self.reference = board_reading.make_reference(frame, self.click_x, self.click_y)
        self.first_table_area = self.reference['table_area_small']
        self.detector = {'state': 'SETTLED', 'still_count': 0, 'blocked': False, 'readable': False}
        self.previous_gray = None
        self.previous_reading = None
        self.frames_since_reading = config.READ_EVERY
        self.window_committed = False
        self.status = 'settling'
        self.tracks = None
        self.last_tracks = None
        self.camera_was_moved = False
        self.shift_window = None
        # Obstruction detection filters (from obstruction.py)
        self.blocked_filter = obstruction.new_blocked_filter()
        self.lump_mask = None  # Stored for per-ball hidden detection during shots

    # def read(self, frame):
    #     """The ball list for one frame."""
    #     return run_video.read_board(frame, self.reference, self.click_x, self.click_y)

    def update(self, frame):
        """
        Call once per video frame. Follows run_video.main(), steps 1-4.
        Output: None, or a dict {'potted', 'note'} when a shot has just finished.
        """

        t = self.frame_index / self.fps # TO DO dont hardcode fps = 30
        self.frame_index = self.frame_index + 1
        # Count the moving pixels on a small grayscale version of the frame, to detect a shot.
        self.status_note = None
        gray = shrink_to_gray(frame)
        moving = moving_pixels(gray, self.previous_gray, self.reference)

        # Obstruction detection - get lump mask for per-ball hidden detection
        # Wrapped in try/except so original tracking still works if obstruction module fails
        try:
            self.lump_mask, lumps = obstruction.unexplained_lumps(frame, self.reference)
            blocked_now = obstruction.view_is_blocked(lumps)
            obstruction.update_blocked(self.blocked_filter, blocked_now)
        except Exception:
            self.lump_mask = None  # Fall back if obstruction detection fails

        camera_moving = self.camera_moved(gray)
        self.previous_gray = gray
        if camera_moving:
            # Someone has knocked or picked up the camera: every position from here on is off.
            # Stop tracking (the GUI falls back to the last kept board) and find the table again
            # before the next reading.
            self.tracks = None
            self.last_tracks = None
            self.camera_was_moved = True

        # Update the shot detectors state machine
        was_shot = self.detector['state'] == 'SHOT'
        readable = update_detector(self.detector, moving, False, self.reference)
        is_shot = self.detector['state'] == 'SHOT'
        
        
       
        # Tracking balls during a shot
        if is_shot:
            self.shot_since_commit = True #remember if pixel movement if due to an actual shot or other occlusion eg arm

            # shot has just started - give every ball from the last committed state a track
            if not was_shot and self.committed_balls is not None:
                self.tracks = initialise_tracking(self.committed_balls, t)
                self.last_tracks = None

            # Mark each ball as hidden or visible based on the obstruction mask.
            # A hidden ball continues to coast and won't be falsely marked as potted.
            if self.tracks is not None and self.lump_mask is not None:
                try:
                    occluder = obstruction.occluder_mask(self.lump_mask)
                    tracks_as_balls = [{'map_x': t['x'], 'map_y': t['y']} for t in self.tracks.values() if not t['lost']]
                    hidden_flags = obstruction.hidden_balls(tracks_as_balls, occluder, self.reference)
                    visible_tracks = [t for t in self.tracks.values() if not t['lost']]
                    for track, is_hidden in zip(visible_tracks, hidden_flags):
                        track['hidden'] = is_hidden
                except Exception:
                    pass  # Fall back to no hidden detection if it fails

            # Someone leaning right over the balls hides them and their clothes can look like
            # balls: hold every ball where it is until they move away. (A player bent over a
            # corner to take a shot is just as big, but is not over the balls, so is ignored.)
            body_over_table = self.tracks is not None and self.covers_balls(frame)
            if body_over_table:
                self.status_note = 'player over table'
            if self.tracks is not None and not camera_moving and not body_over_table:
                # follow every ball frame by frame
                _, events = track_frame(self.tracks, frame, t, 1.0/self.fps, self.reference)
                for event in events:
                    #TO DO coast() reports e.g. 'red3 potted in the top-left pocket'.
                    # Only the colour and pocket are kept: which red it was does not matter
                    # for scoring. 'lost' events (dropped away from a pocket) are ignored;
                    # the board counts in commit() decide what really went down.
                    if ' potted in the ' in event:
                        ball_id, pocket = event[:-len(' pocket')].split(' potted in the ')
                        self.pot_events.append((self.tracks[ball_id]['colour'], pocket))

        elif was_shot:
            # Stop tracking, but keep where the balls ended up so the GUI does not jump
            # back to the pre-shot positions while the table settles.
            self.last_tracks = self.tracks
            self.tracks = None

        result = None
        if not readable:
            self.previous_reading = None
            self.window_committed = False
            self.frames_since_reading = config.READ_EVERY
        elif not self.window_committed:
            self.frames_since_reading = self.frames_since_reading + 1
            if self.frames_since_reading >= config.READ_EVERY:
                self.frames_since_reading = 0
                # Blocked if the old area test says so, or obstruction.py has seen an arm, hand,
                # cue or body over the table in the last CLEAR_FRAMES frames (its smoothed flag).
                self.detector['blocked'] = view_is_blocked(frame, self.reference) or self.blocked_filter['blocked']
                if self.detector['blocked']:
                    self.previous_reading = None
                else:
                    if self.moving_camera or self.camera_was_moved:
                        self.refind_table(frame, allow_big_move=self.camera_was_moved)
                        self.camera_was_moved = False
                    reading = board_reading.read_board(frame, self.reference, self.click_x, self.click_y)
                    if self.previous_reading is not None and board_reading.same_reading(self.previous_reading, reading):
                        self.window_committed = True
                        result = self.commit(reading)
                        self.pot_events = []
                        self.last_tracks = None       # the kept board takes over from here
                    self.previous_reading = reading

        # TO DO - why are these displayed? is this helpful?
        if self.camera_was_moved:
            self.status = 'camera moved'
        elif self.status_note:
            self.status = self.status_note
        elif self.detector['state'] == 'SHOT':
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

    def covers_balls(self, frame):
        """
        True if something big that is not cloth (more than TRACK_PAUSE_OCCLUDER_FRACTION of the
        table, as in shot_detection.view_is_blocked) lies over at least two of the tracked balls.
        An arm or the cue while aiming covers a few percent of the table; a person leaning over
        the balls a quarter or more.
        """
        scale = config.SIGNAL_SCALE
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        _, cloth_small = table_mask.find_cloth_mask(small, 0, 0, self.reference['cloth_colour'])
        not_cloth = cv2.bitwise_and(self.reference['inner_small'], cv2.bitwise_not(cloth_small))
        not_cloth = cv2.morphologyEx(not_cloth, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        count, lumps, stats, _ = cv2.connectedComponentsWithStats(not_cloth)
        big = [i for i in range(1, count)
               if stats[i, 4] > config.TRACK_PAUSE_OCCLUDER_FRACTION * self.reference['table_area_small']]
        if not big:
            return False
        covered = 0
        height, width = lumps.shape
        for track in self.tracks.values():
            if track['lost']:
                continue
            photo_x, photo_y = top_down.move_point(self.reference['to_photo'], track['x'], track['y'])
            x, y = int(photo_x * scale), int(photo_y * scale)
            if 0 <= x < width and 0 <= y < height and lumps[y, x] in big:
                covered = covered + 1
        return covered >= 2

    def camera_moved(self, gray):
        """
        True if the whole picture has shifted since the last frame (the camera was knocked or
        moved), measured by phase correlation of the small grey frames. In normal play, even
        during shots, the picture moves under a pixel; a knocked camera moves it several.
        """
        if self.previous_gray is None:
            return False
        if self.shift_window is None or self.shift_window.shape != gray.shape:
            self.shift_window = cv2.createHanningWindow(gray.shape[::-1], cv2.CV_32F)
        (dx, dy), _ = cv2.phaseCorrelate(self.previous_gray.astype(np.float32), gray.astype(np.float32),
                                        self.shift_window)
        return np.hypot(dx, dy) > config.CAMERA_SHIFT_PX

    def refind_table(self, frame, allow_big_move=False):
        """
        Find the table again, for a camera that has been moved or bumped. Only called when
        the table is still and the view is clear, never during a shot, so tracking is not
        disturbed. A result whose table area is very different from the first one is a failed
        detection (someone in front of the camera), so the last good table is kept instead.
        """
        try:
            click_x, click_y = table_mask.find_cloth_center(frame)
            reference = board_reading.make_reference(frame, click_x, click_y)
        except Exception:
            return
        area_ratio = reference['table_area_small'] / self.first_table_area
        if abs(area_ratio - 1) > config.REFIND_AREA_TOLERANCE:
            return

        # The same four corners can come back numbered from a different corner, which would
        # turn the top-down map round (balls jump to the other end of the table between shots).
        # The camera only moves a little, so keep the numbering that best matches the last table.
        old = self.reference['corners']
        new = reference['corners']
        best = min(range(4), key=lambda shift: np.abs(np.roll(new, -shift, axis=0) - old).sum())
        if not allow_big_move and np.abs(np.roll(new, -best, axis=0) - old).max() > config.REFIND_MAX_CORNER_SHIFT:
            return                                    # too far for a camera to move between readings
        if best:
            reference['corners'] = np.roll(new, -best, axis=0)
            reference['to_map'], reference['to_photo'] = top_down.make_map_transform(reference['corners'])
        self.reference = reference
        self.click_x, self.click_y = click_x, click_y

    def live_balls(self):
        """
        Where the balls are right now, for drawing.
        - during a shot, and until the next board state is kept: where ball_tracking last
          actually found each ball (never a predicted position); potted / lost balls left out
        - otherwise: the last kept board state
        Each record is a ball record (as from find_balls) plus
        - 'seen' : True (only real sightings are drawn)
        - 'path' : where the ball has rolled this shot (empty for a ball that was not hit)
        """
        tracks = self.tracks if self.tracks is not None else self.last_tracks
        if tracks is None:
            return [dict(ball, seen=True, path=[]) for ball in (self.committed_balls or [])]

        balls = []
        to_map = self.reference['to_map']
        to_photo = self.reference['to_photo']
        for track in tracks.values():
            if track['lost']:
                continue
            photo_x, photo_y = top_down.move_point(to_photo, track['seen_x'], track['seen_y'])
            radius = top_down.ball_radius_in_photo(to_map, to_photo, photo_x, photo_y)
            path = [point for point in track['path'] if point[2]] if track['free'] else []
            balls.append({'colour': track['colour'], 'map_x': track['seen_x'], 'map_y': track['seen_y'],
                          'photo_x': photo_x, 'photo_y': photo_y, 'radius': radius, 'seen': True,
                          'path': path})
        return balls

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
        # A shot moves at least one ball well away (the cue ball, at least). A ball whose centre
        # is measured a radius or so differently between two readings (a ball half-hidden against
        # a cushion) is not a shot, so the balls must have moved more than SHOT_MIN_MOVE_RADII.
        same_counts = board_reading.count_colours(before) == board_reading.count_colours(balls)
        reach = config.SHOT_MIN_MOVE_RADII * config.BALL_RADIUS_ON_MAP
        if same_counts and not board_reading.match_balls(before, balls, reach):
            return None

        potted, note = shot_result(before, balls)

        # After a scratch the next player puts the cue ball back by hand. The hand
        # movement looks like a shot, but only the cue ball came back: not a shot.
        cue_placed_back = capped_counts(before)['cue'] == 0 and capped_counts(balls)['cue'] > 0
        nothing_potted = potted['red'] == 0 and potted['yellow'] == 0 and not potted['black']
        if cue_placed_back and nothing_potted:
            return None

        return {'potted': potted, 'note': note}
