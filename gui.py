"""
Step 8: the game window. Live view, top-down map, scoreboard, event log and a
ground-truth logger for measuring scoring accuracy.

- board_watcher.py turns the video into shots (using run_video.py's still / blocked /
  two-readings-agree tests), and game_state.py applies the rules to each shot.
- The camera must be on a tripod and the first frame must show the whole, clear table
  (see run_video.py). Press k to find the table again if the camera is bumped.
- Whose go it is comes from the rules, not the camera, so u (undo) and s (swap turn)
  are there to correct the game when a shot is missed or invented.

Usage
    python gui.py                          webcam 0 (an iPhone via Continuity Camera)
    python gui.py --camera 1               another webcam
    python gui.py --video game.mp4         a recorded game
    python gui.py --photos "Table images"  step through still photos (space = next)
    python gui.py --players Ben Natalie    player names
    python gui.py --video game.mp4 --speed 2   play a recording at twice real time
    python gui.py --video clip.mov --mid-game --record demo.mp4 --headless
                                           clip from part-way through a game, saved as a video

Keys
    q / esc   quit                         space   pause (photos: next photo)
    u         undo last shot               s       swap turn
    n         new game                     k       find the table again
    r y b c   ground truth: a red / yellow / black / cue ball really went down
    enter     ground truth: save the shot just played (with whatever was typed)
"""

import argparse
import copy
import csv
import os
import time

import cv2
import numpy as np

import config
import draw
import run_video
from board_watcher import BoardWatcher
from game_state import GameState

# Window layout, in pixels
VIEW_HEIGHT = 720
PANEL_WIDTH = 560
MARGIN = 20

# Panel colours in BGR order
BACKGROUND = (32, 26, 22)
TEXT = (235, 235, 235)
FADED = (150, 150, 150)
GREEN = (90, 200, 120)
BLUE = (220, 150, 60)
RED = (60, 60, 230)
YELLOW = (0, 200, 255)
GROUP_COLOURS = {'red': RED, 'yellow': YELLOW, None: FADED}
STATUS_COLOURS = {'SHOT': BLUE, 'view blocked': RED, 'settled': GREEN, 'reading': YELLOW, 'settling': FADED}


class GroundTruthLog:
    """
    Writes one CSV row per shot: what the system decided ('system' rows) and what
    really happened ('truth' rows, typed in with r / y / b / c then enter).
    Compare the two afterwards for the scoring-accuracy results.
    """

    FIELDS = ['time', 'source', 'shot', 'shooter', 'red', 'yellow', 'black', 'cue', 'fouls', 'pockets']

    def __init__(self, folder='logs'):
        os.makedirs(folder, exist_ok=True)
        self.path = os.path.join(folder, time.strftime('game_%Y%m%d_%H%M%S.csv'))
        with open(self.path, 'w', newline='') as file:
            csv.writer(file).writerow(self.FIELDS)
        self.pending = {'red': 0, 'yellow': 0, 'black': False, 'cue': False}
        self.truth_shots = 0

    def write(self, source, shot, shooter, potted, fouls, pockets=''):
        row = [time.strftime('%H:%M:%S'), source, shot, shooter, potted['red'], potted['yellow'],
               int(potted['black']), int(potted['cue']), '; '.join(fouls), pockets]
        with open(self.path, 'a', newline='') as file:
            csv.writer(file).writerow(row)

    def save_pending(self):
        # Shooter is left blank: by the time enter is pressed the turn may have changed.
        self.truth_shots = self.truth_shots + 1
        self.write('truth', self.truth_shots, '', self.pending, [])
        self.pending = {'red': 0, 'yellow': 0, 'black': False, 'cue': False}

    def describe_pending(self):
        parts = []
        if self.pending['red']:
            parts.append(f"{self.pending['red']} red")
        if self.pending['yellow']:
            parts.append(f"{self.pending['yellow']} yellow")
        if self.pending['black']:
            parts.append('black')
        if self.pending['cue']:
            parts.append('cue')
        return ', '.join(parts) if parts else 'nothing'


# ---------------------------------------------------------------- drawing

def put_text(image, text, x, y, scale=0.55, colour=TEXT, bold=False):
    cv2.putText(image, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, scale, colour,
                2 if bold else 1, cv2.LINE_AA)


def resize_to_height(image, height):
    scale = height / image.shape[0]
    return cv2.resize(image, (int(image.shape[1] * scale), height))


def resize_to_width(image, width):
    scale = width / image.shape[1]
    return cv2.resize(image, (width, int(image.shape[0] * scale)))


def draw_player(panel, game, player, y):
    """One player's row: turn marker, name, colour, and seven potted/remaining dots."""
    is_turn = game.current == player and game.winner is None
    group = game.groups[player]

    if is_turn:
        cv2.rectangle(panel, (MARGIN - 8, y - 22), (PANEL_WIDTH - MARGIN + 8, y + 40), (60, 50, 44), -1)
        put_text(panel, '>', MARGIN, y, 0.7, GREEN, bold=True)

    put_text(panel, game.players[player], MARGIN + 22, y, 0.7, TEXT, bold=True)
    group_text = group if group is not None else 'open table'
    put_text(panel, group_text, MARGIN + 170, y, 0.55, GROUP_COLOURS[group])
    if game.fouls[player]:
        put_text(panel, f"fouls {game.fouls[player]}", MARGIN + 300, y, 0.5, FADED)
    if is_turn and game.visits_left > 1:
        put_text(panel, '2 visits', PANEL_WIDTH - MARGIN - 80, y, 0.5, GREEN, bold=True)

    # Filled dot = potted, hollow dot = still on the table.
    potted = game.balls_potted(player)
    for i in range(config.BALLS_PER_COLOUR):
        centre = (MARGIN + 32 + i * 30, y + 22)
        colour = GROUP_COLOURS[group]
        if i < potted:
            cv2.circle(panel, centre, 10, colour, -1, cv2.LINE_AA)
        else:
            cv2.circle(panel, centre, 10, colour, 1, cv2.LINE_AA)
    put_text(panel, f"{potted}/{config.BALLS_PER_COLOUR}", MARGIN + 32 + 7 * 30, y + 28, 0.55, TEXT)


def draw_panel(height, map_image, game, status, truth_log, message):
    panel = np.full((height, PANEL_WIDTH, 3), BACKGROUND, dtype=np.uint8)

    # Top-down map
    small_map = resize_to_width(map_image, PANEL_WIDTH - 2 * MARGIN)
    panel[MARGIN:MARGIN + small_map.shape[0], MARGIN:MARGIN + small_map.shape[1]] = small_map
    y = MARGIN + small_map.shape[0] + 40

    # Scoreboard
    draw_player(panel, game, 0, y)
    draw_player(panel, game, 1, y + 72)
    y = y + 140

    if game.winner is not None:
        put_text(panel, f"{game.players[game.winner]} WINS", MARGIN, y + 8, 0.9, GREEN, bold=True)
    else:
        put_text(panel, status.upper(), MARGIN, y, 0.55, STATUS_COLOURS.get(status, TEXT), bold=True)
    if message:
        put_text(panel, message, MARGIN + 180, y, 0.5, FADED)
    y = y + 30

    # Event log: newest last
    for line in game.log[-5:]:
        colour = RED if 'FOUL' in line or 'wins' in line else TEXT
        put_text(panel, line[:62], MARGIN, y, 0.47, colour)
        y = y + 22

    # Ground truth and key help along the bottom
    bottom = height - MARGIN
    put_text(panel, f"Truth for next shot: {truth_log.describe_pending()}   (saved: {truth_log.truth_shots})",
             MARGIN, bottom - 44, 0.47, YELLOW)
    put_text(panel, 'r y b c = really potted   enter = save   u = undo   s = swap turn',
             MARGIN, bottom - 22, 0.45, FADED)
    put_text(panel, 'n = new game   k = find table   space = pause   q = quit',
             MARGIN, bottom, 0.45, FADED)
    return panel


def render(frame, watcher, game, truth_log, message=''):
    """
    Builds the whole window image: the frame with the last kept board state drawn on it
    (left) and the panel (right).
    """
    balls = watcher.committed_balls or []
    view = draw.draw_overlay(frame, watcher.reference['table_hull'], watcher.reference['corners'], balls)
    view = resize_to_height(view, VIEW_HEIGHT)
    panel = draw_panel(VIEW_HEIGHT, draw.draw_map(balls), game, watcher.status, truth_log, message)
    return np.hstack([view, panel])


# ---------------------------------------------------------------- main loop

def open_source(args):
    """
    Returns a function that gives the next frame (or None when finished), and the
    frame rate to play at (None = as fast as possible: photos and live cameras).
    """
    if args.photos:
        names = sorted(n for n in os.listdir(args.photos)
                       if n.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')))
        paths = [os.path.join(args.photos, n) for n in names]

        def next_photo():
            if not paths:
                return None
            return cv2.imread(paths.pop(0))
        return next_photo, None

    capture = cv2.VideoCapture(args.video if args.video else args.camera)
    if not capture.isOpened():
        raise SystemExit('Could not open the camera or video')

    def next_frame():
        ok, frame = capture.read()
        return frame if ok else None

    # A recorded video is played at its real speed (times --speed), so ground truth can
    # be typed in while watching. Processing runs faster than real time, so it keeps up.
    play_fps = None
    if args.video:
        play_fps = (capture.get(cv2.CAP_PROP_FPS) or 30.0) * args.speed
    return next_frame, play_fps


def open_recording(path, picture, fps):
    """
    A video writer the size of the window picture. H.264 ('avc1') plays everywhere
    (QuickTime, phones); 'mp4v' is the fallback if this OpenCV build lacks it.
    """
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    size = (picture.shape[1], picture.shape[0])
    for codec in ('avc1', 'mp4v'):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer
    raise SystemExit(f'Could not open {path} for writing')


def main():
    parser = argparse.ArgumentParser(description='8-ball pool scoring window')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--video')
    parser.add_argument('--photos')
    parser.add_argument('--players', nargs=2, default=['Player 1', 'Player 2'])
    parser.add_argument('--speed', type=float, default=1.0,
                        help='video playback speed, e.g. 2 = twice real time')
    parser.add_argument('--click', nargs=2, type=int, metavar=('X', 'Y'),
                        help='a pixel on bare cloth in the first frame')
    parser.add_argument('--mid-game', action='store_true',
                        help='the video starts part-way through a game: take the ball counts from the first board')
    parser.add_argument('--first-player-colour', choices=['red', 'yellow'],
                        help='with --mid-game: the colour the first named player is on')
    parser.add_argument('--moving-camera', action='store_true',
                        help='handheld camera: find the table again at every board reading')
    parser.add_argument('--record', metavar='PATH', help='also save the window as a video (.mp4)')
    parser.add_argument('--headless', action='store_true',
                        help='no window (use with --record): process the whole video and stop')
    args = parser.parse_args()

    next_frame, play_fps = open_source(args)
    photo_mode = args.photos is not None

    frame = next_frame()
    if frame is None:
        raise SystemExit('No frames to read')
    watcher = BoardWatcher(frame, args.click, args.moving_camera)

    game = GameState(args.players)
    truth_log = GroundTruthLog()
    history = []                       # copies of the game before each shot, for undo
    message = ''
    paused = False
    first = True
    next_show_time = time.time()
    joined_mid_game = False
    writer = None

    if not args.headless:
        cv2.namedWindow('Pool', cv2.WINDOW_NORMAL)

    while True:
        if first or not paused:
            if not first:
                new_frame = next_frame()
                if new_frame is None:
                    message = 'end of input'
                    paused = True
                    new_frame = frame
                else:
                    frame = new_frame
            first = False

            if message != 'end of input':
                try:
                    if photo_mode:
                        # Photos have no motion to measure and may each be from a different
                        # angle: find the table every time and treat each photo as one shot.
                        watcher.calibrate(frame)
                        result = watcher.commit(watcher.read(frame), shot_happened=True)
                        watcher.status = 'settled'
                        paused = True                 # wait for space before the next photo
                    else:
                        result = watcher.update(frame)
                except Exception as error:            # e.g. the table cannot be found
                    message = f"error: {error}"[:40]
                    result = None

                # Picking up a game in progress: the first kept board sets the ball counts.
                if args.mid_game and not joined_mid_game and watcher.committed_balls is not None:
                    counts = run_video.count_colours(watcher.committed_balls)
                    game.start_mid_game(counts['red'], counts['yellow'], args.first_player_colour)
                    joined_mid_game = True

                if result is not None and game.winner is None:
                    history.append(copy.deepcopy(game))
                    shooter = game.current
                    event = game.apply_shot(result['potted'], result['note'])
                    truth_log.write('system', game.shot_number, game.players[shooter],
                                    result['potted'], event['fouls'], result['note'])

        picture = render(frame, watcher, game, truth_log, message)
        if args.record and (not paused or writer is None):
            if writer is None:
                writer = open_recording(args.record, picture, play_fps or 30.0)
            writer.write(picture)
        if args.headless:
            if message == 'end of input':
                break
            continue
        cv2.imshow('Pool', picture)

        # Hold each video frame until its time comes, so the video plays at real speed.
        wait_ms = 30 if paused else 1
        if play_fps and not paused:
            next_show_time = max(next_show_time + 1.0 / play_fps, time.time() - 0.5)
            wait_ms = max(1, int((next_show_time - time.time()) * 1000))
        key = cv2.waitKey(wait_ms) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            paused = not paused
            if message in ('paused', ''):
                message = 'paused' if paused else ''
        elif key == ord('u'):
            if history:
                game = history.pop()
                message = 'undid last shot'
        elif key == ord('s'):
            history.append(copy.deepcopy(game))
            game.current = 1 - game.current
            game.visits_left = 1
            message = 'swapped turn'
        elif key == ord('n'):
            game = GameState(args.players)
            watcher.committed_balls = None
            watcher.shot_since_commit = False
            history = []
            message = 'new game'
        elif key == ord('k'):
            watcher.calibrate(frame, args.click)
            message = 'found table again'
        elif key == ord('r'):
            truth_log.pending['red'] = truth_log.pending['red'] + 1
        elif key == ord('y'):
            truth_log.pending['yellow'] = truth_log.pending['yellow'] + 1
        elif key == ord('b'):
            truth_log.pending['black'] = True
        elif key == ord('c'):
            truth_log.pending['cue'] = True
        elif key in (13, 10):
            truth_log.save_pending()
            message = 'truth saved'

    if writer is not None:
        writer.release()
        print(f"Recording saved to {args.record}")
    if not args.headless:
        cv2.destroyAllWindows()
    print(f"Log saved to {truth_log.path}")


if __name__ == '__main__':
    main()
