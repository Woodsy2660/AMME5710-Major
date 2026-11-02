"""
Step 8: the live tracking window. The camera on the left, a top-down table on the right,
with every ball drawn where it is now and a fading trail behind each rolling ball.

- board_watcher.py follows the game: between shots the balls come from the last agreed board
  reading; during a shot they come from ball_tracking.py, frame by frame.
- The camera should be on a tripod, with the first frame showing the whole table.
  Press k to find the table again if the camera is bumped (or use --moving-camera).
- A scoreboard along the bottom shows whose turn it is, and each player's colour, fouls and balls
  potted (from game_state.py). u (undo), s (swap turn) and c (swap colours) correct it if it goes wrong.

Usage
    python gui.py                          webcam 0 (an iPhone via Continuity Camera)
    python gui.py --camera 1               another webcam
    python gui.py --video game.mp4         a recorded game
    python gui.py --photos "Table images"  step through still photos (space = next)
    python gui.py --video game.mp4 --speed 2               play at twice real time
    python gui.py --video game.mp4 --start 120 --end 200   only part of a recording (seconds)
    python gui.py --video game.mp4 --record demo.mp4 --headless   save the window as a video
    python gui.py --video game.mp4 --start 31 --mid-game --player1-colour yellow --to-play 1
                                           start after the break, with the game as it stood then

Keys
    space   pause (photos: next photo)     k   find the table again     q / esc   quit
    u       undo the last shot             s   swap turn                c   swap colours
    f       the last shot was not a foul    x   the last shot was a foul (missed by the camera)
    n       new game
"""

import argparse
import copy
import os
import time

import cv2
import numpy as np

import board_reading
import config
import draw
import top_down
from board_watcher import BoardWatcher
from game_state import GameState

# Layout, in pixels: video and table side by side at the same height, the scoreboard below.
ROW_HEIGHT = 600
SCORE_HEIGHT = 74
GAP = 12

BACKGROUND = (28, 24, 20)            # BGR
TEXT = (235, 235, 235)
FADED = (150, 150, 150)

GROUP_COLOURS = {'red': (60, 60, 225), 'yellow': (0, 195, 250), None: (130, 130, 130)}   # BGR

TURN_COLOUR = (110, 200, 120)        # BGR

TRAIL_POINTS = 120                   # how much of each ball's path this shot is drawn (frames)


def resize_to_height(image, height):
    return cv2.resize(image, (int(image.shape[1] * height / image.shape[0]), height),
                      interpolation=cv2.INTER_AREA)


def ball_colour(ball):
    """BGR colour for drawing a ball. The black gets a light outline elsewhere so it shows."""
    return draw.DRAW_COLOURS[ball['colour']]


# ---------------------------------------------------------------- the camera view

def draw_camera(frame, balls, reference):
    """
    The camera frame with a thin ring round each ball in its own colour, and the trail of
    each rolling ball this shot.
    """
    picture = frame.copy()
    to_photo = reference['to_photo']
    trails = [ball for ball in balls if len(ball['path']) > 1]
    if trails:
        overlay = picture.copy()
        for ball in trails:
            colour = (235, 235, 235) if ball['colour'] == 'black' else ball_colour(ball)
            points = [top_down.move_point(to_photo, x, y) for x, y, _ in ball['path'][-TRAIL_POINTS:]]
            points = [(int(x), int(y)) for x, y in points]
            for i in range(1, len(points)):
                thickness = 1 + (5 * i) // len(points)            # thicker towards the ball
                cv2.line(overlay, points[i - 1], points[i], colour, thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.7, picture, 0.3, 0, dst=picture)

    for ball in balls:
        colour = (235, 235, 235) if ball['colour'] == 'black' else ball_colour(ball)
        centre = (int(ball['photo_x']), int(ball['photo_y']))
        radius = int(ball['radius'] * 1.3) + 2
        cv2.circle(picture, centre, radius, (0, 0, 0), 4, cv2.LINE_AA)   # dark edge so it shows on any background
        cv2.circle(picture, centre, radius, colour, 2, cv2.LINE_AA)
    return picture


# ---------------------------------------------------------------- the top-down table

def draw_table(balls):
    """
    The top-down table (draw.draw_map) with each rolling ball's trail this shot, turned 180
    degrees to match the camera (as on Ben's branch).
    """
    picture = draw.draw_map(balls)
    border = config.MAP_BORDER
    radius = int(round(config.BALL_RADIUS_ON_MAP)) + 1
    for ball in balls:
        points = [(int(border + x), int(border + y)) for x, y, _ in ball['path'][-TRAIL_POINTS:]]
        for i in range(1, len(points)):
            thickness = 1 + (4 * i) // len(points)                # thicker towards the ball
            cv2.line(picture, points[i - 1], points[i], ball_colour(ball), thickness, cv2.LINE_AA)
    # Redraw the balls on top of their trails, with an outline (white for the black ball).
    for ball in balls:
        centre = (int(border + ball['map_x']), int(border + ball['map_y']))
        cv2.circle(picture, centre, radius, ball_colour(ball), -1, cv2.LINE_AA)
        edge = (230, 230, 230) if ball['colour'] == 'black' else (0, 0, 0)
        cv2.circle(picture, centre, radius, edge, 1, cv2.LINE_AA)
    return cv2.flip(picture, -1)


# ---------------------------------------------------------------- the scoreboard

def text(picture, words, x, y, scale, colour, bold=False):
    cv2.putText(picture, words, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, scale, colour,
                2 if bold else 1, cv2.LINE_AA)


def draw_player(picture, game, player, x, width, top):
    """One player: turn marker, name, colour, foul count, and balls potted (dots and count)."""
    is_turn = game.current == player and game.winner is None
    group = game.groups[player]
    colour = GROUP_COLOURS[group]
    if is_turn:
        cv2.rectangle(picture, (x, top + 8), (x + width, top + SCORE_HEIGHT - 8), (52, 46, 40), -1)
        cv2.rectangle(picture, (x, top + 8), (x + 5, top + SCORE_HEIGHT - 8), TURN_COLOUR, -1)
    text(picture, f"Player {player + 1}", x + 18, top + 34, 0.75, TEXT if is_turn else FADED, bold=True)
    text(picture, group if group else 'open table', x + 18, top + 58, 0.5, colour)
    tag_x = x + 110
    if is_turn:
        text(picture, 'TO PLAY', tag_x, top + 58, 0.45, TURN_COLOUR, bold=True)
        tag_x = tag_x + 80
    fouls = game.fouls[player]
    text(picture, f"{fouls} foul{'' if fouls == 1 else 's'}", tag_x, top + 58, 0.45, FADED, bold=True)
    potted = game.balls_potted(player)
    total = game.totals[group] if group else 7
    dots_x = x + width - 240
    for i in range(total):
        cv2.circle(picture, (dots_x + i * 26, top + 37), 9, colour, -1 if i < potted else 1, cv2.LINE_AA)
    text(picture, f"{potted}/{total}", dots_x + 182, top + 44, 0.7, TEXT, bold=True)


def draw_scoreboard(picture, game, top, width):
    """Player 1 on the left, Player 2 on the right: whose turn, fouls and balls potted."""
    block = min(560, (width - 3 * GAP) // 2)
    draw_player(picture, game, 0, GAP, block, top)
    draw_player(picture, game, 1, width - GAP - block, block, top)


# ---------------------------------------------------------------- the whole window

def render(frame, watcher, game):
    balls = watcher.live_balls()
    camera = resize_to_height(draw_camera(frame, balls, watcher.reference), ROW_HEIGHT)
    table = resize_to_height(draw_table(balls), ROW_HEIGHT)

    width = camera.shape[1] + GAP + table.shape[1]
    picture = np.full((ROW_HEIGHT + SCORE_HEIGHT, width, 3), BACKGROUND, dtype=np.uint8)
    picture[:ROW_HEIGHT, :camera.shape[1]] = camera
    picture[:ROW_HEIGHT, camera.shape[1] + GAP:] = table
    draw_scoreboard(picture, game, ROW_HEIGHT, width)
    if game.winner is not None:
        draw_celebration(picture, game)
    return picture


celebration = {'frame': 0, 'pieces': None}


def draw_celebration(picture, game):
    """
    End of game: a banner with the winner and confetti falling over the whole window.
    The confetti moves one step per frame drawn, so it animates live and in a recording alike.
    """
    height, width = picture.shape[:2]
    rng = np.random.default_rng(5)
    if celebration['pieces'] is None:
        count = 160
        celebration['pieces'] = {
            'x': rng.uniform(0, width, count), 'y': rng.uniform(-height, 0, count),
            'speed': rng.uniform(4, 10, count), 'drift': rng.uniform(-1.5, 1.5, count),
            'size': rng.integers(5, 11, count),
            'colour': rng.choice(np.array([(60, 60, 230), (0, 200, 255), (235, 235, 235), (110, 200, 120),
                                           (230, 160, 70)]), count)}
    pieces = celebration['pieces']
    step = celebration['frame']
    celebration['frame'] = step + 1

    # Banner across the middle
    overlay = picture.copy()
    top, bottom = height // 2 - 70, height // 2 + 70
    cv2.rectangle(overlay, (0, top), (width, bottom), (20, 16, 12), -1)
    cv2.addWeighted(overlay, 0.8, picture, 0.2, 0, dst=picture)
    group = game.groups[game.winner]
    accent = GROUP_COLOURS[group]
    cv2.line(picture, (0, top), (width, top), accent, 4)
    cv2.line(picture, (0, bottom), (width, bottom), accent, 4)
    words = f"PLAYER {game.winner + 1} WINS!"
    pulse = 1.0 + 0.06 * np.sin(step / 4.0)
    scale = 2.2 * pulse
    size = cv2.getTextSize(words, cv2.FONT_HERSHEY_DUPLEX, scale, 4)[0]
    origin = ((width - size[0]) // 2, height // 2 + size[1] // 2)
    cv2.putText(picture, words, origin, cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), 10, cv2.LINE_AA)
    cv2.putText(picture, words, origin, cv2.FONT_HERSHEY_DUPLEX, scale, accent, 4, cv2.LINE_AA)
    sub = f"{game.balls_potted(game.winner)} {group or ''} balls potted".replace('  ', ' ')
    sub_size = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.putText(picture, sub, ((width - sub_size[0]) // 2, bottom - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                TEXT, 2, cv2.LINE_AA)

    # Confetti: falls, sways, and wraps back to the top
    ys = (pieces['y'] + pieces['speed'] * step) % (height + 40) - 20
    xs = (pieces['x'] + pieces['drift'] * step + 12 * np.sin(step / 6.0 + pieces['x'])) % width
    for x, y, size_px, colour in zip(xs, ys, pieces['size'], pieces['colour']):
        cv2.rectangle(picture, (int(x), int(y)), (int(x) + int(size_px), int(y) + int(size_px) // 2),
                      tuple(int(c) for c in colour), -1)


# ---------------------------------------------------------------- input and output

def open_source(args):
    """
    Returns a function that gives the next frame (or None when finished), the video's
    frame rate, the rate to play at (None = as fast as possible), and the frame to find
    the table from (None = the first frame played).
    """
    if args.photos:
        names = sorted(n for n in os.listdir(args.photos)
                       if n.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')))
        paths = [os.path.join(args.photos, n) for n in names]

        def next_photo():
            if not paths:
                return None
            return cv2.imread(paths.pop(0))
        return next_photo, 30.0, None, None

    if args.video:
        capture = cv2.VideoCapture(args.video)
    else:
        # auto = OpenCV's default (Media Foundation on Windows, which Camo needs); dshow = DirectShow.
        backends = {'auto': cv2.CAP_ANY, 'dshow': cv2.CAP_DSHOW, 'msmf': cv2.CAP_MSMF}
        capture = cv2.VideoCapture(args.camera, backends[args.backend])
    if not capture.isOpened():
        raise SystemExit('Could not open the camera or video')

    # A live camera opens at 640x480 on Windows unless asked for more, so ask for the size the
    # settings in config.py were tuned at (1920x1080). If the camera gives less (Camo free = 720p),
    # frames are scaled up to that size so pixel thresholds such as TABLE_EDGE_TRIM still fit.
    upscale_to = None
    if not args.video:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        got_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        got_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f'Camera {args.camera}: {got_w}x{got_h}')
        same_shape = got_h > 0 and abs(got_w / got_h - args.width / args.height) < 0.01
        if same_shape and got_h < args.height:
            upscale_to = (args.width, args.height)
            print(f'  scaling frames up to {args.width}x{args.height}')
        elif not same_shape:
            print(f'  warning: not {args.width}x{args.height} in shape; frames used as they come')
        # A webcam's first frames are often black while it starts up (Camo for a few seconds),
        # and the table is found from the first frame, so wait until a real picture arrives.
        deadline = time.time() + 15
        while True:
            ok, frame = capture.read()
            if ok and frame.mean() > 10:          # not black
                time.sleep(0.5)                   # let exposure settle too
                for _ in range(5):
                    capture.read()
                break
            if time.time() > deadline:
                raise SystemExit(f'Camera {args.camera} is only giving black frames. Check Camo Studio shows '
                                 f'the phone picture, or try --backend dshow or another --camera number.')
            time.sleep(0.1)
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    if not args.video and not 1 <= fps <= 120:
        # Virtual webcams on Windows can report a nonsense frame rate, which video files refuse.
        print(f'  camera reported {fps} fps; using 30')
        fps = 30.0

    last_frame = None
    calibration_frame = None
    if args.video:
        # The camera is on a tripod, so the table is found from the very first frame of the
        # recording (which must show the clear table), even when playing from later on.
        ok, calibration_frame = capture.read()
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(args.start * fps))
        if args.end is not None:
            last_frame = int(args.end * fps)

    def next_frame():
        if last_frame is not None and capture.get(cv2.CAP_PROP_POS_FRAMES) > last_frame:
            return None
        ok, frame = capture.read()
        if not ok:
            return None
        if upscale_to is not None:
            frame = cv2.resize(frame, upscale_to, interpolation=cv2.INTER_LINEAR)
        return frame

    # A recorded video is played at its real speed (times --speed); a live camera already is.
    play_fps = fps * args.speed if args.video else None
    return next_frame, fps, play_fps, calibration_frame


def unused_path(path):
    """path, or path with _2, _3 ... added, so an earlier recording is never overwritten."""
    if not os.path.exists(path):
        return path
    stem, extension = os.path.splitext(path)
    number = 2
    while os.path.exists(f"{stem}_{number}{extension}"):
        number = number + 1
    return f"{stem}_{number}{extension}"


def open_recording(path, picture, fps):
    """
    A video writer the size of the window picture, and the path it actually writes to.
    H.264 ('avc1') plays everywhere (QuickTime, phones); 'mp4v' is the fallback if this OpenCV
    build lacks it. If neither works (seen on Windows), the video is saved as .avi (MJPG),
    which every OpenCV build can write.
    """
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    try:
        with open(path, 'wb'):
            pass                              # can this folder be written to at all?
        os.remove(path)
    except OSError as error:
        raise SystemExit(f'Cannot write to {path}: {error}')
    size = (picture.shape[1], picture.shape[0])
    for codec in ('avc1', 'mp4v'):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer, path
    avi_path = unused_path(os.path.splitext(path)[0] + '.avi')
    writer = cv2.VideoWriter(avi_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
    if writer.isOpened():
        print(f'  .mp4 not available here, saving {avi_path} instead')
        return writer, avi_path
    raise SystemExit(f'Could not open {path} for writing ({size[0]}x{size[1]} at {fps} fps)')


def add_foul(game, shooter):
    """
    Referee call: the last shot was a foul the camera missed. The player who took it gets a foul
    and the other player comes to the table (with two visits, as for any foul).
    """
    if game.winner is not None:
        return
    if shooter is None:
        shooter = game.current
    game.fouls[shooter] = game.fouls[shooter] + 1
    game.current = 1 - shooter
    game.visits_left = 2 if config.FOUL_GIVES_TWO_VISITS else 1
    game.log.append(f"Foul given to {game.players[shooter]} by hand")


def no_foul(game, history, result):
    """
    Referee overrule: the last shot was not a foul after all. The shot is played again from the
    game before it, keeping what was potted but without the contact fouls (wrong ball / no ball
    hit), as if the contact had not been seen.
    """
    if not history or result is None:
        return game
    replay = copy.deepcopy(history[-1])
    replay.apply_shot(result['potted'], result['note'])
    return replay


# ---------------------------------------------------------------- main loop

def main():
    parser = argparse.ArgumentParser(description='Live pool ball tracking window')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--backend', choices=['auto', 'dshow', 'msmf'],
                        default='auto',
                        help='live camera: OpenCV capture backend (auto = Media Foundation on Windows)')
    parser.add_argument('--width', type=int, default=1920, help='live camera: frame width to ask for')
    parser.add_argument('--height', type=int, default=1080, help='live camera: frame height to ask for')
    parser.add_argument('--video')
    parser.add_argument('--photos')
    parser.add_argument('--click', nargs=2, type=int, metavar=('X', 'Y'),
                        help='a pixel on bare cloth in the first frame')
    parser.add_argument('--start', type=float, default=0.0, help='video: start this many seconds in')
    parser.add_argument('--end', type=float, default=None, help='video: stop at this many seconds')
    parser.add_argument('--speed', type=float, default=1.0, help='video playback speed, e.g. 2 = twice real time')
    parser.add_argument('--table-edge-trim', type=int, default=None,
                        help='override config.TABLE_EDGE_TRIM (about 41 for a high, far camera angle)')
    parser.add_argument('--moving-camera', action='store_true',
                        help='the camera may be moved or bumped: find the table again before each board reading')
    parser.add_argument('--mid-game', action='store_true',
                        help='no longer needed: the balls left are always taken from the first board reading')
    parser.add_argument('--player1-colour', choices=['red', 'yellow'],
                        help="with --mid-game: Player 1's colour, if the table is no longer open")
    parser.add_argument('--to-play', type=int, choices=[1, 2], default=1,
                        help='with --mid-game: which player is at the table first')
    parser.add_argument('--correct', action='append', default=[], metavar='SECONDS:ACTION',
                        help="a referee correction at a time in the video, like pressing the key live: "
                             "ACTION is swap (turn), colours, undo, foul or nofoul, e.g. --correct 201:nofoul. Can repeat.")
    parser.add_argument('--reds', type=int, default=7, help='red balls in the set (default 7)')
    parser.add_argument('--yellows', type=int, default=7, help='yellow balls in the set (default 7)')
    parser.add_argument('--save-raw', metavar='PATH',
                        help='live camera: also save the plain camera feed (.mp4), so the game can be re-run '
                             'afterwards with --video to check or tune it')
    parser.add_argument('--record', metavar='PATH', help='also save the window as a video (.mp4)')
    parser.add_argument('--headless', action='store_true',
                        help='no window (use with --record): process the whole video and stop')
    args = parser.parse_args()
    # Never overwrite an earlier recording: a name already used gets _2, _3 ... added.
    if args.record:
        args.record = unused_path(args.record)
    if args.save_raw:
        args.save_raw = unused_path(args.save_raw)

    if args.table_edge_trim is not None:
        config.TABLE_EDGE_TRIM = args.table_edge_trim
    next_frame, fps, play_fps, calibration_frame = open_source(args)
    photo_mode = args.photos is not None

    frame = next_frame()
    if frame is None:
        raise SystemExit('No frames to read')
    start_frame = calibration_frame if calibration_frame is not None else frame
    try:
        watcher = BoardWatcher(start_frame, args.click, fps, args.moving_camera)
    except Exception as error:
        cv2.imwrite('first_frame.jpg', start_frame)
        raise SystemExit(f"Could not find the table in the first frame ({type(error).__name__}: {error}).\n"
                         f"That frame is saved as first_frame.jpg: check it shows the whole, clear table.\n"
                         f"If it does, try --click X Y with a pixel on bare cloth.")

    corrections = sorted((float(c.split(':')[0]), c.split(':')[1]) for c in args.correct)
    frame_number = int(args.start * fps) if args.video else 0
    game = GameState(totals={'red': args.reds, 'yellow': args.yellows})
    game.current = args.to_play - 1
    # The ball counts always come from the first board reading. If the game starts with balls
    # already gone (e.g. 3 yellows and 2 reds left), they are credited to whoever ends up on that
    # colour as soon as the colours are decided. (--mid-game is kept for older commands.)
    joined = False
    history = []                       # copies of the game before each shot, for undo
    last_result = None                 # the last shot, so a wrong foul call can be taken back
    last_shooter = None                # who took the last shot, for a foul added by hand
    message = ''
    paused = False
    first = True
    finished = False
    next_show_time = time.time()
    live_start = time.time()
    writer = None
    raw_writer = None

    if not args.headless:
        cv2.namedWindow('Pool tracking', cv2.WINDOW_NORMAL)

    while True:
        if (first or not paused) and not finished:
            if not first:
                new_frame = next_frame()
                if new_frame is None:
                    finished = True
                    paused = True
                    message = 'end of video'
                else:
                    frame = new_frame
                    frame_number = frame_number + 1
                    if args.save_raw:
                        if raw_writer is None:
                            raw_writer, args.save_raw = open_recording(args.save_raw, frame, fps)
                        raw_writer.write(frame)
            first = False

            # Scheduled referee corrections (--correct), applied when the video reaches them.
            while corrections and frame_number / fps >= corrections[0][0]:
                _, action = corrections.pop(0)
                history.append(copy.deepcopy(game))
                if action == 'swap':
                    game.current = 1 - game.current
                    game.visits_left = 1
                elif action == 'colours' and game.groups[0] is not None:
                    game.groups = [game.groups[1], game.groups[0]]
                elif action == 'undo':
                    history.pop()
                    if history:
                        game = history.pop()
                elif action == 'foul':
                    add_foul(game, last_shooter)
                elif action == 'nofoul':
                    history.pop()
                    game = no_foul(game, history, last_result)
                message = f"correction: {action}"

            if not finished:
                try:
                    if photo_mode:
                        # Photos have no motion to measure and may each be from a different angle:
                        # find the table every time and show what is on it.
                        watcher.calibrate(frame)
                        reading = board_reading.read_board(frame, watcher.reference,
                                                           watcher.click_x, watcher.click_y)
                        watcher.commit(reading, shot_happened=True)   # photos: shown, not scored
                        watcher.status = 'settled'
                        paused = True                 # wait for space before the next photo
                        result = None
                    else:
                        # Live camera: real clock time (frames are dropped if processing falls behind).
                        live_t = None if (args.video or photo_mode) else time.time() - live_start
                        result = watcher.update(frame, live_t)
                except Exception as error:            # e.g. the table cannot be found
                    message = f"error: {error}"[:60]
                    result = None
                # Starting part-way through: the first agreed board sets how many balls are left.
                if not joined and watcher.committed_balls is not None:
                    counts = board_reading.count_colours(watcher.committed_balls)
                    game.start_mid_game(counts['red'], counts['yellow'], args.player1_colour)
                    joined = True
                if result is not None and game.winner is None:
                    history.append(copy.deepcopy(game))
                    last_result = result
                    last_shooter = game.current
                    game.apply_shot(result['potted'], result['note'], result['first_hit'],
                                    result['tracked'], result['balls_moved'])

        picture = render(frame, watcher, game)
        if message and not args.headless:
            print(message)                    # key presses and errors go to the terminal, not the window
            message = ''
        if args.record and (not paused or writer is None):
            if writer is None:
                writer, args.record = open_recording(args.record, picture, fps)
            writer.write(picture)
        if args.headless:
            if finished:
                break
            continue
        cv2.imshow('Pool tracking', picture)

        # Hold each video frame until its time comes, so the video plays at real speed.
        wait_ms = 30 if paused else 1
        if play_fps and not paused:
            next_show_time = max(next_show_time + 1.0 / play_fps, time.time() - 0.5)
            wait_ms = max(1, int((next_show_time - time.time()) * 1000))
        key = cv2.waitKey(wait_ms) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord(' ') and not finished:
            paused = not paused
            message = 'paused' if paused else ''
        elif key == ord('k'):
            watcher.calibrate(frame, args.click)
            message = 'found table again'
        elif key == ord('u'):
            if history:
                game = history.pop()
                message = 'undid last shot'
        elif key == ord('s'):
            history.append(copy.deepcopy(game))
            game.current = 1 - game.current
            game.visits_left = 1
            message = 'swapped turn'
        elif key == ord('c'):
            if game.groups[0] is not None:
                history.append(copy.deepcopy(game))
                game.groups = [game.groups[1], game.groups[0]]
                message = 'swapped colours'
        elif key == ord('x'):
            history.append(copy.deepcopy(game))
            add_foul(game, last_shooter)
            message = 'foul added'
        elif key == ord('f'):
            game = no_foul(game, history, last_result)
            message = 'last foul cancelled'
        elif key == ord('n'):
            game = GameState(totals={'red': args.reds, 'yellow': args.yellows})
            history = []
            joined = False                    # take the ball counts from the next board reading
            message = 'new game'

    if raw_writer is not None:
        raw_writer.release()
        print(f"Camera feed saved to {args.save_raw}")
    if writer is not None:
        writer.release()
        print(f"Recording saved to {args.record}")
    if not args.headless:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()