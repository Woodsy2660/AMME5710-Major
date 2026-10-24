"""
Step 8: the live tracking window. The camera on the left, a top-down table on the right,
with every ball drawn where it is now and a fading trail behind each rolling ball.

- board_watcher.py follows the game: between shots the balls come from the last agreed board
  reading; during a shot they come from ball_tracking.py, frame by frame.
- The camera should be on a tripod, with the first frame showing the whole table.
  Press k to find the table again if the camera is bumped (or use --moving-camera).
- Scoring (game_state.py) is not shown here for now: this window is only about the tracking.

Usage
    python gui.py                          webcam 0 (an iPhone via Continuity Camera)
    python gui.py --camera 1               another webcam
    python gui.py --video game.mp4         a recorded game
    python gui.py --photos "Table images"  step through still photos (space = next)
    python gui.py --video game.mp4 --speed 2               play at twice real time
    python gui.py --video game.mp4 --start 120 --end 200   only part of a recording (seconds)
    python gui.py --video game.mp4 --record demo.mp4 --headless   save the window as a video

Keys
    space   pause (photos: next photo)     k   find the table again     q / esc   quit
"""

import argparse
import os
import time

import cv2
import numpy as np

import board_reading
import config
import draw
import top_down
from board_watcher import BoardWatcher

# Layout, in pixels: video and table side by side at the same height, a status line below.
ROW_HEIGHT = 600
STATUS_HEIGHT = 44
GAP = 12

BACKGROUND = (28, 24, 20)            # BGR
TEXT = (235, 235, 235)
FADED = (150, 150, 150)
STATUS_COLOURS = {'SHOT': (230, 160, 70), 'settled': (110, 200, 120), 'reading': (0, 200, 255),
                  'settling': FADED, 'view blocked': (70, 70, 230), 'camera moved': (70, 70, 230),
                  'player over table': (70, 70, 230)}

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


# ---------------------------------------------------------------- the whole window

def render(frame, watcher, message=''):
    balls = watcher.live_balls()
    camera = resize_to_height(draw_camera(frame, balls, watcher.reference), ROW_HEIGHT)
    table = resize_to_height(draw_table(balls), ROW_HEIGHT)

    width = camera.shape[1] + GAP + table.shape[1]
    picture = np.full((ROW_HEIGHT + STATUS_HEIGHT, width, 3), BACKGROUND, dtype=np.uint8)
    picture[:ROW_HEIGHT, :camera.shape[1]] = camera
    picture[:ROW_HEIGHT, camera.shape[1] + GAP:] = table

    status = watcher.status
    colour = STATUS_COLOURS.get(status, TEXT)
    if watcher.tracks is not None and status == 'SHOT':
        status = f"SHOT - tracking {len(balls)} balls"
    y = ROW_HEIGHT + 30
    cv2.putText(picture, status.upper(), (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2, cv2.LINE_AA)
    if message:
        cv2.putText(picture, message, (420, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, FADED, 1, cv2.LINE_AA)
    keys = 'space = pause    k = find table    q = quit'
    size = cv2.getTextSize(keys, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.putText(picture, keys, (width - size[0] - 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, FADED, 1, cv2.LINE_AA)
    return picture


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

    capture = cv2.VideoCapture(args.video if args.video else args.camera)
    if not capture.isOpened():
        raise SystemExit('Could not open the camera or video')
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

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
        return frame if ok else None

    # A recorded video is played at its real speed (times --speed); a live camera already is.
    play_fps = fps * args.speed if args.video else None
    return next_frame, fps, play_fps, calibration_frame


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


# ---------------------------------------------------------------- main loop

def main():
    parser = argparse.ArgumentParser(description='Live pool ball tracking window')
    parser.add_argument('--camera', type=int, default=0)
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
    parser.add_argument('--record', metavar='PATH', help='also save the window as a video (.mp4)')
    parser.add_argument('--headless', action='store_true',
                        help='no window (use with --record): process the whole video and stop')
    args = parser.parse_args()

    if args.table_edge_trim is not None:
        config.TABLE_EDGE_TRIM = args.table_edge_trim
    next_frame, fps, play_fps, calibration_frame = open_source(args)
    photo_mode = args.photos is not None

    frame = next_frame()
    if frame is None:
        raise SystemExit('No frames to read')
    watcher = BoardWatcher(calibration_frame if calibration_frame is not None else frame, args.click, fps,
                           args.moving_camera)

    message = ''
    paused = False
    first = True
    finished = False
    next_show_time = time.time()
    writer = None

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
            first = False

            if not finished:
                try:
                    if photo_mode:
                        # Photos have no motion to measure and may each be from a different angle:
                        # find the table every time and show what is on it.
                        watcher.calibrate(frame)
                        reading = board_reading.read_board(frame, watcher.reference,
                                                           watcher.click_x, watcher.click_y)
                        watcher.commit(reading, shot_happened=True)
                        watcher.status = 'settled'
                        paused = True                 # wait for space before the next photo
                    else:
                        watcher.update(frame)
                except Exception as error:            # e.g. the table cannot be found
                    message = f"error: {error}"[:60]

        picture = render(frame, watcher, message)
        if args.record and (not paused or writer is None):
            if writer is None:
                writer = open_recording(args.record, picture, fps)
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

    if writer is not None:
        writer.release()
        print(f"Recording saved to {args.record}")
    if not args.headless:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
