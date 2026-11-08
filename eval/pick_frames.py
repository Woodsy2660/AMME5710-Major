"""
Settled-board frames to hand-label for the ball detection test.

How to run:
    python eval/pick_frames.py "Videos/Game videos at normal diagonal view/test_3.mp4" --times 12.5,40,61
    python eval/pick_frames.py "Videos/Game videos at normal diagonal view/test_3.mp4" --auto

- --times: the frames at these times (seconds), as they are
- --auto: plays the whole video through the shot detector exactly as BoardWatcher.update does
  (shrink_to_gray, moving_pixels, update_detector, with the table found from the first frame as
  BoardWatcher.calibrate does it) and takes the frame 1 s after each SHOT -> SETTLED change.
  Frames where obstruction.view_is_blocked is true are skipped, then at most MAX_AUTO frames are
  kept, evenly spread over the game (so the table is seen full, half-cleared and nearly empty).
- If the video has a start_s in eval/videos.csv (camera still being set up before it), the table is
  found from that frame and nothing earlier is used.

Frames are saved full size to eval/frames/<stem>_<mmss>.jpg (JPEG quality 100, no colour
subsampling) and listed in eval/frames/frames.csv, which ball_eval.py uses to find the video.
"""

import argparse
import csv
import os

import cv2
import numpy as np

import common
import board_reading
import config
import obstruction
import table_mask
from shot_detection import shrink_to_gray, moving_pixels, update_detector

FRAMES_DIR = os.path.join(common.EVAL_DIR, 'frames')
FRAMES_CSV = os.path.join(FRAMES_DIR, 'frames.csv')
FRAME_COLUMNS = ['image', 'video', 'frame', 'time_s', 'source', 'blocked', 'settled_frame']
MAX_AUTO = 8


def start_seconds(file):
    """start_s from videos.csv for this video, or 0 (the very first frame, as the GUI uses)."""
    if os.path.exists(common.VIDEOS_CSV):
        for row in common.read_videos():
            if os.path.normpath(row['file']) == os.path.normpath(file):
                return common.number_or_none(row.get('start_s')) or 0.0
    return 0.0


def calibrate(frame):
    """The table, found as BoardWatcher.calibrate does it."""
    click_x, click_y = table_mask.find_cloth_center(frame)
    return board_reading.make_reference(frame, click_x, click_y)


def is_blocked(frame, reference):
    _, lumps = obstruction.unexplained_lumps(frame, reference)
    return obstruction.view_is_blocked(lumps)


def settled_frames(capture, first_index, reference):
    """Frame numbers where the shot detector goes from SHOT to SETTLED, from first_index to the end."""
    capture.set(cv2.CAP_PROP_POS_FRAMES, first_index)
    detector = {'state': 'SETTLED', 'still_count': 0, 'blocked': False, 'readable': False}
    previous_gray = None
    changes = []
    index = first_index
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        gray = shrink_to_gray(frame)
        moving = moving_pixels(gray, previous_gray, reference)
        previous_gray = gray
        was_shot = detector['state'] == 'SHOT'
        update_detector(detector, moving, False, reference)
        if was_shot and detector['state'] == 'SETTLED':
            changes.append(index)
        index = index + 1
    return changes


def spread_evenly(items, count):
    """At most count items, evenly spaced through the list (first and last kept)."""
    if len(items) <= count:
        return items
    return [items[int(round(i))] for i in np.linspace(0, len(items) - 1, count)]


def image_name(file, seconds, taken):
    """<stem>_<mmss>.jpg, with _2, _3 ... if two frames fall in the same second."""
    stem = f"{common.video_stem(file)}_{int(seconds // 60):02d}{int(seconds % 60):02d}"
    name = f"{stem}.jpg"
    number = 2
    while name in taken:
        name = f"{stem}_{number}.jpg"
        number = number + 1
    return name


def save_frame(path, frame):
    params = [cv2.IMWRITE_JPEG_QUALITY, 100]
    if hasattr(cv2, 'IMWRITE_JPEG_SAMPLING_FACTOR'):
        params += [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
    cv2.imwrite(path, frame, params)


def read_frames_csv():
    if not os.path.exists(FRAMES_CSV):
        return {}
    with open(FRAMES_CSV, newline='') as handle:
        return {row['image']: row for row in csv.DictReader(handle)}


def main():
    parser = argparse.ArgumentParser(description='Export settled-board frames for hand labelling')
    parser.add_argument('video', help='video file, relative to the repo root or absolute')
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--times', help='comma-separated times in seconds, e.g. 12.5,40,61')
    choice.add_argument('--auto', action='store_true', help='1 s after each SHOT -> SETTLED, unblocked, at most 8')
    parser.add_argument('--table-edge-trim', type=int, default=None,
                        help="as gui.py: override config.TABLE_EDGE_TRIM at run time (41 for the IMG_0711 camera "
                             "position, where 15 leaves a strip of cushion that reads as a cue and blocks every frame)")
    args = parser.parse_args()
    if args.table_edge_trim is not None:
        config.TABLE_EDGE_TRIM = args.table_edge_trim

    capture, fps, frame_count = common.open_video(args.video)
    start_index = int(round(start_seconds(args.video) * fps))
    reference = calibrate(common.read_frame(capture, start_index))
    one_second = int(round(fps))

    rows = []                       # one per candidate frame: exported or skipped
    if args.times:
        wanted = [(int(round(float(t) * fps)), '') for t in args.times.split(',') if t.strip()]
        source = 'times'
    else:
        print(f"Running the shot detector over {frame_count} frames ...")
        changes = settled_frames(capture, start_index, reference)
        wanted = [(change + one_second, change) for change in changes if change + one_second < frame_count]
        source = 'auto'
        print(f"{len(changes)} SHOT -> SETTLED changes")

    candidates = []
    for index, settled in wanted:
        frame = common.read_frame(capture, index)
        if frame is None:
            rows.append({'frame': index, 'time_s': round(index / fps, 2), 'status': 'failed',
                         'reason': 'could not read frame'})
            continue
        try:
            blocked = is_blocked(frame, reference)
        except Exception as error:
            blocked, reason = None, f"blocked test failed: {error}"[:80]
        if args.auto and blocked is not False:
            rows.append({'frame': index, 'time_s': round(index / fps, 2), 'status': 'skipped',
                         'reason': 'view blocked' if blocked else reason})
            continue
        candidates.append((index, settled, blocked))
    chosen = spread_evenly(candidates, MAX_AUTO) if args.auto else candidates

    os.makedirs(FRAMES_DIR, exist_ok=True)
    listed = read_frames_csv()
    taken = {name for name, row in listed.items() if row['video'] != args.video}
    for row in [r for r in listed.values() if r['video'] == args.video and r['source'] == source]:
        listed.pop(row['image'])    # a new run replaces this video's earlier list from the same source
    for index, settled, blocked in candidates:
        if (index, settled, blocked) not in chosen:
            rows.append({'frame': index, 'time_s': round(index / fps, 2), 'status': 'not chosen',
                         'reason': f"more than {MAX_AUTO} unblocked"})
            continue
        name = image_name(args.video, index / fps, taken)
        taken.add(name)
        save_frame(os.path.join(FRAMES_DIR, name), common.read_frame(capture, index))
        listed[name] = {'image': name, 'video': args.video, 'frame': index, 'time_s': round(index / fps, 2),
                        'source': source, 'blocked': blocked, 'settled_frame': settled}
        rows.append({'frame': index, 'time_s': round(index / fps, 2), 'status': 'exported', 'reason': name})
    capture.release()
    common.write_csv(FRAMES_CSV, sorted(listed.values(), key=lambda r: r['image']), FRAME_COLUMNS)

    print('Exported:')
    for row in rows:
        if row['status'] == 'exported':
            print(f"  eval/frames/{row['reason']}")
    rows.sort(key=lambda r: r['frame'])
    common.write_summary(f"pick_frames_{common.video_stem(args.video)}", ['frame', 'time (s)', 'status', 'file / reason'],
                         [[r['frame'], r['time_s'], r['status'], r['reason']] for r in rows],
                         notes=[f"Video: {args.video}, {fps:.2f} fps, table found from frame {start_index}, "
                                f"TABLE_EDGE_TRIM {config.TABLE_EDGE_TRIM}"
                                + (' (set at run time with --table-edge-trim, as gui.py)' if args.table_edge_trim
                                   is not None else ' (config default)') + '.',
                                f"Source: {source}" + (f", frame {one_second} frames (1 s) after each SHOT -> SETTLED "
                                                       f"change, view_is_blocked frames skipped, at most {MAX_AUTO} "
                                                       f"kept, evenly spread." if args.auto else '.')])


if __name__ == '__main__':
    main()
