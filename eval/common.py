"""
Shared helpers for the evaluation scripts in eval/.

- paths: eval/videos.csv (the test videos), eval/gt/ (hand-clicked ground truth),
  results/eval/ (every output)
- deterministic frame sampling and frame reading
- the Markdown summary table every script prints and saves

The pipeline modules are imported from the repo root, unchanged.
"""

import csv
import os
import sys

import cv2
import numpy as np

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(EVAL_DIR)
sys.path.insert(0, REPO_DIR)

import config

# Cushion-to-cushion playing area of the test table, in mm (standard 7 ft bar table: 198 x 99 cm,
# the same 2:1 as the map). Every mm number in the results scales with this.
PLAY_LENGTH_MM = 1980
PLAY_WIDTH_MM = 990
MM_PER_MAP_PX = PLAY_LENGTH_MM / config.MAP_WIDTH

# Horizontal field of view assumed for the phone's 1x camera in video mode (26 mm equivalent lens,
# a little cropped by stabilisation). Only used for the camera elevation estimate, which moves by
# about 1 degree over 60-76 degrees of field of view.
PHONE_HFOV_DEG = 65

VIDEOS_CSV = os.path.join(EVAL_DIR, 'videos.csv')
GT_DIR = os.path.join(EVAL_DIR, 'gt')
RESULTS_DIR = os.path.join(REPO_DIR, 'results', 'eval')

# view = end | side | corner, height_m = lens height above the cloth,
# distance_m = horizontal distance from the lens to the table centre,
# lighting = led | window | mixed | dim, start_s = first second to sample (blank = 1; later where the
# camera was still being set up)
VIDEO_COLUMNS = ['file', 'label', 'view', 'height_m', 'distance_m', 'lighting', 'session', 'start_s', 'notes']
VIDEO_FOLDERS = ['Videos/Game videos at normal diagonal view', 'Videos/Game videos at other view angles']
VIDEO_EXTENSIONS = ('.mp4', '.mov')

# The map rectangle the four corners are sent to (top_down.make_map_transform).
MAP_CORNERS = np.array([[0, 0], [config.MAP_WIDTH, 0], [config.MAP_WIDTH, config.MAP_HEIGHT],
                        [0, config.MAP_HEIGHT]], dtype=np.float32)


# ---------------------------------------------------------------- the video list

def write_videos_template(path=VIDEOS_CSV):
    """
    A videos.csv with the header and one row per video in the two game-video folders.
    Only file and label are filled in; the rest must be measured and typed in by hand.
    """
    rows = []
    for folder in VIDEO_FOLDERS:
        folder_path = os.path.join(REPO_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        for name in sorted(os.listdir(folder_path)):
            if name.lower().endswith(VIDEO_EXTENSIONS):
                rows.append({'file': f"{folder}/{name}", 'label': video_stem(name)})
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=VIDEO_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return len(rows)


def read_videos(path=VIDEOS_CSV):
    """The rows of videos.csv that name a file, in file order. Blank label = the file's stem."""
    with open(path, newline='') as handle:
        rows = [row for row in csv.DictReader(handle) if (row.get('file') or '').strip()]
    for row in rows:
        row['file'] = row['file'].strip()
        if not (row.get('label') or '').strip():
            row['label'] = video_stem(row['file'])
    return rows


def video_path(file):
    """A videos.csv file entry as a full path (entries are relative to the repo root)."""
    return file if os.path.isabs(file) else os.path.join(REPO_DIR, file)


def video_stem(file):
    return os.path.splitext(os.path.basename(file))[0]


def gt_path(file):
    return os.path.join(GT_DIR, f"{video_stem(file)}_corners.json")


def number_or_none(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def elevation_deg(row):
    """Camera elevation above the cloth, degrees(atan(height / distance)), or None if not filled in."""
    height = number_or_none(row.get('height_m'))
    distance = number_or_none(row.get('distance_m'))
    if height is None or distance is None or distance <= 0:
        return None
    return float(np.degrees(np.arctan(height / distance)))


def camera_from_corners(corners, width, height):
    """
    Where the camera was, worked out from the four ground-truth playing-area corners (pipeline order,
    so corner 0 -> 1 is a long side) and the known table size, for a camera with PHONE_HFOV_DEG.
    Returns height above the cloth and horizontal distance to the table centre (m), the elevation
    angle (deg) and the reprojection error (px, how well that camera explains the clicks).
    """
    focal = width / 2 / np.tan(np.radians(PHONE_HFOV_DEG / 2))
    camera = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    table = np.array([[0, 0, 0], [PLAY_LENGTH_MM, 0, 0], [PLAY_LENGTH_MM, PLAY_WIDTH_MM, 0],
                      [0, PLAY_WIDTH_MM, 0]], dtype=np.float32)
    corners = np.asarray(corners, dtype=np.float32)
    ok, rotation, translation = cv2.solvePnP(table, corners, camera, None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    projected, _ = cv2.projectPoints(table, rotation, translation, camera, None)
    matrix, _ = cv2.Rodrigues(rotation)
    position = (-matrix.T @ translation).ravel() / 1000                        # camera centre, m
    above = abs(position[2])
    across = np.hypot(position[0] - PLAY_LENGTH_MM / 2000, position[1] - PLAY_WIDTH_MM / 2000)
    return {'height_m': float(above), 'distance_m': float(across),
            'elevation_deg': float(np.degrees(np.arctan2(above, across))),
            'reprojection_px': float(np.abs(projected.reshape(4, 2) - corners).max())}


# ---------------------------------------------------------------- frames

def open_video(file):
    """(capture, fps, frame_count). Raises if the video cannot be opened."""
    capture = cv2.VideoCapture(video_path(file))
    if not capture.isOpened():
        raise RuntimeError('could not open video')
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    return capture, fps, int(capture.get(cv2.CAP_PROP_FRAME_COUNT))


def sample_indices(frame_count, fps, count=20, start_s=1.0):
    """count frame numbers evenly spaced from start_s seconds to the last frame (no randomness)."""
    first = int(round(start_s * fps))
    last = frame_count - 1
    if last < first:
        return []
    return sorted(set(int(round(i)) for i in np.linspace(first, last, count)))


def read_frame(capture, index):
    """Frame number index, or None if it cannot be read."""
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    return frame if ok else None


# ---------------------------------------------------------------- output

def fmt(value, digits=1):
    """A number for a table: blank-safe ('n/a' for missing)."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return 'n/a'
    return f"{value:.{digits}f}"


def write_csv(path, rows, columns):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ('' if row.get(key) is None else row.get(key)) for key in columns})


def write_summary(script_name, header, rows, notes=()):
    """Prints the Markdown table (plus notes) and saves it as results/eval/<script>_summary.md."""
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    lines += ['| ' + ' | '.join(str(cell) for cell in row) + ' |' for row in rows]
    if notes:
        lines.append('')
        lines += [f"- {note}" for note in notes]
    text = '\n'.join(lines) + '\n'
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{script_name}_summary.md")
    with open(path, 'w') as handle:
        handle.write(text)
    print()
    print(text)
    print(f"Saved {path}")
    return path
