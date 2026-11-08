"""
How reliably and accurately does table calibration find the table, for different camera
positions and lighting?

How to run:  python eval/calibration_eval.py [--csv eval/videos.csv]

For every video in eval/videos.csv, 20 frames evenly spaced from 1 s to the end go through the
same calibration as the GUI (BoardWatcher.calibrate): table_mask.find_cloth_center, then
board_reading.make_reference (cloth colour, cloth mask, outline, corners, long side first,
homography). Nothing in the pipeline is changed or re-tuned.

Per frame: found, corners, corner error against the hand-clicked ground truth (eval/gt/, from
click_corners.py) in px and in mm on the table, quad IoU, the cloth colour the pipeline sampled, and
the brightness of the cloth (lighting is measured from the frames, not typed in).
Per video: found rate, corner error, IoU, corner jitter across frames, median cloth colour, lighting
(cloth brightness, how uneven it is across the table, how much it changes over the video), and the
camera elevation, both from videos.csv (tripod measurements, where known) and estimated from the
ground-truth corners (common.camera_from_corners).

Outputs in results/eval/: calibration_frames.csv, calibration.csv, calibration_overlay_<stem>.jpg,
calibration_montage.jpg and calibration_summary.md.
"""

import argparse
import json
import os

import cv2
import numpy as np

import common
import board_reading
import table_mask
import top_down
from common import fmt

FRAMES_PER_VIDEO = 20
GROSS_FRACTION = 0.10            # mean corner error over this fraction of the play length = a gross failure
MAX_POSE_REPROJECTION_PX = 50    # camera elevation is only reported where the camera model fits the clicks
OVERLAY_WIDTH = 1280
TILE_WIDTH = 640

FRAME_COLUMNS = (['label', 'file', 'frame', 'time_s', 'status', 'reason', 'found', 'flipped', 'start_shift']
                 + [f"c{i}_{axis}" for i in range(4) for axis in 'xy']
                 + [f"err{i}_px" for i in range(4)]
                 + ['err_mean_px', 'err_max_px', 'err_mean_mm', 'err_max_mm', 'iou',
                    'cloth_h', 'cloth_s', 'cloth_v', 'light_v_median', 'light_v_p5', 'light_v_p95',
                    'click_x', 'click_y', 'elevation_tripod_deg', 'elevation_image_deg'])
VIDEO_COLUMNS = ['label', 'file', 'view', 'elevation_tripod_deg', 'elevation_image_deg', 'camera_height_m_image',
                 'camera_distance_m_image', 'pose_reprojection_px', 'height_m', 'distance_m', 'lighting', 'session',
                 'start_s', 'status', 'reason', 'has_gt', 'frames_sampled', 'frames_read', 'frames_found', 'found_pct',
                 'gross_pct', 'flipped_frames', 'err_mean_px', 'err_max_px', 'err_median_px', 'err_mean_mm',
                 'err_max_mm', 'err_median_mm', 'iou_mean', 'jitter_px', 'jitter_clean_px', 'cloth_h', 'cloth_s', 'cloth_v', 'light_v', 'light_spread', 'light_drift', 'overlay']


# ---------------------------------------------------------------- one frame

def calibrate(frame):
    """The table calibration exactly as BoardWatcher.calibrate does it."""
    click_x, click_y = table_mask.find_cloth_center(frame)
    reference = board_reading.make_reference(frame, click_x, click_y)
    return reference, click_x, click_y


def load_gt(file, frame_shape, first_index):
    """(corners 4 x 2 in pipeline order, None) or (None, reason) if there is no usable ground truth."""
    path = common.gt_path(file)
    if not os.path.exists(path):
        return None, 'no ground truth (run eval/click_corners.py)'
    with open(path) as handle:
        record = json.load(handle)
    if (record['height'], record['width']) != tuple(frame_shape[:2]):
        return None, f"ground truth is {record['width']}x{record['height']}, video frames are not"
    if record['frame'] < first_index:
        # The camera was still being set up before start_s, so corners clicked then do not hold later.
        return None, f"ground truth clicked on frame {record['frame']}, before start_s: click again after it"
    return np.array(record['corners'], dtype=np.float32), None


def match_to_gt(corners, gt):
    """
    Lines the detected corners up with the ground-truth ones.
    Both are clockwise with corner 0 -> 1 on a long side, but the convention leaves open which long
    side is first, so shifts of 0 and 2 both follow it. If a shift of 1 or 3 fits better, the
    pipeline put a SHORT side first (the map would be the wrong way round): flipped = True and the
    nearest matching is used.
    Returns (corners in ground-truth order, shift, flipped).
    """
    def mean_error(shift):
        return np.hypot(*(np.roll(corners, -shift, axis=0) - gt).T).mean()

    by_convention = min((0, 2), key=mean_error)
    nearest = min(range(4), key=mean_error)
    flipped = nearest in (1, 3) and mean_error(nearest) < mean_error(by_convention)
    shift = nearest if flipped else by_convention
    return np.roll(corners, -shift, axis=0), shift, flipped


def quad_iou(quad_a, quad_b, frame_shape):
    """
    Intersection over union of two quads, from cv2.fillPoly masks on a canvas round both
    (clipped to one frame size beyond the picture, so a wild detection cannot overflow).
    """
    height, width = frame_shape[:2]
    both = np.vstack([quad_a, quad_b])
    low = np.maximum(np.floor(both.min(axis=0)), [-width, -height])
    high = np.minimum(np.ceil(both.max(axis=0)), [2 * width, 2 * height])
    size = (high - low + 1).astype(int)
    masks = []
    for quad in (quad_a, quad_b):
        mask = np.zeros((size[1], size[0]), dtype=np.uint8)
        points = np.clip(quad, low, high) - low
        cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
        masks.append(mask.astype(bool))
    union = np.count_nonzero(masks[0] | masks[1])
    return np.count_nonzero(masks[0] & masks[1]) / union if union else 0.0


def corner_errors_mm(aligned, gt):
    """Each detected corner's distance from its ground-truth corner on the table, in mm."""
    to_map_gt, _ = top_down.make_map_transform(gt)
    moved = np.array([top_down.move_point(to_map_gt, x, y) for x, y in aligned])
    return np.hypot(*(moved - common.MAP_CORNERS).T) * common.MM_PER_MAP_PX


def cloth_brightness(frame, reference, click_x, click_y, gt):
    """
    Lighting on the table: the HSV value (brightness, 0-255) of the pixels the pipeline accepts as
    cloth (table_mask.find_cloth_mask, as read_board calls it), inside the ground-truth quad (or
    the detected hull without ground truth). Returns (median, 5th percentile, 95th percentile).
    """
    image_hsv, cloth_mask = table_mask.find_cloth_mask(frame, click_x, click_y, reference['cloth_colour'])
    if gt is None:
        region = reference['hull_mask']
    else:
        region = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(region, [np.round(gt).astype(np.int32)], 255)
    values = image_hsv[:, :, 2][(region > 0) & (cloth_mask > 0)]
    if values.size == 0:
        return None, None, None
    return tuple(float(v) for v in np.percentile(values, [50, 5, 95]))


def evaluate_frame(frame, gt):
    """Everything measured on one frame (a dict for calibration_frames.csv)."""
    reference, click_x, click_y = calibrate(frame)
    corners = np.asarray(reference['corners'], dtype=np.float32)
    row = {'click_x': click_x, 'click_y': click_y}
    row['cloth_h'], row['cloth_s'], row['cloth_v'] = reference['cloth_colour']
    row['light_v_median'], row['light_v_p5'], row['light_v_p95'] = cloth_brightness(frame, reference,
                                                                                     click_x, click_y, gt)
    row['found'] = corners.shape == (4, 2) and bool(np.isfinite(corners).all())
    if not row['found']:
        return row, None, None
    for i, (x, y) in enumerate(corners):
        row[f"c{i}_x"], row[f"c{i}_y"] = round(float(x), 1), round(float(y), 1)
    if gt is None:
        return row, corners, None

    aligned, shift, flipped = match_to_gt(corners, gt)
    errors_px = np.hypot(*(aligned - gt).T)
    errors_mm = corner_errors_mm(aligned, gt)
    row.update({'start_shift': shift, 'flipped': flipped,
                'err_mean_px': float(errors_px.mean()), 'err_max_px': float(errors_px.max()),
                'err_mean_mm': float(errors_mm.mean()), 'err_max_mm': float(errors_mm.max()),
                'iou': quad_iou(corners, gt, frame.shape)})
    for i in range(4):
        row[f"err{i}_px"] = float(errors_px[i])
    return row, corners, aligned


# ---------------------------------------------------------------- one video

def align_to_first(all_corners):
    """Without ground truth: each frame's corners turned to best match the first found frame."""
    first = all_corners[0]
    return [min((np.roll(c, -s, axis=0) for s in range(4)), key=lambda r: np.abs(r - first).sum())
            for c in all_corners]


def corner_jitter(aligned_corners):
    """
    Mean over the four corners of how much that corner moves between frames: the RMS distance
    of its position from its mean position, sqrt(std_x^2 + std_y^2), in px.
    """
    if len(aligned_corners) < 2:
        return None
    stack = np.array(aligned_corners)                       # frames x 4 x 2
    return float(np.sqrt(stack.var(axis=0).sum(axis=1)).mean())


def evaluate_video(row, frame_rows):
    """One row of calibration.csv. Appends one row per sampled frame to frame_rows."""
    file = row['file']
    elevation = common.elevation_deg(row)
    video = {key: row.get(key, '') for key in ('label', 'file', 'view', 'height_m', 'distance_m',
                                               'lighting', 'session')}
    video.update({'elevation_tripod_deg': elevation, 'status': 'ok', 'reason': '', 'has_gt': False,
                  'frames_sampled': 0, 'frames_read': 0, 'frames_found': 0, 'flipped_frames': 0})
    capture, fps, frame_count = common.open_video(file)
    start_s = common.number_or_none(row.get('start_s')) or 1.0       # later where the camera was still being set up
    video['start_s'] = start_s
    indices = common.sample_indices(frame_count, fps, FRAMES_PER_VIDEO, start_s)
    video['frames_sampled'] = len(indices)
    if not indices:
        raise RuntimeError(f"video is shorter than start_s = {start_s} s")

    gt = None
    found = []                        # (frame index, pipeline corners, ground-truth-aligned corners)
    rows = []
    for index in indices:
        frame_row = {'label': row['label'], 'file': file, 'frame': index, 'time_s': round(index / fps, 2),
                     'status': 'ok', 'reason': '', 'found': False, 'elevation_tripod_deg': elevation}
        frame = common.read_frame(capture, index)
        if frame is None:
            frame_row.update({'status': 'failed', 'reason': 'could not read frame'})
            rows.append(frame_row)
            continue
        video['frames_read'] += 1
        if gt is None and not video['reason']:
            gt, gt_problem = load_gt(file, frame.shape, indices[0])
            video['has_gt'] = gt is not None
            video['reason'] = gt_problem or ''
            camera = common.camera_from_corners(gt, frame.shape[1], frame.shape[0]) if gt is not None else None
            if camera:
                video['pose_reprojection_px'] = camera['reprojection_px']
            if camera and camera['reprojection_px'] <= MAX_POSE_REPROJECTION_PX:
                video.update({'elevation_image_deg': camera['elevation_deg'],
                              'camera_height_m_image': camera['height_m'],
                              'camera_distance_m_image': camera['distance_m']})
        try:
            measured, corners, aligned = evaluate_frame(frame, gt)
            frame_row.update(measured)
            if corners is None:
                frame_row.update({'status': 'failed', 'reason': 'corners not finite'})
            else:
                found.append((index, corners, aligned))
        except Exception as error:                            # the pipeline could not find the table
            frame_row.update({'status': 'failed', 'reason': f"{type(error).__name__}: {error}"[:120]})
        rows.append(frame_row)
    for frame_row in rows:
        frame_row['elevation_image_deg'] = video.get('elevation_image_deg')
    frame_rows.extend(rows)

    video.update(video_numbers(rows, found, video['frames_read']))
    video['overlay'] = save_overlay(capture, video, indices, found, gt)
    capture.release()
    return video


def video_numbers(rows, found, frames_read):
    """The per-video numbers from the frame rows."""
    ok = [r for r in rows if r['found']]
    numbers = {'frames_found': len(found),
               'found_pct': 100.0 * len(found) / frames_read if frames_read else None}
    for key in ('cloth_h', 'cloth_s', 'cloth_v'):
        values = [r[key] for r in rows if r.get(key) is not None]
        numbers[key] = float(np.median(values)) if values else None
    # Lighting: how bright the cloth is, how uneven across the table (p95 - p5 within a frame),
    # and how much it changes over the video (std of the per-frame medians).
    lit = [r for r in rows if r.get('light_v_median') is not None]
    if lit:
        medians = [r['light_v_median'] for r in lit]
        numbers.update({'light_v': float(np.median(medians)),
                        'light_spread': float(np.median([r['light_v_p95'] - r['light_v_p5'] for r in lit])),
                        'light_drift': float(np.std(medians))})
    with_gt = [r for r in ok if r.get('err_mean_px') is not None]
    if with_gt:
        # A gross failure is a quad that is not the table at all (a player over it, the wrong
        # patch of cloth): mean corner error over GROSS_FRACTION of the table length. The mean
        # errors and the jitter include them; the medians, the flips and the clean jitter do not.
        gross_mm = GROSS_FRACTION * common.PLAY_LENGTH_MM
        clean = [r for r in with_gt if r['err_mean_mm'] <= gross_mm]
        clean_frames = {r['frame'] for r in clean}
        numbers.update({'err_mean_px': float(np.mean([r['err_mean_px'] for r in with_gt])),
                        'err_max_px': float(np.max([r['err_max_px'] for r in with_gt])),
                        'err_median_px': float(np.median([r['err_mean_px'] for r in with_gt])),
                        'err_mean_mm': float(np.mean([r['err_mean_mm'] for r in with_gt])),
                        'err_max_mm': float(np.max([r['err_max_mm'] for r in with_gt])),
                        'err_median_mm': float(np.median([r['err_mean_mm'] for r in with_gt])),
                        'iou_mean': float(np.mean([r['iou'] for r in with_gt])),
                        'gross_pct': 100.0 * (len(with_gt) - len(clean)) / len(with_gt),
                        'flipped_frames': sum(bool(r['flipped']) for r in clean)})
        numbers['jitter_px'] = corner_jitter([aligned for _, _, aligned in found])
        numbers['jitter_clean_px'] = corner_jitter([aligned for index, _, aligned in found if index in clean_frames])
    elif found:
        numbers['jitter_px'] = corner_jitter(align_to_first([corners for _, corners, _ in found]))
    return numbers


# ---------------------------------------------------------------- pictures

def dashed_polygon(picture, quad, colour, thickness, dash):
    """A closed polygon drawn as dashes dash px long (OpenCV has no dashed line)."""
    for i in range(4):
        start, end = quad[i], quad[(i + 1) % 4]
        length = np.hypot(*(end - start))
        for d in np.arange(0, length, 2 * dash):
            a = start + (end - start) * d / length
            b = start + (end - start) * min(d + dash, length) / length
            cv2.line(picture, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)),
                     colour, thickness, cv2.LINE_AA)


def number_corners(picture, quad, colour, size, offset):
    for i, (x, y) in enumerate(quad):
        origin = (int(x + offset[0]), int(y + offset[1]))
        cv2.putText(picture, str(i), origin, cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(picture, str(i), origin, cv2.FONT_HERSHEY_SIMPLEX, size, colour, 2, cv2.LINE_AA)


def save_overlay(capture, video, indices, found, gt):
    """
    The sampled frame nearest the middle of the video (a found one if any): detected quad in green,
    ground truth in red dashes, corners numbered in pipeline order, 1280 px wide. Returns its path.
    """
    middle = indices[len(indices) // 2]
    choices = [(index, corners) for index, corners, _ in found] or [(index, None) for index in indices]
    index, corners = min(choices, key=lambda item: abs(item[0] - middle))
    frame = common.read_frame(capture, index)
    if frame is None:
        return ''
    scale = OVERLAY_WIDTH / frame.shape[1]
    picture = cv2.resize(frame, (OVERLAY_WIDTH, int(round(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    if gt is not None:
        dashed_polygon(picture, gt * scale, (0, 0, 255), 2, 12)
        number_corners(picture, gt * scale, (0, 0, 255), 0.8, (8, 26))
    if corners is not None:
        cv2.polylines(picture, [np.round(corners * scale).astype(np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)
        number_corners(picture, corners * scale, (0, 255, 0), 0.8, (8, -8))
    words = [f"{video['label']}  frame {index}",
             f"found {fmt(video.get('found_pct'), 0)}%  err {fmt(video.get('err_mean_px'))} px  "
             f"IoU {fmt(video.get('iou_mean'), 3)}   green = detected, red dashed = ground truth"]
    for i, line in enumerate(words):
        cv2.putText(picture, line, (12, 32 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(picture, line, (12, 32 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    path = os.path.join(common.RESULTS_DIR, f"calibration_overlay_{common.video_stem(video['file'])}.jpg")
    cv2.imwrite(path, picture)
    return path


def save_montage(paths):
    """Every overlay in two columns, rows padded to the taller tile."""
    tiles = []
    for path in paths:
        picture = cv2.imread(path) if path else None
        if picture is not None:
            tiles.append(cv2.resize(picture, (TILE_WIDTH, int(round(picture.shape[0] * TILE_WIDTH / picture.shape[1]))),
                                    interpolation=cv2.INTER_AREA))
    if not tiles:
        return None
    if len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[-1]))
    rows = []
    for left, right in zip(tiles[0::2], tiles[1::2]):
        height = max(left.shape[0], right.shape[0])
        pad = [cv2.copyMakeBorder(t, 0, height - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in (left, right)]
        rows.append(np.hstack(pad))
    path = os.path.join(common.RESULTS_DIR, 'calibration_montage.jpg')
    cv2.imwrite(path, np.vstack(rows))
    return path


# ---------------------------------------------------------------- main

def summary_row(video):
    if video['status'] == 'failed':
        return [video['label'], video.get('view', ''), 'n/a', fmt(video.get('elevation_tripod_deg')), 'n/a',
                'failed', video['reason'], '', '', '', '', '', '', '', '']
    no_gt = not video['has_gt']
    errors = 'no GT' if no_gt else f"{fmt(video.get('err_mean_px'))} / {fmt(video.get('err_max_px'))}"
    errors_mm = 'no GT' if no_gt else f"{fmt(video.get('err_mean_mm'), 0)} / {fmt(video.get('err_median_mm'), 0)}"
    jitter = f"{fmt(video.get('jitter_px'))} / {fmt(video.get('jitter_clean_px'))}"
    frames = str(video['frames_read'])
    if video['frames_read'] != video['frames_sampled']:
        frames = f"{video['frames_read']}/{video['frames_sampled']}"
    cloth = ', '.join(fmt(video.get(key), 0) for key in ('cloth_h', 'cloth_s', 'cloth_v'))
    light = ' / '.join(fmt(video.get(key), 0) for key in ('light_v', 'light_spread', 'light_drift'))
    return [video['label'], video.get('view') or 'n/a', fmt(video.get('elevation_image_deg')),
            fmt(video.get('elevation_tripod_deg')), light, frames, fmt(video.get('found_pct'), 0),
            'no GT' if no_gt else fmt(video.get('gross_pct'), 0), errors, errors_mm,
            'no GT' if no_gt else fmt(video.get('err_median_px')),
            'no GT' if no_gt else fmt(video.get('iou_mean'), 3), jitter,
            'no GT' if no_gt else str(video.get('flipped_frames', 0)), cloth]


def main():
    parser = argparse.ArgumentParser(description='Table calibration accuracy over camera position and lighting')
    parser.add_argument('--csv', default=common.VIDEOS_CSV, help='video list (default eval/videos.csv)')
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        count = common.write_videos_template(args.csv)
        raise SystemExit(f"Created {args.csv} with {count} videos. Fill in label, view, height_m, distance_m, "
                         f"lighting and session (delete rows you do not want), then run this again.")

    os.makedirs(common.RESULTS_DIR, exist_ok=True)
    videos = []
    frame_rows = []
    for row in common.read_videos(args.csv):
        print(f"{row['label']}: {row['file']}")
        try:
            video = evaluate_video(row, frame_rows)
        except Exception as error:                            # the video itself could not be used
            video = {key: row.get(key, '') for key in ('label', 'file', 'view', 'height_m', 'distance_m',
                                                       'lighting', 'session')}
            video.update({'elevation_tripod_deg': common.elevation_deg(row), 'status': 'failed',
                          'reason': f"{type(error).__name__}: {error}"[:120], 'has_gt': False})
        print(f"  {video['status']}  found {fmt(video.get('found_pct'), 0)}%  "
              f"err {fmt(video.get('err_mean_px'))} px  {video.get('reason', '')}")
        videos.append(video)

    common.write_csv(os.path.join(common.RESULTS_DIR, 'calibration_frames.csv'), frame_rows, FRAME_COLUMNS)
    common.write_csv(os.path.join(common.RESULTS_DIR, 'calibration.csv'), videos, VIDEO_COLUMNS)
    montage = save_montage([video.get('overlay') for video in videos])

    header = ['label', 'view', 'elevation, image (deg)', 'elevation, tripod (deg)', 'light V / spread / drift',
              'frames', 'found %', 'gross fail %', 'corner err mean/max (px)', 'corner err mean/median (mm)',
              'corner err median (px)', 'IoU', 'jitter all/clean (px)', 'flipped', 'cloth H,S,V']
    late = [f"{v['label']} from {fmt(v['start_s'], 0)} s" for v in videos if v.get('start_s', 1.0) != 1.0]
    notes = [f"{FRAMES_PER_VIDEO} frames per video, evenly spaced from 1 s to the end"
             + (f" (later where the camera was still being set up: {', '.join(late)})" if late else '')
             + "; calibration exactly as BoardWatcher.calibrate (find_cloth_center + board_reading.make_reference), "
             "config.py unchanged.",
             "Ground truth: hand-clicked playing-area corners (cushion noses extended) on one frame per video "
             "(eval/gt/); the camera does not move after the start frame (checked by phase correlation for IMG_0712).",
             f"Gross fail = mean corner error over {GROSS_FRACTION:.0%} of the play length "
             f"({GROSS_FRACTION * common.PLAY_LENGTH_MM:.0f} mm): the quad is not the table (e.g. a player over it). "
             "These frames dominate the means; medians, clean jitter and flipped exclude them.",
             "Systematic offset: the pipeline's corners follow the outer edge of the cloth on the cushion tops, "
             "the ground truth is the playing area (cushion noses), so even a correct calibration is about one "
             "cushion width out and its IoU stays below 1 (about 0.85-0.9).",
             f"mm = error on the table plane through the ground-truth homography, at "
             f"{common.PLAY_LENGTH_MM} mm play length ({common.MM_PER_MAP_PX:.3f} mm per map px).",
             "Found = the pipeline returned 4 finite corners without an error. It has no confidence check, "
             "so a wrong quad (e.g. a player over the table) still counts as found: use error and IoU for correctness.",
             "Corner err mean = mean over found frames of the per-frame mean of the 4 corner errors; max = largest "
             "single corner error. IoU = detected vs ground-truth quad. Jitter = mean over corners of "
             "sqrt(std_x^2 + std_y^2) across found frames.",
             "Cloth H,S,V = median over frames of the pipeline's own 40x40 cloth sample (OpenCV HSV, H 0-180).",
             f"Elevation, image = camera elevation above the table centre, from solvePnP on the ground-truth corners "
             f"with a {common.PLAY_LENGTH_MM}x{common.PLAY_WIDTH_MM} mm table and an assumed "
             f"{common.PHONE_HFOV_DEG} deg horizontal field of view (about +/-1 deg over 60-76 deg); worst corner "
             f"reprojection error per video in calibration.csv (pose_reprojection_px); n/a where it is over "
             f"{MAX_POSE_REPROJECTION_PX} px (the screen recording is cropped and rescaled, so the phone camera "
             f"model does not fit it). Elevation, tripod = "
             f"degrees(atan(height_m / distance_m)) from videos.csv where measured.",
             "Light (not recorded on the day, measured from the frames): V = median brightness (HSV value, 0-255) "
             "of cloth pixels inside the table; spread = median over frames of p95 - p5 of that brightness "
             "(uneven light, shadows, window glare); drift = std over the video of the per-frame median "
             "(light changing during the game)."]
    notes.append("Flipped = frames (not gross failures) where put_long_side_first put a SHORT side first, so the "
                 "top-down map would come out the wrong way round.")
    failed = [r for r in frame_rows if r['status'] == 'failed']
    notes.append(f"Failed frames: {len(failed)} of {len(frame_rows)} (reasons in calibration_frames.csv).")
    if montage:
        notes.append(f"Overlays: results/eval/calibration_overlay_<video>.jpg, montage: {os.path.relpath(montage, common.REPO_DIR)}")
    common.write_summary('calibration', header, [summary_row(v) for v in videos], notes)


if __name__ == '__main__':
    main()
