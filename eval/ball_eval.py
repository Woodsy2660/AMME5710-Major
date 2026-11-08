"""
How well does the settled-board ball detector find and colour the balls, compared with a
Hough-circle baseline, and what does the black-vs-shadow test do?

How to run:
    python eval/ball_eval.py                          the pipeline (as read_board)
    python eval/ball_eval.py --method hough           the Hough-circle baseline
    python eval/ball_eval.py --no-shadow-test         the pipeline without the black-vs-shadow test

Every frame in eval/frames/ with a hand annotation (eval/gt/<stem>_balls.json, from annotate_balls.py):
- detection: the table is found on that frame exactly as make_reference does (find_cloth_center first,
  as BoardWatcher.calibrate), then the balls exactly as read_board does, or with the baseline.
  Both methods get the same calibration.
- scoring: detections are paired with the hand-clicked centres by linear_sum_assignment on photo
  distance, a pair only counting if it is within one expected ball radius. The expected radius, the
  top-down positions (mm) and the cushion line come from the hand-clicked table corners of the
  video (eval/gt/<video>_corners.json), not from the pipeline's own calibration, so a calibration
  error cannot change the yardstick. Frames whose video has no corner ground truth fall back to the
  pipeline's calibration (scoring_map = 'pipeline' in the per-ball CSV).

Baseline: cv2.HoughCircles on the blurred grey frame inside the table hull; each circle is coloured
(and dropped if it holds no ball colour) with the pipeline's own ball_detection.colour_at, so only the
step that finds the centres differs.

Ablation: --no-shadow-test sets config.BLACK_MIN_COLOURLESS_SHARE = 0 at run time (config.py is not
edited), so a dark patch on the cloth that keeps the cloth's colour is no longer rejected as a shadow.

Outputs in results/eval/: ball_eval_<run>_per_ball.csv, ball_eval_<run>_summary.csv / .md,
ball_eval_<run>_confusion.csv, ball_overlays/<run>/<frame>.jpg, and, from whichever runs exist,
ball_eval_methods.md and ball_eval_examples.jpg.
"""

import argparse
import csv
import glob
import json
import os

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

import common
import config
import ball_detection
import board_reading
import pick_frames
import table_mask
import top_down
from common import fmt

COLOURS = ['red', 'yellow', 'black', 'cue']
LETTER = {'red': 'R', 'yellow': 'Y', 'black': 'B', 'cue': 'C', '': '?'}
TOUCH_RADII = 2.2            # another ball centre this close (map radii) = touching
CUSHION_RADII = 1.5          # this close to the cushion line (photo radii) = against the cushion
HOUGH_PARAM2 = [14, 18, 22, 26, 30]
HOUGH_BLUR = (9, 9)
HOUGH_SIGMA = 2
FAR = 1e6                    # assignment cost for a pair further apart than one radius

PER_BALL_COLUMNS = ['image', 'run', 'status', 'reason', 'kind', 'gt_colour', 'pred_colour', 'colour_correct',
                    'hidden', 'touching', 'cushion', 'gt_x', 'gt_y', 'det_x', 'det_y', 'radius_px', 'dist_px',
                    'pos_err_mm', 'scoring_map']
SUMMARY_COLUMNS = ['row', 'GT', 'TP', 'FP', 'FN', 'precision', 'recall', 'F1', 'colour_acc',
                   'pos_err_median_mm', 'pos_err_mean_mm']
SUMMARY_ROWS = COLOURS + ['all', 'isolated', 'touching', 'hidden', 'cushion']


# ---------------------------------------------------------------- frames and ground truth

def annotated_frames():
    """[(image path, ground-truth balls, frames.csv row or None)] for every annotated frame."""
    listed = pick_frames.read_frames_csv()
    frames = []
    for image in sorted(glob.glob(os.path.join(pick_frames.FRAMES_DIR, '*.jpg'))):
        name = os.path.basename(image)
        gt_file = os.path.join(common.GT_DIR, f"{os.path.splitext(name)[0]}_balls.json")
        if os.path.exists(gt_file):
            with open(gt_file) as handle:
                frames.append((image, json.load(handle), listed.get(name)))
    return frames


def scoring_geometry(image, frame_row, reference):
    """
    (to_map, to_photo, outline contour, source) used to SCORE a frame: the hand-clicked playing-area
    corners of its video if there are any, else the pipeline's own calibration of the frame.
    """
    if frame_row:
        path = common.gt_path(frame_row['video'])
        if os.path.exists(path):
            with open(path) as handle:
                record = json.load(handle)
            # Corners clicked before start_s (camera still being set up) do not hold for later frames.
            fps = float(frame_row['frame']) / float(frame_row['time_s']) if float(frame_row['time_s']) else 30.0
            start_frame = pick_frames.start_seconds(frame_row['video']) * fps
            if (record['height'], record['width']) == image.shape[:2] and record['frame'] >= start_frame:
                corners = np.array(record['corners'], dtype=np.float32)
                to_map, to_photo = top_down.make_map_transform(corners)
                return to_map, to_photo, corners.reshape(-1, 1, 2), 'gt corners'
    return reference['to_map'], reference['to_photo'], reference['table_hull'].astype(np.float32), 'pipeline'


def tag_balls(gt_balls, to_map, to_photo, outline):
    """Adds radius_px, map_x, map_y, touching and cushion to each ground-truth ball."""
    for ball in gt_balls:
        ball['radius_px'] = top_down.ball_radius_in_photo(to_map, to_photo, ball['x'], ball['y'])
        ball['map_x'], ball['map_y'] = top_down.move_point(to_map, ball['x'], ball['y'])
    touch = TOUCH_RADII * config.BALL_RADIUS_ON_MAP
    for ball in gt_balls:
        ball['touching'] = any(other is not ball and np.hypot(ball['map_x'] - other['map_x'],
                                                              ball['map_y'] - other['map_y']) <= touch
                               for other in gt_balls)
        # Signed distance to the cushion line, positive inside the table.
        inside = cv2.pointPolygonTest(outline, (float(ball['x']), float(ball['y'])), True)
        ball['cushion'] = bool(inside <= CUSHION_RADII * ball['radius_px'])
        ball['hidden'] = bool(ball.get('hidden', False))


# ---------------------------------------------------------------- detection

def calibrate(image):
    """The table found on this frame exactly as BoardWatcher.calibrate / make_reference do it."""
    click_x, click_y = table_mask.find_cloth_center(image)
    return board_reading.make_reference(image, click_x, click_y), click_x, click_y


def detect_pipeline(image, reference, click_x, click_y):
    """[(x, y, colour)] exactly as read_board finds them."""
    balls = board_reading.read_board(image, reference, click_x, click_y)
    return [(ball['photo_x'], ball['photo_y'], ball['colour']) for ball in balls]


def hough_circles(image, reference, param2):
    """Raw Hough circles [(x, y)] inside the table hull, sized from the expected radius at the table centre."""
    centre = top_down.move_point(reference['to_photo'], config.MAP_WIDTH / 2, config.MAP_HEIGHT / 2)
    radius = top_down.ball_radius_in_photo(reference['to_map'], reference['to_photo'], *centre)
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), HOUGH_BLUR, HOUGH_SIGMA)
    gray[reference['hull_mask'] == 0] = 0
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=1.5 * radius, param1=100, param2=param2,
                               minRadius=int(round(0.7 * radius)), maxRadius=int(round(1.3 * radius)))
    return [] if circles is None else [(float(x), float(y)) for x, y, _ in circles[0]]


def colour_circles(image, reference, click_x, click_y, circles):
    """
    Each circle coloured by the pipeline's own colour_at, on the same blurred HSV image read_board
    uses, with the same disc (0.55 of the expected radius there). A circle with no ball colour is
    dropped, as the pipeline drops such a candidate.
    """
    image_hsv, _ = table_mask.find_cloth_mask(image, click_x, click_y, reference['cloth_colour'])
    found = []
    for x, y in circles:
        radius = top_down.ball_radius_in_photo(reference['to_map'], reference['to_photo'], x, y)
        for family in ('red', 'yellow', 'darkwhite'):
            colour = ball_detection.colour_at(image_hsv, x, y, 0.55 * radius, family)
            if colour is not None:
                found.append((x, y, colour))
                break
    return found


# ---------------------------------------------------------------- scoring

def match(gt_balls, detections):
    """Pairs (gt index, detection index) within one expected radius, by linear_sum_assignment."""
    if not gt_balls or not detections:
        return []
    distance = np.array([[np.hypot(g['x'] - x, g['y'] - y) for x, y, _ in detections] for g in gt_balls])
    allowed = distance <= np.array([[g['radius_px']] for g in gt_balls])
    rows, cols = linear_sum_assignment(np.where(allowed, distance, FAR))
    return [(g, d) for g, d in zip(rows, cols) if allowed[g, d]]


def score_frame(name, run, gt_balls, detections, to_map, to_photo, source):
    """One per-ball row per TP, FN and FP."""
    rows = []
    pairs = match(gt_balls, detections)
    paired_gt = {g for g, _ in pairs}
    paired_det = {d for _, d in pairs}
    for g, ball in enumerate(gt_balls):
        row = {'image': name, 'run': run, 'status': 'ok', 'gt_colour': ball['colour'], 'hidden': ball['hidden'],
               'touching': ball['touching'], 'cushion': ball['cushion'], 'gt_x': ball['x'], 'gt_y': ball['y'],
               'radius_px': round(ball['radius_px'], 2), 'scoring_map': source}
        partner = [d for gg, d in pairs if gg == g]
        if partner:
            x, y, colour = detections[partner[0]]
            det_map = top_down.move_point(to_map, x, y)
            row.update({'kind': 'TP', 'pred_colour': colour, 'colour_correct': colour == ball['colour'],
                        'det_x': round(x, 1), 'det_y': round(y, 1),
                        'dist_px': round(float(np.hypot(x - ball['x'], y - ball['y'])), 2),
                        'pos_err_mm': float(np.hypot(det_map[0] - ball['map_x'], det_map[1] - ball['map_y'])
                                            * common.MM_PER_MAP_PX)})
        else:
            row['kind'] = 'FN'
        rows.append(row)
    for d, (x, y, colour) in enumerate(detections):
        if d not in paired_det:
            radius = top_down.ball_radius_in_photo(to_map, to_photo, x, y)
            rows.append({'image': name, 'run': run, 'status': 'ok', 'kind': 'FP', 'pred_colour': colour,
                         'det_x': round(x, 1), 'det_y': round(y, 1), 'radius_px': round(radius, 2),
                         'scoring_map': source})
    return rows


def is_true(value):
    """True for True, numpy True and the 'True' read back from a CSV."""
    return str(value) == 'True'


def summary_rows(per_ball):
    """The rows red, yellow, black, cue, all, isolated, touching, hidden, cushion."""
    gt = [r for r in per_ball if r['kind'] in ('TP', 'FN')]
    fps = [r for r in per_ball if r['kind'] == 'FP']
    groups = {colour: ([r for r in gt if r['gt_colour'] == colour], [r for r in fps if r['pred_colour'] == colour])
              for colour in COLOURS}
    groups['all'] = (gt, fps)
    groups['isolated'] = ([r for r in gt if not is_true(r['touching'])], None)
    for tag in ('touching', 'hidden', 'cushion'):
        groups[tag] = ([r for r in gt if is_true(r[tag])], None)       # false positives have no tags

    rows = []
    for name in SUMMARY_ROWS:
        balls, false = groups[name]
        tp = [r for r in balls if r['kind'] == 'TP']
        errors = [float(r['pos_err_mm']) for r in tp]
        row = {'row': name, 'GT': len(balls), 'TP': len(tp), 'FN': len(balls) - len(tp),
               'recall': len(tp) / len(balls) if balls else None,
               'colour_acc': sum(is_true(r['colour_correct']) for r in tp) / len(tp) if tp else None,
               'pos_err_median_mm': float(np.median(errors)) if errors else None,
               'pos_err_mean_mm': float(np.mean(errors)) if errors else None}
        if false is not None:
            row['FP'] = len(false)
            row['precision'] = len(tp) / (len(tp) + len(false)) if tp or false else None
            if row['precision'] and row['recall']:
                row['F1'] = 2 * row['precision'] * row['recall'] / (row['precision'] + row['recall'])
        rows.append(row)
    return rows


def confusion_rows(per_ball):
    rows = []
    for colour in COLOURS:
        balls = [r for r in per_ball if r['kind'] in ('TP', 'FN') and r['gt_colour'] == colour]
        row = {'gt': colour, 'missed': sum(r['kind'] == 'FN' for r in balls)}
        for predicted in COLOURS:
            row[predicted] = sum(r['kind'] == 'TP' and r['pred_colour'] == predicted for r in balls)
        rows.append(row)
    row = {'gt': 'false positive', 'missed': ''}
    for predicted in COLOURS:
        row[predicted] = sum(r['kind'] == 'FP' and r['pred_colour'] == predicted for r in per_ball)
    rows.append(row)
    return rows


# ---------------------------------------------------------------- pictures

def dashed_circle(picture, centre, radius, colour, thickness):
    for start in range(0, 360, 30):
        cv2.ellipse(picture, centre, (radius, radius), 0, start, start + 15, colour, thickness, cv2.LINE_AA)


def label(picture, words, origin, colour):
    """Text on a black box (an outline drawn as thicker text does not line up in OpenCV 5)."""
    (width, height), _ = cv2.getTextSize(words, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(picture, (origin[0] - 2, origin[1] - height - 4), (origin[0] + width + 2, origin[1] + 4),
                  (0, 0, 0), -1)
    cv2.putText(picture, words, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)


def draw_overlay(image, rows):
    """TP green, FP red, FN yellow dashed; colour letter at each (wrong colour: predicted(true))."""
    picture = image.copy()
    for row in rows:
        radius = int(round(float(row['radius_px']) * 1.2)) + 2
        if row['kind'] == 'FN':
            centre = (int(round(float(row['gt_x']))), int(round(float(row['gt_y']))))
            dashed_circle(picture, centre, radius, (0, 230, 255), 2)
            label(picture, LETTER[row['gt_colour']], (centre[0] + radius, centre[1]), (0, 230, 255))
            continue
        centre = (int(round(float(row['det_x']))), int(round(float(row['det_y']))))
        colour = (0, 220, 0) if row['kind'] == 'TP' else (0, 0, 255)
        cv2.circle(picture, centre, radius, colour, 2, cv2.LINE_AA)
        words = LETTER[row['pred_colour']]
        if row['kind'] == 'TP' and not is_true(row['colour_correct']):
            words = f"{words}({LETTER[row['gt_colour']]})"
        label(picture, words, (centre[0] + radius, centre[1]), colour)
    label(picture, 'green = found (TP)   red = false positive   yellow dashed = missed (FN)   '
                   'letter = colour given, (true colour) if wrong', (12, 30), (255, 255, 255))
    return picture


def overlay_path(run, image_name):
    return os.path.join(common.RESULTS_DIR, 'ball_overlays', run, image_name)


# ---------------------------------------------------------------- one run

def detect_frame(name, image, gt_balls, frame_row, method):
    """
    One frame: the table found on it, the ground truth tagged, and the balls detected (pipeline), or
    the Hough circles for every param2 (hough, coloured later in choose_param2).
    Returns the result dict score_all uses, with 'reason' instead if the table could not be found.
    """
    gt_balls = [dict(ball) for ball in gt_balls]
    try:
        reference, click_x, click_y = calibrate(image)
        to_map, to_photo, outline, source = scoring_geometry(image, frame_row, reference)
        tag_balls(gt_balls, to_map, to_photo, outline)
        if method == 'pipeline':
            detections = detect_pipeline(image, reference, click_x, click_y)
            circles = None
        else:
            circles = {p: hough_circles(image, reference, p) for p in HOUGH_PARAM2}
            detections = None
        return {'name': name, 'image': image, 'gt': gt_balls, 'to_map': to_map, 'to_photo': to_photo,
                'source': source, 'reference': reference, 'click': (click_x, click_y), 'circles': circles,
                'detections': detections}
    except Exception as error:                                # the table could not be found on this frame
        return {'name': name, 'image': image, 'gt': gt_balls, 'reason': f"{type(error).__name__}: {error}"[:120]}


def evaluate_frame(name, image, gt_balls, frame_row, run='pipeline'):
    """
    The full pipeline on one frame (calibration, read_board, scoring against the hand labels).
    Returns (per-ball rows, the result dict). Used by brightness_sweep.py on modified frames.
    """
    result = detect_frame(name, image, gt_balls, frame_row, 'pipeline')
    return score_all([result], run), result


def detect_all(frames, method):
    """detect_frame for every annotated frame."""
    return [detect_frame(os.path.basename(path), cv2.imread(path), gt_balls, frame_row, method)
            for path, gt_balls, frame_row in frames]


def score_all(results, run):
    rows = []
    for result in results:
        if 'reason' in result:
            # The system cannot see the table, so it finds none of the balls: each one is a miss.
            rows += [{'image': result['name'], 'run': run, 'status': 'failed', 'reason': result['reason'],
                      'kind': 'FN', 'gt_colour': b['colour'], 'hidden': bool(b.get('hidden')), 'touching': False,
                      'cushion': False, 'gt_x': b['x'], 'gt_y': b['y'], 'radius_px': 0} for b in result['gt']]
            continue
        rows += score_frame(result['name'], run, result['gt'], result['detections'], result['to_map'],
                            result['to_photo'], result['source'])
    return rows


def choose_param2(results):
    """Colours the circles for every param2 and keeps the one with the best overall F1."""
    f1_by_param2 = {}
    for param2 in HOUGH_PARAM2:
        for result in results:
            if 'reason' not in result:
                result['detections'] = colour_circles(result['image'], result['reference'], *result['click'],
                                                      result['circles'][param2])
        all_row = summary_rows(score_all(results, 'hough'))[SUMMARY_ROWS.index('all')]
        f1_by_param2[param2] = all_row.get('F1') or 0.0
        print(f"  param2 {param2}: F1 {f1_by_param2[param2]:.3f}")
    best = max(HOUGH_PARAM2, key=lambda p: f1_by_param2[p])          # ties: the first (lowest) value
    for result in results:
        if 'reason' not in result:
            result['detections'] = colour_circles(result['image'], result['reference'], *result['click'],
                                                  result['circles'][best])
    return best, f1_by_param2


def save_run(run, results, rows, only_black):
    common.write_csv(os.path.join(common.RESULTS_DIR, f"ball_eval_{run}_per_ball.csv"), rows, PER_BALL_COLUMNS)
    summary = summary_rows(rows)
    if only_black:
        summary = [row for row in summary if row['row'] == 'black']
    common.write_csv(os.path.join(common.RESULTS_DIR, f"ball_eval_{run}_summary.csv"), summary, SUMMARY_COLUMNS)
    common.write_csv(os.path.join(common.RESULTS_DIR, f"ball_eval_{run}_confusion.csv"), confusion_rows(rows),
                     ['gt'] + COLOURS + ['missed'])
    os.makedirs(os.path.dirname(overlay_path(run, 'x')), exist_ok=True)
    for result in results:
        cv2.imwrite(overlay_path(run, result['name']),
                    draw_overlay(result['image'], [r for r in rows if r['image'] == result['name']]))
    return summary


def table_cells(row):
    return [row['row'], row['GT'], row['TP'], row.get('FP', 'n/a'), row['FN'], fmt(row.get('precision'), 3),
            fmt(row.get('recall'), 3), fmt(row.get('F1'), 3), fmt(row.get('colour_acc'), 3),
            fmt(row.get('pos_err_median_mm')), fmt(row.get('pos_err_mean_mm'))]


# ---------------------------------------------------------------- comparison and examples

def read_per_ball(run):
    path = os.path.join(common.RESULTS_DIR, f"ball_eval_{run}_per_ball.csv")
    if not os.path.exists(path):
        return None
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


def write_methods(param2):
    """ball_eval_methods.md: pipeline vs hough vs pipeline without the shadow test, from the saved runs."""
    lines = ['| method | all P | all R | all F1 | all colour acc | all pos err median (mm) | touching GT | '
             'touching recall | touching colour acc | black FP |', '|' + '---|' * 10]
    for run, title in (('pipeline', 'pipeline'), ('hough', f"hough (param2 = {param2 or 'see hough run'})"),
                       ('pipeline_noshadow', 'pipeline, no shadow test')):
        rows = read_per_ball(run)
        if rows is None:
            lines.append(f"| {title} | not run yet |" + ' |' * 8)
            continue
        summary = {row['row']: row for row in summary_rows(rows)}
        every, touching = summary['all'], summary['touching']
        lines.append(f"| {title} | {fmt(every.get('precision'), 3)} | {fmt(every.get('recall'), 3)} | "
                     f"{fmt(every.get('F1'), 3)} | {fmt(every.get('colour_acc'), 3)} | "
                     f"{fmt(every.get('pos_err_median_mm'))} | {touching['GT']} | {fmt(touching.get('recall'), 3)} | "
                     f"{fmt(touching.get('colour_acc'), 3)} | {summary['black']['FP']} |")
    lines += ['', '- P / R / F1 are for finding balls (a pair within one expected radius), whatever colour was '
                  'given; colour acc = share of found balls given the right colour.',
              '- The Hough param2 was chosen on these same frames (best overall F1), which flatters the baseline.']
    path = os.path.join(common.RESULTS_DIR, 'ball_eval_methods.md')
    with open(path, 'w') as handle:
        handle.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f"Saved {path}")


def crop_around(picture, x, y, caption, width=400, height=225):
    """A width x height crop centred on (x, y), zoomed 2x, with a caption strip below."""
    big_height, big_width = picture.shape[:2]
    left = int(np.clip(x - width / 2, 0, max(0, big_width - width)))
    top = int(np.clip(y - height / 2, 0, max(0, big_height - height)))
    crop = cv2.resize(picture[top:top + height, left:left + width], (2 * width, 2 * height),
                      interpolation=cv2.INTER_NEAREST)
    return with_caption(crop, caption)


def with_caption(tile, caption):
    strip = np.full((44, tile.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(strip, caption, (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([tile, strip])


def empty_tile(caption):
    return with_caption(np.full((450, 800, 3), 60, dtype=np.uint8), caption)


def busiest_point(points, width=400, height=225):
    """The point whose crop window holds the most of the points."""
    return max(points, key=lambda p: sum(abs(q[0] - p[0]) < width / 2 and abs(q[1] - p[1]) < height / 2
                                         for q in points))


def write_examples():
    """ball_eval_examples.jpg: (a) most touching balls found, (b) a shadow, (c), (d) the worst two frames."""
    default = read_per_ball('pipeline')
    no_shadow = read_per_ball('pipeline_noshadow')
    tiles = []
    if default:
        by_frame = sorted({r['image'] for r in default})
        touching = {name: [r for r in default if r['image'] == name and r['kind'] == 'TP' and is_true(r['touching'])]
                    for name in by_frame}
        name = max(by_frame, key=lambda n: (len(touching[n]), n))
        points = [(float(r['det_x']), float(r['det_y'])) for r in touching[name]]
        picture = cv2.imread(overlay_path('pipeline', name))
        if points and picture is not None:
            x, y = busiest_point(points)
            tiles.append(crop_around(picture, x, y, f"(a) {name}: {len(points)} touching balls found (pipeline)"))
        else:
            tiles.append(empty_tile('(a) no touching balls found'))
    else:
        tiles.append(empty_tile('(a) run the pipeline first'))

    if default and no_shadow:
        shadow = None
        for row in sorted(no_shadow, key=lambda r: (r['image'], float(r['det_x'] or 0))):
            if row['kind'] != 'FP' or row['pred_colour'] != 'black':
                continue
            x, y, radius = float(row['det_x']), float(row['det_y']), float(row['radius_px'])
            # Rejected by the default test: the default run detected nothing at this spot.
            near = [r for r in default if r['image'] == row['image'] and r['det_x'] and
                    np.hypot(float(r['det_x']) - x, float(r['det_y']) - y) <= radius]
            if not near:
                shadow = row
                break
        picture = cv2.imread(overlay_path('pipeline_noshadow', shadow['image'])) if shadow else None
        if picture is not None:
            tiles.append(crop_around(picture, float(shadow['det_x']), float(shadow['det_y']),
                                     f"(b) {shadow['image']}: dark patch rejected by the shadow test, "
                                     f"black FP without it"))
        else:
            tiles.append(empty_tile('(b) no dark patch turned into a black FP'))
    else:
        tiles.append(empty_tile('(b) run the pipeline and --no-shadow-test first'))

    if default:
        errors = {}
        for row in default:
            if row['kind'] in ('FN', 'FP'):
                x = float(row['gt_x'] if row['kind'] == 'FN' else row['det_x'])
                y = float(row['gt_y'] if row['kind'] == 'FN' else row['det_y'])
                errors.setdefault(row['image'], []).append((x, y))
        worst = sorted(errors, key=lambda n: (-len(errors[n]), n))[:2]
        for letter, name in zip('cd', worst):
            picture = cv2.imread(overlay_path('pipeline', name))
            x, y = busiest_point(errors[name])
            tiles.append(crop_around(picture, x, y, f"({letter}) {name}: {len(errors[name])} FN + FP (pipeline)"))
    while len(tiles) < 4:
        tiles.append(empty_tile(f"({'abcd'[len(tiles)]}) not enough frames with errors"))
    sheet = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    path = os.path.join(common.RESULTS_DIR, 'ball_eval_examples.jpg')
    cv2.imwrite(path, sheet)
    print(f"Saved {path}")


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description='Ball detection against hand-labelled frames')
    parser.add_argument('--method', choices=['pipeline', 'hough'], default='pipeline')
    parser.add_argument('--no-shadow-test', action='store_true',
                        help='set config.BLACK_MIN_COLOURLESS_SHARE = 0 at run time; report the black rows only')
    args = parser.parse_args()

    if args.no_shadow_test:
        config.BLACK_MIN_COLOURLESS_SHARE = 0
    run = args.method + ('_noshadow' if args.no_shadow_test else '')
    frames = annotated_frames()
    if not frames:
        raise SystemExit('No annotated frames: run eval/pick_frames.py and eval/annotate_balls.py first')
    print(f"{len(frames)} annotated frames, run '{run}'")

    results = detect_all(frames, args.method)
    param2, f1_by_param2 = None, {}
    if args.method == 'hough':
        param2, f1_by_param2 = choose_param2(results)
    rows = score_all(results, run)
    summary = save_run(run, results, rows, only_black=args.no_shadow_test)

    failed = sorted({r['image'] for r in rows if r['status'] == 'failed'})
    sources = {r['image']: r.get('scoring_map') for r in rows if r['status'] == 'ok'}
    fallback = sorted(name for name, source in sources.items() if source == 'pipeline')
    notes = [f"{len(frames)} hand-labelled frames, {sum(len(f[1]) for f in frames)} balls; config.py unchanged"
             + (" (BLACK_MIN_COLOURLESS_SHARE = 0 at run time; black row only)" if args.no_shadow_test else '') + '.',
             f"A detection counts if it is within one expected ball radius of a hand-clicked centre "
             f"(linear_sum_assignment on photo distance). Per colour: TP and FN by true colour, FP by the colour given, "
             f"so precision = TP / (TP + FP). Tag rows (isolated = not touching, touching = another centre within "
             f"{TOUCH_RADII} radii, hidden as annotated, cushion = within {CUSHION_RADII} radii of the cushion line) "
             f"have no FP.",
             "Radius, mm and the cushion line come from the hand-clicked table corners of each video"
             + (f"; pipeline calibration used instead for {', '.join(fallback)}" if fallback else '') + '.',
             f"Frames where the table could not be found (all their balls counted as missed): "
             f"{', '.join(failed) if failed else 'none'}."]
    if param2:
        notes.append(f"Hough param2 = {param2}, the best overall F1 of "
                     + ', '.join(f"{p}: {f:.3f}" for p, f in f1_by_param2.items())
                     + f"; blur {HOUGH_BLUR} sigma {HOUGH_SIGMA}, dp 1.2, param1 100, radius 0.7-1.3 x the expected "
                       f"radius at the table centre. Chosen on the test frames themselves, which flatters the baseline.")
    common.write_summary(f"ball_eval_{run}", SUMMARY_COLUMNS, [table_cells(row) for row in summary], notes)

    if param2:
        with open(os.path.join(common.RESULTS_DIR, 'ball_eval_hough_param2.json'), 'w') as handle:
            json.dump({'chosen': param2, 'f1': f1_by_param2}, handle, indent=1)
    saved = os.path.join(common.RESULTS_DIR, 'ball_eval_hough_param2.json')
    if param2 is None and os.path.exists(saved):
        with open(saved) as handle:
            param2 = json.load(handle)['chosen']
    write_methods(param2)
    write_examples()


if __name__ == '__main__':
    main()
