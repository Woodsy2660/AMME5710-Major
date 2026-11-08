"""
How does the whole pipeline hold up when the scene is darker or brighter?

How to run:  python eval/brightness_sweep.py

For each factor in FACTORS, every hand-labelled frame (as ball_eval.py) is converted to HSV, its V
(brightness) channel multiplied by the factor and clipped to 0-255, and converted back. The FULL
pipeline then runs on the modified frame, calibration included, so a lighting change that breaks
table detection shows up: ball_eval.evaluate_frame (find_cloth_center, make_reference, read_board,
scored against the hand labels). The scoring yardstick (hand-clicked table corners) does not change.
Factor 1.0 goes through the same HSV round trip, so every factor is treated alike.

Per factor: calibration found rate (4 corners, no error), calibration correct rate (also not a gross
failure and not a short side first, against the hand-clicked corners, as calibration_eval.py),
precision, recall, F1, colour accuracy and black false positives.

Outputs in results/eval/: brightness_sweep.csv, brightness_sweep_per_ball.csv,
fig_brightness.pdf / .png (300 dpi) and brightness_sweep_summary.md.
"""

import json
import os

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import common
import ball_eval
import calibration_eval
from common import fmt

FACTORS = [0.6, 0.7, 0.85, 1.0, 1.15, 1.3, 1.4]
COLUMNS = ['factor', 'frames', 'calib_found_pct', 'calib_correct_pct', 'failed_frames', 'GT', 'TP', 'FP', 'FN',
           'precision', 'recall', 'F1', 'colour_acc', 'black_fp']


def scale_brightness(image, factor):
    """The frame with its HSV value channel multiplied by factor (clipped to 0-255)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(np.round(hsv).astype(np.uint8), cv2.COLOR_HSV2BGR)


def calibration_correct(result, frame_row):
    """
    True if the table found on this frame is the right one the right way round: not a gross failure
    and no short side first, against the video's hand-clicked corners. None without corner ground truth.
    """
    if 'reason' in result or result['source'] != 'gt corners':
        return None if 'reason' not in result else False
    with open(common.gt_path(frame_row['video'])) as handle:
        gt = np.array(json.load(handle)['corners'], dtype=np.float32)
    corners = np.asarray(result['reference']['corners'], dtype=np.float32)
    aligned, _, flipped = calibration_eval.match_to_gt(corners, gt)
    gross = calibration_eval.corner_errors_mm(aligned, gt).mean() > \
        calibration_eval.GROSS_FRACTION * common.PLAY_LENGTH_MM
    return not flipped and not gross


def sweep_factor(frames, factor):
    """(one row of brightness_sweep.csv, per-ball rows) for one brightness factor."""
    per_ball = []
    found = 0
    correct = []
    for path, gt_balls, frame_row in frames:
        image = scale_brightness(cv2.imread(path), factor)
        rows, result = ball_eval.evaluate_frame(os.path.basename(path), image, gt_balls, frame_row,
                                                run=f"brightness_{factor}")
        for row in rows:
            row['factor'] = factor
        per_ball += rows
        if 'reason' not in result and np.isfinite(result['reference']['corners']).all():
            found = found + 1
        correct.append(calibration_correct(result, frame_row))

    summary = {row['row']: row for row in ball_eval.summary_rows(per_ball)}
    every = summary['all']
    judged = [c for c in correct if c is not None]
    return {'factor': factor, 'frames': len(frames), 'calib_found_pct': 100.0 * found / len(frames),
            'calib_correct_pct': 100.0 * sum(judged) / len(judged) if judged else None,
            'failed_frames': len({r['image'] for r in per_ball if r['status'] == 'failed'}),
            'GT': every['GT'], 'TP': every['TP'], 'FP': every['FP'], 'FN': every['FN'],
            'precision': every.get('precision'), 'recall': every.get('recall'), 'F1': every.get('F1'),
            'colour_acc': every.get('colour_acc'), 'black_fp': summary['black']['FP']}, per_ball


def draw_figure(rows):
    """Precision and recall against the brightness factor, 3.4 x 2.2 in, 8 pt, no title."""
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 8,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8})
    figure, axes = plt.subplots(figsize=(3.4, 2.2), layout='constrained')
    factors = [row['factor'] for row in rows]
    axes.plot(factors, [row['precision'] or 0 for row in rows], marker='o', markersize=3.5, linewidth=1.2,
              label='Precision')
    axes.plot(factors, [row['recall'] or 0 for row in rows], marker='s', markersize=3.5, linewidth=1.2,
              label='Recall')
    axes.set_xlabel('Brightness factor (HSV value scale)')
    axes.set_ylabel('Score')
    axes.set_ylim(0, 1)
    axes.set_xticks(factors)
    axes.set_xticklabels([f"{f:g}" for f in factors])
    axes.grid(True, color='0.85', linewidth=0.5)
    axes.legend(loc='lower center', frameon=True)
    for extension in ('pdf', 'png'):
        figure.savefig(os.path.join(common.RESULTS_DIR, f"fig_brightness.{extension}"), dpi=300)
    plt.close(figure)


def main():
    frames = ball_eval.annotated_frames()
    if not frames:
        raise SystemExit('No annotated frames: run eval/pick_frames.py and eval/annotate_balls.py first')
    os.makedirs(common.RESULTS_DIR, exist_ok=True)
    rows = []
    per_ball = []
    for factor in FACTORS:
        row, balls = sweep_factor(frames, factor)
        print(f"factor {factor}: P {fmt(row['precision'], 3)} R {fmt(row['recall'], 3)} "
              f"calibration correct {fmt(row['calib_correct_pct'], 0)}%")
        rows.append(row)
        per_ball += balls

    common.write_csv(os.path.join(common.RESULTS_DIR, 'brightness_sweep.csv'), rows, COLUMNS)
    common.write_csv(os.path.join(common.RESULTS_DIR, 'brightness_sweep_per_ball.csv'), per_ball,
                     ['factor'] + ball_eval.PER_BALL_COLUMNS)
    draw_figure(rows)

    header = ['factor', 'calib found %', 'calib correct %', 'precision', 'recall', 'F1', 'colour acc', 'black FP',
              'TP / FP / FN']
    table = [[f"{r['factor']:g}", fmt(r['calib_found_pct'], 0), fmt(r['calib_correct_pct'], 0),
              fmt(r['precision'], 3), fmt(r['recall'], 3), fmt(r['F1'], 3), fmt(r['colour_acc'], 3), r['black_fp'],
              f"{r['TP']} / {r['FP']} / {r['FN']}"] for r in rows]
    notes = [f"{len(frames)} hand-labelled frames, {rows[0]['GT']} balls per factor; each frame's HSV value channel "
             f"x factor (clipped), then the full pipeline including calibration; config.py unchanged.",
             "Calib found = 4 finite corners without an error. Calib correct = also not a gross failure "
             f"(mean corner error over {calibration_eval.GROSS_FRACTION:.0%} of the play length) and not a short side "
             "first, against the hand-clicked corners.",
             "Factor 1.0 also goes through the HSV round trip, so it can differ slightly from ball_eval.py on the "
             "unmodified frames.",
             "Figure: results/eval/fig_brightness.pdf and .png (precision and recall against the factor)."]
    common.write_summary('brightness_sweep', header, table, notes)


if __name__ == '__main__':
    main()
