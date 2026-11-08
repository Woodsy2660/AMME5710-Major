"""
How sensitive is shot detection to its two thresholds?

How to run:  python eval/threshold_sweep.py <video> --shots eval/shots_<stem>.csv
             [--start S] [--end S] [--table-edge-trim N] [--moving-camera]

The clip runs through the headless gui.py loop (timing.run_headless) once per setting:
- motion threshold: config.MOVING_PIXEL_DIFF (the grey-level change that counts a pixel as moving,
  the 25 in the M_t formula) in MOTION_VALUES, everything else at its default
- settle hold: config.HOLD_FRAMES (still frames before SHOT becomes SETTLED) in HOLD_VALUES
Each value is set on config at run time and put back afterwards; config.py is not edited. The run
with both defaults is the same in both sweeps, so it is run once.

Logged shots are every shot BoardWatcher reports (what gui.py logs, plus any after the game is
over, so a wrong early win does not hide later detections). They are matched one to one with the
hand-annotated shot end times (eval/shots_<stem>.csv, column end_s): a logged shot matches an
annotated shot if it was logged between MATCH_BEFORE_S before and MATCH_AFTER_S after the annotated
end. Delay = logged time - annotated end time.

Outputs in results/eval/: threshold_sweep.csv, threshold_sweep_matches.csv, fig_thresholds.pdf / .png
and threshold_sweep_summary.md.
"""

import argparse
import csv
import os

import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import common
import config
import timing
from common import fmt

MOTION_VALUES = [15, 20, 25, 30, 35, 40]
HOLD_VALUES = [8, 12, 15, 20, 25]
MATCH_BEFORE_S = 3.0
MATCH_AFTER_S = 10.0
FAR = 1e6
COLUMNS = ['parameter', 'value', 'is_default', 'annotated', 'logged', 'detected', 'missed', 'false',
           'mean_delay_s', 'median_delay_s', 'recall', 'precision']


def read_shots(path):
    """Annotated shot end times in seconds (column end_s), sorted."""
    with open(path, newline='') as handle:
        rows = list(csv.DictReader(handle))
    return sorted(float(row['end_s']) for row in rows if common.number_or_none(row.get('end_s')) is not None)


def write_shots_template(path):
    with open(path, 'w', newline='') as handle:
        handle.write('end_s,notes\n')


def match_shots(annotated, logged):
    """
    One-to-one pairs (annotated index, logged index) with MATCH_BEFORE_S <= logged - annotated <= MATCH_AFTER_S,
    choosing the pairing with the smallest total |delay| (linear_sum_assignment).
    """
    if not annotated or not logged:
        return []
    delay = np.array([[t - end for t in logged] for end in annotated])
    allowed = (delay >= -MATCH_BEFORE_S) & (delay <= MATCH_AFTER_S)
    rows, cols = linear_sum_assignment(np.where(allowed, np.abs(delay), FAR))
    return [(a, l) for a, l in zip(rows, cols) if allowed[a, l]]


def run_setting(args, parameter, value, annotated):
    """(row of threshold_sweep.csv, match rows) for one setting."""
    default = getattr(config, parameter)
    setattr(config, parameter, value)
    try:
        _, shots, _, fps = timing.run_headless(args.video, args.start, args.end, args.moving_camera)
    finally:
        setattr(config, parameter, default)
    logged = [frame / fps for frame, _, _ in shots]
    pairs = match_shots(annotated, logged)
    delays = [logged[l] - annotated[a] for a, l in pairs]
    row = {'parameter': parameter, 'value': value, 'is_default': value == default, 'annotated': len(annotated),
           'logged': len(logged), 'detected': len(pairs), 'missed': len(annotated) - len(pairs),
           'false': len(logged) - len(pairs),
           'mean_delay_s': float(np.mean(delays)) if delays else None,
           'median_delay_s': float(np.median(delays)) if delays else None,
           'recall': len(pairs) / len(annotated) if annotated else None,
           'precision': len(pairs) / len(logged) if logged else None}
    matched_a = {a for a, _ in pairs}
    matched_l = {l for _, l in pairs}
    matches = [{'parameter': parameter, 'value': value, 'kind': 'detected', 'annotated_end_s': annotated[a],
                'logged_s': round(logged[l], 2), 'delay_s': round(logged[l] - annotated[a], 2)} for a, l in pairs]
    matches += [{'parameter': parameter, 'value': value, 'kind': 'missed', 'annotated_end_s': annotated[a]}
                for a in range(len(annotated)) if a not in matched_a]
    matches += [{'parameter': parameter, 'value': value, 'kind': 'false', 'logged_s': round(logged[l], 2)}
                for l in range(len(logged)) if l not in matched_l]
    return row, matches


def draw_figure(rows, annotated_count):
    """Two panels side by side: detected and false shots against each setting (fig_brightness style)."""
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8})
    figure, panels = plt.subplots(1, 2, figsize=(3.4, 2.0), layout='constrained', sharey=True)
    for axes, (parameter, label) in zip(panels, (('MOVING_PIXEL_DIFF', 'Motion threshold (grey levels)'),
                                                 ('HOLD_FRAMES', 'Settle hold (frames)'))):
        sweep = sorted((r for r in rows if r['parameter'] == parameter), key=lambda r: r['value'])
        values = [r['value'] for r in sweep]
        axes.axhline(annotated_count, color='0.6', linewidth=0.8, linestyle='--', label='Annotated')
        axes.plot(values, [r['detected'] for r in sweep], marker='o', markersize=3.5, linewidth=1.2, label='Detected')
        axes.plot(values, [r['false'] for r in sweep], marker='s', markersize=3.5, linewidth=1.2, label='False')
        default = next(r['value'] for r in sweep if r['is_default'])
        axes.axvline(default, color='0.75', linewidth=0.8, linestyle=':')
        axes.set_xlabel(label)
        axes.set_xticks(values)
        axes.set_ylim(bottom=0)
        axes.grid(True, color='0.85', linewidth=0.5)
        axes.set_axisbelow(True)
    panels[0].set_ylabel('Shots')
    panels[0].legend(loc='center left', frameon=True)
    for extension in ('pdf', 'png'):
        figure.savefig(os.path.join(common.RESULTS_DIR, f"fig_thresholds.{extension}"), dpi=300)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description='Shot detection against the motion threshold and settle hold')
    parser.add_argument('video')
    parser.add_argument('--shots', required=True, help='annotated shot end times, e.g. eval/shots_test_4.csv')
    parser.add_argument('--start', type=float, default=0.0, help='as gui.py')
    parser.add_argument('--end', type=float, default=None, help='as gui.py')
    parser.add_argument('--table-edge-trim', type=int, default=None, help='as gui.py')
    parser.add_argument('--moving-camera', action='store_true', help='as gui.py')
    args = parser.parse_args()
    if args.table_edge_trim is not None:
        config.TABLE_EDGE_TRIM = args.table_edge_trim

    if not os.path.exists(args.shots):
        write_shots_template(args.shots)
        raise SystemExit(f"Created {args.shots}: add one row per shot with the time (s) the balls stopped "
                         f"moving in end_s, then run this again.")
    annotated = read_shots(args.shots)
    if args.start or args.end is not None:
        annotated = [t for t in annotated if t >= args.start and (args.end is None or t <= args.end)]
    if not annotated:
        raise SystemExit(f"No shot times in {args.shots}")

    rows = []
    matches = []
    default_result = None
    for parameter, values in (('MOVING_PIXEL_DIFF', MOTION_VALUES), ('HOLD_FRAMES', HOLD_VALUES)):
        for value in values:
            both_default = (value == getattr(config, parameter) and
                            config.MOVING_PIXEL_DIFF == 25 and config.HOLD_FRAMES == 15)
            if both_default and default_result is not None:
                row = dict(default_result[0], parameter=parameter, value=value)
                found = [dict(m, parameter=parameter, value=value) for m in default_result[1]]
            else:
                try:
                    row, found = run_setting(args, parameter, value, annotated)
                except Exception as error:                    # recorded, the sweep carries on
                    row, found = {'parameter': parameter, 'value': value, 'annotated': len(annotated),
                                  'logged': f"failed: {type(error).__name__}: {error}"[:100]}, []
                if both_default:
                    default_result = (row, found)
            print(f"{parameter} = {value}: detected {row.get('detected')}, missed {row.get('missed')}, "
                  f"false {row.get('false')}, mean delay {fmt(row.get('mean_delay_s'), 2)} s")
            rows.append(row)
            matches += found

    common.write_csv(os.path.join(common.RESULTS_DIR, 'threshold_sweep.csv'), rows, COLUMNS)
    common.write_csv(os.path.join(common.RESULTS_DIR, 'threshold_sweep_matches.csv'), matches,
                     ['parameter', 'value', 'kind', 'annotated_end_s', 'logged_s', 'delay_s'])
    draw_figure([r for r in rows if 'detected' in r], len(annotated))

    table = [[r['parameter'], str(r['value']) + (' (default)' if r.get('is_default') else ''), r.get('logged'),
              r.get('detected', ''), r.get('missed', ''), r.get('false', ''), fmt(r.get('recall'), 3),
              fmt(r.get('precision'), 3), fmt(r.get('mean_delay_s'), 2), fmt(r.get('median_delay_s'), 2)]
             for r in rows]
    notes = [f"Video: {args.video}; {len(annotated)} annotated shots ({args.shots}); TABLE_EDGE_TRIM "
             f"{config.TABLE_EDGE_TRIM}; one setting changed at a time at run time, the other at its default.",
             f"A logged shot matches an annotated one if it was logged between {MATCH_BEFORE_S:g} s before and "
             f"{MATCH_AFTER_S:g} s after the annotated end (one to one, smallest total |delay|). "
             "Delay = logged - annotated end, so it includes the settle hold and the two agreeing board readings.",
             "Logged = every shot BoardWatcher reports, including any after a (possibly wrong) end of game.",
             "HOLD_FRAMES counts frames, so the same value is half the time on a 60 fps clip.",
             "Pairs per setting: results/eval/threshold_sweep_matches.csv. Figure: results/eval/fig_thresholds.pdf."]
    common.write_summary('threshold_sweep', ['parameter', 'value', 'logged', 'detected', 'missed', 'false', 'recall',
                                             'precision', 'mean delay (s)', 'median delay (s)'], table, notes)


if __name__ == '__main__':
    main()
