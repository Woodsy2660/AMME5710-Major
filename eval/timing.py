"""
How long does the system take per frame?

How to run:  python eval/timing.py "Videos/Game videos at normal diagonal view/IMG_7429.MOV" [--start S] [--end S]
             [--table-edge-trim N] [--moving-camera]

The video goes through BoardWatcher exactly as gui.py's main loop does for --video (the table found
from the very first frame of the recording, then watcher.update(frame, None) on every frame, the game
started from the first agreed board and each finished shot applied to GameState), without drawing,
the window or the shot log. gui.py itself is not changed: its per-frame processing is the single call
watcher.update, which can be called on its own; the game bookkeeping after it is repeated here.

Each frame is timed with time.perf_counter (decoding the frame is timed separately and is not in
total_ms, as in gui.py's shot log). The stages are timed by wrapping, from outside and only while
this script runs, the functions board_watcher.py calls:
- motion:      shrink_to_gray, moving_pixels, update_detector, BoardWatcher.camera_moved
- obstruction: obstruction.unexplained_lumps / view_is_blocked / update_blocked / occluder_mask /
               hidden_balls, shot_detection.view_is_blocked, BoardWatcher.covers_balls
- board_read:  board_reading.read_board / same_reading, BoardWatcher.refind_table / commit
- tracking:    initialise_tracking, track_frame
- other:       the rest of update() and the game bookkeeping (total minus the four stages)

Outputs in results/eval/: timing.csv (one row per frame), fig_timing.pdf / .png (300 dpi) and
timing_summary.md, which also has the hardware line for the report.
"""

import argparse
import os
import platform
import sys
import time

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import common
import config
import board_reading
import board_watcher
import obstruction
from board_watcher import BoardWatcher
from game_state import GameState
from common import fmt

STAGES = ['motion', 'obstruction', 'board_read', 'tracking']
REAL_TIME_MS = 1000 / 30
COLUMNS = ['frame', 'state', 'status', 'total_ms', 'decode_ms'] + [f"{stage}_ms" for stage in STAGES] + \
          ['other_ms', 'shot']


# ---------------------------------------------------------------- stage timers

clock = {'stage_ms': dict.fromkeys(STAGES, 0.0), 'inside': False}


def timed(stage, function):
    """function, timed into clock['stage_ms'][stage] (a call made inside another timed call is not counted twice)."""
    def wrapper(*args, **kwargs):
        if clock['inside']:
            return function(*args, **kwargs)
        clock['inside'] = True
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            clock['stage_ms'][stage] += (time.perf_counter() - started) * 1000
            clock['inside'] = False
    return wrapper


def install_timers():
    """Wraps the calls board_watcher.py makes. Nothing in the pipeline's files is changed."""
    for name in ('shrink_to_gray', 'moving_pixels', 'update_detector'):
        setattr(board_watcher, name, timed('motion', getattr(board_watcher, name)))
    BoardWatcher.camera_moved = timed('motion', BoardWatcher.camera_moved)
    for name in ('unexplained_lumps', 'view_is_blocked', 'update_blocked', 'occluder_mask', 'hidden_balls'):
        setattr(obstruction, name, timed('obstruction', getattr(obstruction, name)))
    board_watcher.view_is_blocked = timed('obstruction', board_watcher.view_is_blocked)
    BoardWatcher.covers_balls = timed('obstruction', BoardWatcher.covers_balls)
    for name in ('read_board', 'same_reading'):
        setattr(board_reading, name, timed('board_read', getattr(board_reading, name)))
    BoardWatcher.refind_table = timed('board_read', BoardWatcher.refind_table)
    BoardWatcher.commit = timed('board_read', BoardWatcher.commit)
    for name in ('initialise_tracking', 'track_frame'):
        setattr(board_watcher, name, timed('tracking', getattr(board_watcher, name)))


# ---------------------------------------------------------------- the run

def run_headless(video, start=0.0, end=None, moving_camera=False):
    """
    gui.py's --video loop without drawing. Returns (one row per frame, shots, calibration ms, fps).
    shots: (frame, BoardWatcher result, game event or None) for every shot BoardWatcher reports; the
    event is None for a shot after the game was over (gui.py does not score those).
    Also used by threshold_sweep.py.
    """
    capture, fps, frame_count = common.open_video(video)
    ok, calibration_frame = capture.read()          # gui.py: the table is found from the very first frame
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    last_frame = int(end * fps) if end is not None else None

    started = time.perf_counter()
    watcher = BoardWatcher(calibration_frame, None, fps, moving_camera)
    calibration_ms = (time.perf_counter() - started) * 1000
    game = GameState(totals={'red': 7, 'yellow': 7})
    joined = False
    frame_number = int(start * fps)
    rows = []
    shots = []
    while last_frame is None or capture.get(cv2.CAP_PROP_POS_FRAMES) <= last_frame:
        decode_started = time.perf_counter()
        ok, frame = capture.read()
        decode_ms = (time.perf_counter() - decode_started) * 1000
        if not ok:
            break
        clock['stage_ms'] = dict.fromkeys(STAGES, 0.0)
        started = time.perf_counter()
        try:
            result = watcher.update(frame, None)
        except Exception as error:                   # gui.py shows the error and carries on
            result = None
            print(f"frame {frame_number}: {error}")
        if not joined and watcher.committed_balls is not None:
            counts = board_reading.count_colours(watcher.committed_balls)
            game.start_mid_game(counts['red'], counts['yellow'], None)
            joined = True
        if result is not None:
            event = None
            if game.winner is None:
                event = game.apply_shot(result['potted'], result['note'], result['first_hit'], result['tracked'],
                                        result['balls_moved'])
            shots.append((frame_number, result, event))
        total_ms = (time.perf_counter() - started) * 1000

        row = {'frame': frame_number, 'state': watcher.detector['state'], 'status': watcher.status,
               'total_ms': total_ms, 'decode_ms': decode_ms, 'shot': result is not None}
        for stage in STAGES:
            row[f"{stage}_ms"] = clock['stage_ms'][stage]
        row['other_ms'] = total_ms - sum(clock['stage_ms'].values())
        rows.append(row)
        frame_number = frame_number + 1
    capture.release()
    return rows, shots, calibration_ms, fps


# ---------------------------------------------------------------- report

def hardware_line():
    """CPU model, cores, RAM, OS, Python, OpenCV and NumPy versions."""
    cpu = platform.processor() or 'unknown CPU'
    try:
        with open('/proc/cpuinfo') as handle:
            names = [line.split(':', 1)[1].strip() for line in handle if line.startswith('model name')]
        cpu = names[0] if names else cpu
    except OSError:
        pass
    ram = 'unknown RAM'
    try:
        with open('/proc/meminfo') as handle:
            kb = int(next(line for line in handle if line.startswith('MemTotal')).split()[1])
        ram = f"{kb / 1024 ** 2:.1f} GB RAM"
    except (OSError, StopIteration):
        pass
    system = f"{platform.system()} {platform.release()}"
    if 'microsoft' in platform.release().lower():
        system = system + ' (WSL2; RAM is what the WSL VM is given)'
    return (f"{cpu}, {os.cpu_count()} logical cores, {ram}, {system}, Python {platform.python_version()}, "
            f"OpenCV {cv2.__version__} ({cv2.getNumThreads()} threads), NumPy {np.__version__}")


def stats(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return None, None, None
    return float(np.median(values)), float(np.mean(values)), float(np.percentile(values, 95))


def draw_figure(totals):
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8})
    figure, axes = plt.subplots(figsize=(3.4, 2.0), layout='constrained')
    median, _, p95 = stats(totals)
    top = max(np.percentile(totals, 99.5), REAL_TIME_MS) * 1.1          # the long tail is cut, not hidden in a count
    axes.hist(np.clip(totals, 0, top), bins=60, range=(0, top), color='0.55', edgecolor='none')
    axes.axvline(median, color='tab:blue', linewidth=1.2, label=f"median {median:.1f} ms")
    axes.axvline(p95, color='tab:orange', linewidth=1.2, linestyle='--', label=f"95th pct {p95:.1f} ms")
    axes.axvline(REAL_TIME_MS, color='tab:red', linewidth=1.2, linestyle=':', label='33 ms (30 fps)')
    axes.set_xlabel('Processing time per frame (ms)')
    axes.set_ylabel('Frames')
    axes.set_xlim(0, top)
    axes.grid(True, color='0.85', linewidth=0.5)
    axes.set_axisbelow(True)
    axes.legend(loc='upper right', frameon=True)
    for extension in ('pdf', 'png'):
        figure.savefig(os.path.join(common.RESULTS_DIR, f"fig_timing.{extension}"), dpi=300)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description='Per-frame processing time, headless, as gui.py --video')
    parser.add_argument('video')
    parser.add_argument('--start', type=float, default=0.0, help='as gui.py: start this many seconds in')
    parser.add_argument('--end', type=float, default=None, help='as gui.py: stop at this many seconds')
    parser.add_argument('--table-edge-trim', type=int, default=None, help='as gui.py')
    parser.add_argument('--moving-camera', action='store_true', help='as gui.py')
    args = parser.parse_args()
    if args.table_edge_trim is not None:
        config.TABLE_EDGE_TRIM = args.table_edge_trim

    install_timers()
    rows, shots, calibration_ms, fps = run_headless(args.video, args.start, args.end, args.moving_camera)
    shots = [(frame, event) for frame, _, event in shots if event is not None]     # the shots gui.py scores
    if not rows:
        raise SystemExit('No frames were read')
    os.makedirs(common.RESULTS_DIR, exist_ok=True)
    common.write_csv(os.path.join(common.RESULTS_DIR, 'timing.csv'), rows, COLUMNS)
    # The shots scored on the way, to check this run follows the game as gui.py does (compare with --log).
    common.write_csv(os.path.join(common.RESULTS_DIR, 'timing_shots.csv'),
                     [{'frame': frame, 'event': event} for frame, event in shots], ['frame', 'event'])
    totals = [row['total_ms'] for row in rows]
    draw_figure(totals)

    table = []
    for name, subset in (('all frames', rows), ('SHOT', [r for r in rows if r['state'] == 'SHOT']),
                         ('SETTLED', [r for r in rows if r['state'] == 'SETTLED'])):
        median, mean, p95 = stats([r['total_ms'] for r in subset])
        stage_means = [fmt(float(np.mean([r[f"{s}_ms"] for r in subset])), 2) if subset else 'n/a'
                       for s in STAGES + ['other']]
        over = 100.0 * sum(r['total_ms'] > REAL_TIME_MS for r in subset) / len(subset) if subset else None
        table.append([name, len(subset), fmt(median), fmt(mean), fmt(p95), fmt(over)] + stage_means)
    decode_median, decode_mean, decode_p95 = stats([r['decode_ms'] for r in rows])
    hardware = hardware_line()
    notes = [f"Video: {args.video} ({fps:.2f} fps), frames {rows[0]['frame']}-{rows[-1]['frame']}, "
             f"TABLE_EDGE_TRIM {config.TABLE_EDGE_TRIM}, moving camera {args.moving_camera}.",
             "Total = watcher.update plus the game bookkeeping, exactly as gui.py's --video loop without drawing; "
             "stages timed by wrapping the calls board_watcher.py makes (other = the rest).",
             f"Not in total: decoding the frame (median {fmt(decode_median)} ms, mean {fmt(decode_mean)} ms, "
             f"95th pct {fmt(decode_p95)} ms) and the one-off table calibration ({calibration_ms:.0f} ms).",
             f"Shots scored on the way: {len(shots)} (results/eval/timing_shots.csv).",
             f"fig_timing: x axis ends at 1.1 x the 99.5th percentile ({np.percentile(totals, 99.5):.0f} ms); "
             f"the slowest frames are counted in the last bar (max {max(totals):.0f} ms).",
             f"Hardware: {hardware}"]
    common.write_summary('timing', ['frames', 'n', 'median ms', 'mean ms', '95th pct ms', '% over 33 ms']
                         + [f"mean {s} ms" for s in STAGES + ['other']], table, notes)
    print(f"\nHardware line for the report:\n{hardware}")


if __name__ == '__main__':
    main()
