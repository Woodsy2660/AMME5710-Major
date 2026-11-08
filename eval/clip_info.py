"""
The evaluation clips at a glance, for the report's data table.

How to run:  python eval/clip_info.py [more videos ...]

Every video in the two game-video folders (common.VIDEO_FOLDERS), plus any paths given, with
duration, fps, frame count and resolution (from the decoded first frame, so a rotated phone video
shows as it is processed). view and session come from eval/videos.csv; lighting too where it was
recorded there, otherwise the lighting measured from the frames by calibration_eval.py
(results/eval/calibration.csv: cloth brightness V and its spread across the table).

Prints a Markdown table (also saved as results/eval/clip_info_summary.md) and the same rows as
LaTeX tabular lines: clip & duration & frames & view & lighting.
"""

import argparse
import csv
import os

import common
from common import fmt

CALIBRATION_CSV = os.path.join(common.RESULTS_DIR, 'calibration.csv')


def listed_videos():
    """{normalised file: videos.csv row}."""
    if not os.path.exists(common.VIDEOS_CSV):
        return {}
    return {os.path.normpath(row['file']): row for row in common.read_videos()}


def measured_lighting():
    """{normalised file: 'V 225, spread 60'} from the calibration results, where they exist."""
    if not os.path.exists(CALIBRATION_CSV):
        return {}
    with open(CALIBRATION_CSV, newline='') as handle:
        return {os.path.normpath(row['file']): f"V {float(row['light_v']):.0f}, spread {float(row['light_spread']):.0f}"
                for row in csv.DictReader(handle) if row.get('light_v')}


def all_videos(extra):
    files = []
    for folder in common.VIDEO_FOLDERS:
        folder_path = os.path.join(common.REPO_DIR, folder)
        if os.path.isdir(folder_path):
            files += [f"{folder}/{name}" for name in sorted(os.listdir(folder_path))
                      if name.lower().endswith(common.VIDEO_EXTENSIONS)]
    return files + [path for path in extra if path not in files]


def clip_row(file, listed, lighting):
    row = {'file': file, 'clip': common.video_stem(file)}
    meta = listed.get(os.path.normpath(file))
    if meta:
        row.update({'clip': meta['label'], 'view': meta.get('view') or '', 'session': meta.get('session') or '',
                    'lighting': meta.get('lighting') or lighting.get(os.path.normpath(file), '')})
    else:
        row.update({'view': '', 'session': '', 'lighting': '', 'note': 'not in videos.csv'})
    try:
        capture, fps, frame_count = common.open_video(file)
        frame = common.read_frame(capture, 0)
        capture.release()
        if frame is None:
            raise RuntimeError('no frames')
        row.update({'status': 'ok', 'fps': fps, 'frames': frame_count, 'duration': frame_count / fps,
                    'resolution': f"{frame.shape[1]}x{frame.shape[0]}"})
    except Exception as error:
        row.update({'status': 'failed', 'note': f"{type(error).__name__}: {error}"[:80]})
    return row


def latex_escape(text):
    return str(text).replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('&', r'\&').replace('%', r'\%')


def main():
    parser = argparse.ArgumentParser(description='Duration, fps, frames and resolution of the evaluation clips')
    parser.add_argument('videos', nargs='*', help='more video files (relative to the repo root or absolute)')
    args = parser.parse_args()

    listed = listed_videos()
    lighting = measured_lighting()
    rows = [clip_row(file, listed, lighting) for file in all_videos(args.videos)]

    table = [[r['clip'], r['file'], fmt(r.get('duration')), fmt(r.get('fps'), 2), r.get('frames', ''),
              r.get('resolution', ''), r['view'] or 'n/a', r['lighting'] or 'n/a', r['session'] or 'n/a',
              r.get('note', '')] for r in rows]
    common.write_summary('clip_info', ['clip', 'file', 'duration (s)', 'fps', 'frames', 'resolution', 'view',
                                       'lighting', 'session', 'note'], table,
                         notes=['Lighting: as recorded in videos.csv, otherwise measured from the frames '
                                '(calibration_eval.py: median cloth brightness V 0-255, and its p95 - p5 spread).',
                                'Duration = frames / fps from the file header; resolution from the decoded first frame.'])

    print('LaTeX (clip & duration & frames & view & lighting), clips listed in videos.csv:')
    for r in rows:
        if r['status'] == 'ok' and 'note' not in r:
            print(f"{latex_escape(r['clip'])} & {r['duration']:.0f} & {r['frames']} & {latex_escape(r['view'])} & "
                  f"{latex_escape(r['lighting'])} \\\\")


if __name__ == '__main__':
    main()
