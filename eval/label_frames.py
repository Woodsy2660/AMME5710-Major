"""
Opens the frames to hand-label, one after another, in annotate_balls.py.

How to run:
    python eval/label_frames.py                 every video
    python eval/label_frames.py IMG_0754        one video (any part of its frame names)
    python eval/label_frames.py --redo          open frames that are already labelled too

PER_VIDEO frames are chosen from each video's exported frames (eval/frames/frames.csv): the first,
the middle and the last, so each video is seen with a full, a half-cleared and a nearly empty table.
Frames already saved in eval/gt/ are skipped, so labelling can stop and carry on later. Press s in
the tool to save and go on to the next frame, q to skip it.
"""

import argparse
import os
import subprocess
import sys

import numpy as np

import common
import pick_frames

PER_VIDEO = 3


def chosen_frames():
    """{video: [image names]}: PER_VIDEO evenly spread frames per video, in time order."""
    by_video = {}
    for row in pick_frames.read_frames_csv().values():
        by_video.setdefault(row['video'], []).append(row)
    chosen = {}
    for video, rows in sorted(by_video.items()):
        rows.sort(key=lambda r: int(r['frame']))
        picks = sorted({int(round(i)) for i in np.linspace(0, len(rows) - 1, min(PER_VIDEO, len(rows)))})
        chosen[video] = [rows[i]['image'] for i in picks]
    return chosen


def is_labelled(image):
    return os.path.exists(os.path.join(common.GT_DIR, f"{os.path.splitext(image)[0]}_balls.json"))


def main():
    parser = argparse.ArgumentParser(description='Hand-label the chosen frames one after another')
    parser.add_argument('video', nargs='?', default='', help='only frames whose name contains this')
    parser.add_argument('--redo', action='store_true', help='also open frames that are already labelled')
    args = parser.parse_args()

    chosen = chosen_frames()
    queue = [image for images in chosen.values() for image in images
             if args.video in image and (args.redo or not is_labelled(image))]
    print(f"{len(queue)} frames to label")
    tool = os.path.join(common.EVAL_DIR, 'annotate_balls.py')
    for number, image in enumerate(queue, 1):
        print(f"[{number}/{len(queue)}] {image}")
        subprocess.run([sys.executable, tool, os.path.join(pick_frames.FRAMES_DIR, image)])

    rows = [[video, image, 'done' if is_labelled(image) else 'to do']
            for video, images in chosen.items() for image in images]
    done = sum(row[2] == 'done' for row in rows)
    common.write_summary('label_frames', ['video', 'frame', 'labelled'], rows,
                         notes=[f"{done} of {len(rows)} frames labelled ({PER_VIDEO} per video: first, middle and "
                                f"last exported frame; fewer where the video has fewer)."])


if __name__ == '__main__':
    main()
