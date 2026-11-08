"""
Ground truth for the shot detection test: mark the moment each shot ends while watching the video.

How to run:  python eval/mark_shots.py "Videos/Game videos at normal diagonal view/test_4.mp4"

Keys
    space        play / pause
    m            mark a shot end at the current frame (the moment the last ball stops rolling)
    u            remove the last mark
    j / l        back / forward 1 s          , / .   back / forward 1 frame (when paused)
    s            save and quit               q / esc  quit without saving

Saved to eval/shots_<stem>.csv (column end_s, seconds from the start of the video). An existing file
is loaded, so marking can be finished later. Mark what you see, without looking at the system's log.
"""

import argparse
import csv
import os

import cv2

import common

WINDOW_WIDTH = 1280


def shots_path(video):
    return os.path.join(common.EVAL_DIR, f"shots_{common.video_stem(video)}.csv")


def load_marks(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='') as handle:
        return sorted(float(row['end_s']) for row in csv.DictReader(handle)
                      if common.number_or_none(row.get('end_s')) is not None)


def save_marks(path, marks):
    with open(path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['end_s', 'notes'])
        for end in sorted(marks):
            writer.writerow([f"{end:.2f}", ''])


def draw(frame, index, fps, marks, paused):
    scale = WINDOW_WIDTH / frame.shape[1]
    picture = cv2.resize(frame, (WINDOW_WIDTH, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    seconds = index / fps
    near = [m for m in marks if abs(m - seconds) < 1.0]
    lines = [f"{seconds:7.2f} s  frame {index}  {'PAUSED' if paused else 'playing'}   shots marked: {len(marks)}"
             + ('   << marked here' if near else ''),
             "space play/pause   m mark shot end   u undo   j/l -/+1 s   ,/. -/+1 frame   s save   q quit"]
    cv2.rectangle(picture, (0, 0), (WINDOW_WIDTH, 58), (25, 25, 25), -1)
    for i, line in enumerate(lines):
        cv2.putText(picture, line, (10, 22 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 220, 255) if near and i == 0 else (255, 255, 255), 1, cv2.LINE_AA)
    return picture


def main():
    parser = argparse.ArgumentParser(description='Mark shot end times while watching a video')
    parser.add_argument('video')
    args = parser.parse_args()

    capture, fps, frame_count = common.open_video(args.video)
    path = shots_path(args.video)
    marks = load_marks(path)
    index = 0
    frame = common.read_frame(capture, index)
    paused = True
    window = f"mark shots - {common.video_stem(args.video)}"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    while True:
        cv2.imshow(window, draw(frame, index, fps, marks, paused))
        key = cv2.waitKey(30 if paused else max(1, int(1000 / fps) - 5)) & 0xFF     # about real speed
        jump = None
        if key in (ord('q'), 27):
            print('Quit without saving')
            break
        if key == ord('s'):
            save_marks(path, marks)
            print(f"Saved {len(marks)} shot ends to {path}")
            break
        if key == ord(' '):
            paused = not paused
        elif key == ord('m'):
            marks.append(round(index / fps, 2))
        elif key == ord('u') and marks:
            marks.pop()
        elif key == ord('j'):
            jump = index - int(round(fps))
        elif key == ord('l'):
            jump = index + int(round(fps))
        elif key == ord(',') and paused:
            jump = index - 1
        elif key == ord('.') and paused:
            jump = index + 1
        if jump is not None:
            wanted = min(max(0, jump), frame_count - 1)
            new_frame = common.read_frame(capture, wanted)
            if new_frame is not None:
                index, frame = wanted, new_frame
        elif not paused:
            ok, next_frame = capture.read()
            if ok:
                index, frame = index + 1, next_frame
            else:
                paused = True
    capture.release()
    cv2.destroyAllWindows()
    common.write_summary(f"mark_shots_{common.video_stem(args.video)}", ['shot', 'end (s)'],
                         [[i + 1, f"{m:.2f}"] for i, m in enumerate(sorted(marks))],
                         notes=[f"{len(marks)} shot ends marked by hand on {args.video} ({path})."])


if __name__ == '__main__':
    main()
