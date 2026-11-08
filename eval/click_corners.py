"""
Ground truth for the calibration test: click the four table corners on one frame of a video.

How to run:  python eval/click_corners.py "Videos/Game videos at normal diagonal view/IMG_7429.MOV" [--frame N]

- Shows frame N (default: the first frame after 1 s), scaled to fit the screen,
  with a 4x magnifier round the mouse for exact clicks.
- Click the four corners of the PLAYING AREA: where the two cushion noses would meet if they
  carried on across the pocket, at cloth level. That is the area PLAY_LENGTH_MM measures.
- Your first two clicks must be the two ends of one LONG side (a cushion with a middle pocket).
  The other two can go in either direction.
- u = undo the last click, s = save, q / esc = quit without saving.

Saved to eval/gt/<stem>_corners.json, in ORIGINAL pixel units, reordered into the pipeline's
corner order (see ORDER_TEXT).
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np

import common

# The corner order the pipeline uses, read from table_corners.py:
# - find_corners sorts the four sides by the angle of their outward normal (atan2, image y down),
#   so the corners come out going CLOCKWISE round the table as seen on screen
# - put_long_side_first then shifts by one if needed so corner 0 -> corner 1 is a LONG side
#   (decided from where the middle pockets are), which is the map edge (0,0) -> (800,0)
# - which of the two long sides comes first is NOT fixed: it depends on which side's normal has
#   the smallest angle, i.e. on how the table sits in the picture
ORDER_TEXT = ["Pipeline order: clockwise on screen, corner 0 -> 1 is a LONG side",
              "(which long side comes first depends on the table's angle)"]
CLICK_TEXT = "Click playing-area corners: first 2 clicks = ends of a LONG side.  u undo  s save  q quit"

MAX_WIDTH = 1600
MAX_HEIGHT = 900
ZOOM = 4
ZOOM_HALF = 40                 # original pixels each side of the cursor shown in the magnifier
# Clicks that no camera can explain this badly are refused. Good clicks give 5-30 px (about 110 on the
# cropped, rescaled screen recording); a short side clicked first gives about 1200.
MAX_REPROJECTION_PX = 200


def clockwise_from_long_side(clicks):
    """
    The four clicks reordered clockwise on screen, starting so that corner 0 -> corner 1 is the
    long side given by the first two clicks. None if those two clicks are not neighbouring corners.
    """
    points = np.array(clicks, dtype=float)
    centre = points.mean(axis=0)
    # Image y points down, so increasing atan2 angle goes clockwise on screen.
    order = list(np.argsort(np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0])))
    first, second = order.index(0), order.index(1)
    if (first + 1) % 4 == second:
        start = first                  # clicked clockwise along the long side
    elif (second + 1) % 4 == first:
        start = second                 # clicked anticlockwise: start from the second click
    else:
        return None
    return points[np.roll(order, -start)]


def draw_screen(frame, scale, clicks, mouse):
    """The scaled frame with the clicks, the pipeline numbering preview and the magnifier."""
    picture = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    for i, (x, y) in enumerate(clicks):
        point = (int(x * scale), int(y * scale))
        cv2.circle(picture, point, 5, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(picture, f"click {i + 1}", (point[0] + 8, point[1] + 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 255), 2, cv2.LINE_AA)
    if len(clicks) == 4:
        ordered = clockwise_from_long_side(clicks)
        if ordered is None:
            status = "First two clicks are not neighbouring corners: press u and redo"
        else:
            status = "Pipeline numbering shown in green: press s to save"
            for i, (x, y) in enumerate(ordered):
                cv2.putText(picture, str(i), (int(x * scale) + 8, int(y * scale) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.polylines(picture, [np.round(ordered * scale).astype(np.int32)], True, (0, 255, 0), 1)
    else:
        status = f"{len(clicks)}/4 corners"
    for i, words in enumerate([CLICK_TEXT] + ORDER_TEXT + [status]):
        cv2.putText(picture, words, (10, 24 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(picture, words, (10, 24 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    if mouse is not None:
        draw_magnifier(picture, frame, mouse)
    return picture


def draw_magnifier(picture, frame, mouse):
    """A ZOOM x close-up of the original frame round the cursor, bottom-right, with a crosshair."""
    x, y = int(round(mouse[0])), int(round(mouse[1]))
    padded = cv2.copyMakeBorder(frame, ZOOM_HALF, ZOOM_HALF, ZOOM_HALF, ZOOM_HALF, cv2.BORDER_CONSTANT)
    crop = padded[y:y + 2 * ZOOM_HALF + 1, x:x + 2 * ZOOM_HALF + 1]
    close_up = cv2.resize(crop, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_NEAREST)
    size = close_up.shape[0]
    middle = size // 2
    cv2.line(close_up, (middle, 0), (middle, size), (0, 255, 255), 1)
    cv2.line(close_up, (0, middle), (size, middle), (0, 255, 255), 1)
    height, width = picture.shape[:2]
    if size < height and size < width:
        picture[height - size:, width - size:] = close_up


def save_gt(file, index, frame, clicks):
    ordered = clockwise_from_long_side(clicks)
    os.makedirs(common.GT_DIR, exist_ok=True)
    record = {'video': file, 'frame': int(index), 'width': int(frame.shape[1]), 'height': int(frame.shape[0]),
              'corners': [[round(float(x), 1), round(float(y), 1)] for x, y in ordered],
              'clicks': [[round(float(x), 1), round(float(y), 1)] for x, y in clicks],
              'corner_order': 'pipeline: clockwise on screen, corner 0 -> 1 is a long side',
              'corner_definition': 'playing area: cushion-nose lines extended across the pocket, at cloth level'}
    path = common.gt_path(file)
    with open(path, 'w') as handle:
        json.dump(record, handle, indent=2)
    return path


def summarise_gt():
    """Markdown table of every ground-truth file in eval/gt/."""
    rows = []
    for path in sorted(glob.glob(os.path.join(common.GT_DIR, '*_corners.json'))):
        try:
            with open(path) as handle:
                record = json.load(handle)
            corners = ' '.join(f"({x:.0f},{y:.0f})" for x, y in record['corners'])
            camera = common.camera_from_corners(record['corners'], record['width'], record['height'])
            pose = ['n/a'] * 4 if camera is None else [
                common.fmt(camera['elevation_deg']), common.fmt(camera['height_m'], 2),
                common.fmt(camera['distance_m'], 2), common.fmt(camera['reprojection_px'], 0)]
            rows.append([record['video'], record['frame'], f"{record['width']}x{record['height']}", corners] + pose)
        except Exception as error:
            rows.append([os.path.basename(path), 'failed', str(error)[:60], '', '', '', '', ''])
    header = ['video', 'frame', 'size (px)', 'corners 0-3 (px)', 'elevation (deg)', 'camera height (m)',
              'camera distance (m)', 'reprojection (px)']
    common.write_summary('click_corners', header, rows,
                         notes=['Corner order: ' + ' '.join(ORDER_TEXT),
                                'Corners are playing-area corners (cushion noses extended), original pixels.',
                                f"Camera position from the corners (common.camera_from_corners, "
                                f"{common.PHONE_HFOV_DEG} deg field of view assumed). A reprojection error of more "
                                f"than about 20 px means the clicks are off, or the first two were a short side."])


def main():
    parser = argparse.ArgumentParser(description='Click the four table corners for calibration ground truth')
    parser.add_argument('video', help='video file, relative to the repo root or absolute')
    parser.add_argument('--frame', type=int, default=None, help='frame number (default: first frame after 1 s)')
    args = parser.parse_args()

    for line in ORDER_TEXT:
        print(line)
    capture, fps, frame_count = common.open_video(args.video)
    index = args.frame if args.frame is not None else int(round(fps * 1.0))
    frame = common.read_frame(capture, index)
    capture.release()
    if frame is None:
        raise SystemExit(f"Could not read frame {index} of {args.video}")

    scale = min(1.0, MAX_WIDTH / frame.shape[1], MAX_HEIGHT / frame.shape[0])
    clicks = []
    mouse = [None]

    def on_mouse(event, x, y, flags, param):
        mouse[0] = (x / scale, y / scale)          # back to original pixel units
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append(mouse[0])

    window = f"click corners - {common.video_stem(args.video)} frame {index}"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        cv2.imshow(window, draw_screen(frame, scale, clicks, mouse[0]))
        key = cv2.waitKey(20) & 0xFF
        if key in (ord('q'), 27):
            print('Quit without saving')
            break
        if key == ord('u') and clicks:
            clicks.pop()
        if key == ord('s'):
            ordered = clockwise_from_long_side(clicks) if len(clicks) == 4 else None
            camera = common.camera_from_corners(ordered, frame.shape[1], frame.shape[0]) if ordered is not None else None
            if camera and camera['reprojection_px'] > MAX_REPROJECTION_PX:
                # A real table seen from this camera cannot look like these clicks: almost always
                # the first two clicks were a SHORT side.
                print(f"Not saved: reprojection {camera['reprojection_px']:.0f} px. Were the first two clicks "
                      f"a short side? Press u four times and click again.")
            elif ordered is not None:
                print(f"Saved {save_gt(args.video, index, frame, clicks)}")
                break
            else:
                print('Need 4 clicks, the first two along a long side')
    cv2.destroyAllWindows()
    summarise_gt()


if __name__ == '__main__':
    main()
