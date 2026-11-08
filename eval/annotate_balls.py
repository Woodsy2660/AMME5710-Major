"""
Ground truth for the ball detection test: click every ball centre on one frame.

How to run:  python eval/annotate_balls.py eval/frames/test_3_0112.jpg

Keys and mouse
    r / y / b / c    colour for the next clicks: red / yellow / black / cue
    left click       add a ball centre in the current colour
    right click      remove the nearest ball
    h                hidden on / off for the nearest ball (partly covered by another ball, an arm or the cue)
    u                undo         s  save         q / esc  quit without saving
    mouse wheel      zoom in / out round the cursor     middle drag  pan     f  fit the whole frame

Each ball is drawn as a circle of the radius the pipeline expects at that pixel (the table is
found on this image with board_reading.make_reference, the radius from top_down.ball_radius_in_photo),
so a mis-click shows as a circle that does not sit on the ball.

Saved to eval/gt/<stem>_balls.json: a list of {x, y, colour, hidden} in original pixel units.
An existing file is loaded, so a frame can be finished later.
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np

import common
import board_reading
import table_mask
import top_down

WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 900
COLOURS = {'r': 'red', 'y': 'yellow', 'b': 'black', 'c': 'cue'}
DRAW = {'red': (60, 60, 230), 'yellow': (0, 210, 255), 'black': (40, 40, 40), 'cue': (255, 255, 255)}   # BGR
LETTER = {'red': 'R', 'yellow': 'Y', 'black': 'B', 'cue': 'C'}
FALLBACK_RADIUS = 12          # px, only if the table cannot be found on this image


def balls_path(image_path):
    return os.path.join(common.GT_DIR, f"{os.path.splitext(os.path.basename(image_path))[0]}_balls.json")


def radius_function(image):
    """Expected ball radius (photo px) at a pixel, from the pipeline's own calibration of this image."""
    try:
        click_x, click_y = table_mask.find_cloth_center(image)
        reference = board_reading.make_reference(image, click_x, click_y)
    except Exception as error:
        print(f"Table not found on this image ({error}): circles use a fixed {FALLBACK_RADIUS} px")
        return lambda x, y: FALLBACK_RADIUS
    return lambda x, y: top_down.ball_radius_in_photo(reference['to_map'], reference['to_photo'], x, y)


def nearest(balls, x, y):
    if not balls:
        return None
    return int(np.argmin([np.hypot(b['x'] - x, b['y'] - y) for b in balls]))


# ---------------------------------------------------------------- the zoomable view

def fit_view(image):
    """Zoom and centre that show the whole image in the window."""
    height, width = image.shape[:2]
    return {'zoom': min(WINDOW_WIDTH / width, WINDOW_HEIGHT / height), 'cx': width / 2, 'cy': height / 2}


def to_image(view, x, y):
    """Window pixel -> original image pixel."""
    return view['cx'] + (x - WINDOW_WIDTH / 2) / view['zoom'], view['cy'] + (y - WINDOW_HEIGHT / 2) / view['zoom']


def to_window(view, x, y):
    return (int(round((x - view['cx']) * view['zoom'] + WINDOW_WIDTH / 2)),
            int(round((y - view['cy']) * view['zoom'] + WINDOW_HEIGHT / 2)))


def render(image, view, balls, colour, radius_at, message):
    # The part of the image in the window, by an affine warp (fast at any zoom, grey outside the image).
    zoom = view['zoom']
    warp = np.array([[zoom, 0, WINDOW_WIDTH / 2 - view['cx'] * zoom],
                     [0, zoom, WINDOW_HEIGHT / 2 - view['cy'] * zoom]])
    picture = cv2.warpAffine(image, warp, (WINDOW_WIDTH, WINDOW_HEIGHT),
                             flags=cv2.INTER_NEAREST if zoom > 1 else cv2.INTER_AREA,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(70, 70, 70))
    for ball in balls:
        centre = to_window(view, ball['x'], ball['y'])
        radius = max(2, int(round(radius_at(ball['x'], ball['y']) * zoom)))
        outline = (0, 0, 0) if ball['colour'] != 'black' else (230, 230, 230)
        cv2.circle(picture, centre, radius, outline, 3, cv2.LINE_AA)
        cv2.circle(picture, centre, radius, DRAW[ball['colour']], 1, cv2.LINE_AA)
        cv2.drawMarker(picture, centre, DRAW[ball['colour']], cv2.MARKER_CROSS, 8, 1)
        label = LETTER[ball['colour']] + (' hidden' if ball['hidden'] else '')
        (width, height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        corner = (centre[0] + radius + 2, centre[1])
        cv2.rectangle(picture, (corner[0] - 1, corner[1] - height - 3), (corner[0] + width + 1, corner[1] + 3),
                      (0, 0, 0), -1)
        cv2.putText(picture, label, corner, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    counts = {name: sum(b['colour'] == name for b in balls) for name in DRAW}
    lines = [f"colour: {colour.upper()}   (r/y/b/c)   left add  right remove  h hidden  u undo  s save  q quit",
             f"red {counts['red']}  yellow {counts['yellow']}  black {counts['black']}  cue {counts['cue']}  "
             f"hidden {sum(b['hidden'] for b in balls)}   wheel zoom  middle-drag pan  f fit   zoom {zoom:.2f}x",
             message]
    # A dark band behind the text (an outline drawn as thicker text does not line up in OpenCV 5).
    cv2.rectangle(picture, (0, 0), (WINDOW_WIDTH, 24 * len(lines) + 10), (25, 25, 25), -1)
    for i, line in enumerate(lines):
        cv2.putText(picture, line, (10, 24 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, DRAW[colour] if i == 0
                    else (255, 255, 255), 1, cv2.LINE_AA)
    return picture


# ---------------------------------------------------------------- summary

def summarise_annotations():
    """Markdown table of every ball annotation in eval/gt/."""
    rows = []
    for path in sorted(glob.glob(os.path.join(common.GT_DIR, '*_balls.json'))):
        try:
            with open(path) as handle:
                balls = json.load(handle)
            counts = [sum(b['colour'] == name for b in balls) for name in ('red', 'yellow', 'black', 'cue')]
            rows.append([os.path.basename(path)] + counts + [sum(b['hidden'] for b in balls), len(balls)])
        except Exception as error:
            rows.append([os.path.basename(path), 'failed', str(error)[:50], '', '', '', ''])
    common.write_summary('annotate_balls', ['file', 'red', 'yellow', 'black', 'cue', 'hidden', 'total'], rows,
                         notes=[f"{len(rows)} annotated frames; centres in original pixel units."])


def main():
    parser = argparse.ArgumentParser(description='Click the ball centres on one frame')
    parser.add_argument('image')
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"Could not read {args.image}")
    radius_at = radius_function(image)
    path = balls_path(args.image)
    balls = []
    if os.path.exists(path):
        with open(path) as handle:
            balls = json.load(handle)
        print(f"Loaded {len(balls)} balls from {path}")

    state = {'view': fit_view(image), 'colour': 'red', 'history': [], 'pan': None, 'message': ''}

    def change(new_balls):
        state['history'].append([dict(b) for b in balls])
        balls[:] = new_balls

    def on_mouse(event, x, y, flags, param):
        view = state['view']
        image_x, image_y = to_image(view, x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            change(balls + [{'x': round(image_x, 1), 'y': round(image_y, 1), 'colour': state['colour'],
                             'hidden': False}])
        elif event == cv2.EVENT_RBUTTONDOWN:
            index = nearest(balls, image_x, image_y)
            if index is not None:
                change(balls[:index] + balls[index + 1:])
        elif event == cv2.EVENT_MOUSEWHEEL:
            # Zoom round the cursor: the image pixel under it stays under it.
            factor = 1.25 if cv2.getMouseWheelDelta(flags) > 0 else 0.8
            view['zoom'] = float(np.clip(view['zoom'] * factor, fit_view(image)['zoom'], 16))
            view['cx'] = image_x - (x - WINDOW_WIDTH / 2) / view['zoom']
            view['cy'] = image_y - (y - WINDOW_HEIGHT / 2) / view['zoom']
        elif event == cv2.EVENT_MBUTTONDOWN:
            state['pan'] = (x, y, view['cx'], view['cy'])
        elif event == cv2.EVENT_MBUTTONUP:
            state['pan'] = None
        elif event == cv2.EVENT_MOUSEMOVE and state['pan'] is not None:
            start_x, start_y, start_cx, start_cy = state['pan']
            view['cx'] = start_cx - (x - start_x) / view['zoom']
            view['cy'] = start_cy - (y - start_y) / view['zoom']
        state['mouse'] = (image_x, image_y)

    window = f"annotate balls - {os.path.basename(args.image)}"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        cv2.imshow(window, render(image, state['view'], balls, state['colour'], radius_at, state['message']))
        key = cv2.waitKey(20) & 0xFF
        if key in (ord('q'), 27):
            print('Quit without saving')
            break
        if chr(key) in COLOURS:
            state['colour'] = COLOURS[chr(key)]
        elif key == ord('h') and balls and 'mouse' in state:
            index = nearest(balls, *state['mouse'])
            change([dict(b, hidden=not b['hidden']) if i == index else b for i, b in enumerate(balls)])
        elif key == ord('u') and state['history']:
            balls[:] = state['history'].pop()
        elif key == ord('f'):
            state['view'] = fit_view(image)
        elif key == ord('s'):
            os.makedirs(common.GT_DIR, exist_ok=True)
            with open(path, 'w') as handle:
                json.dump(balls, handle, indent=1)
            print(f"Saved {len(balls)} balls to {path}")
            break
    cv2.destroyAllWindows()
    summarise_annotations()


if __name__ == '__main__':
    main()
