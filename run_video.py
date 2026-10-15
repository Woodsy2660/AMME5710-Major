"""
Run ball detection and the game-state tracker on an mp4 video.

Usage:
    python run_video.py "Table videos/game1.mp4"
    python run_video.py game1.mp4 --step 3 --start 120 --end 900 --show
    python run_video.py game1.mp4 --full-rack        # wait for 1/1/7/7 before tracking
    python run_video.py game1.mp4 --detect-only      # ball detection only, no tracking

For every processed frame:
    1. Find the table again (handheld camera, so it drifts). The corners are
       smoothed over time so the top-down map doesn't wobble.
    2. Detect the balls.
    3. Update the tracker, print any events, and decide whether a shot is in progress.
    4. Write an annotated frame (photo overlay + top-down map) to the output video.
"""

import argparse
import os
import time

import cv2
import numpy as np

import config
import table_mask
import table_corners
import top_down
import ball_detection
import draw
import update_game_state
from run_photo import find_cloth_center
from top_down import move_point

# Tracker functions from update_game_state.py. Rename here if yours differ.
start_tracking = update_game_state.initialise_tracking
match_detections = update_game_state.match_detections
update_states = update_game_state.compare_frames
ACTIVE, LOST, SUNK = update_game_state.ACTIVE, update_game_state.LOST, update_game_state.SUNK

# ---------------------------------------------------------------- settings
CORNER_SMOOTHING = 0.3          # 0-1: how much each new frame's corners count. Lower = steadier map.
MAX_CORNER_JUMP = 40            # photo pixels. A bigger jump is treated as a bad table detection.
MOVING_THRESHOLD = 0.01 * config.MAP_WIDTH   # map units, about a third of a ball's width
MIN_REST_FRAMES = 3             # processed frames without movement before a shot counts as over
SUNK_SECONDS = 1.0              # how long a ball must be missing near a pocket to count as sunk
FULL_RACK = {'cue': 1, 'black': 1, 'red': 7, 'yellow': 7}

DRAW_COLOURS = {'cue': (255, 255, 255), 'black': (40, 40, 40),
                'red': (0, 0, 255), 'yellow': (0, 255, 255)}     # OpenCV uses (blue, green, red)
LOST_COLOUR = (160, 160, 160)


# ---------------------------------------------------------------- table corners
class CornerStabiliser:
    """
    Smooths the table corners from frame to frame.

    - Fixes the corner order if put_long_side_first picked the opposite end this frame
      (otherwise the whole map would flip 180 degrees).
    - Ignores frames where the corners jump a long way (an arm over a corner, a bad mask).
    - Otherwise blends the new corners into the running average.
    """

    def __init__(self):
        self.smoothed = None
        self.rejected = 0

    def update(self, corners):
        original_shape = np.shape(corners)
        original_dtype = np.asarray(corners).dtype
        new = np.asarray(corners, dtype=float).reshape(-1, 2)

        if self.smoothed is None:
            self.smoothed = new
        else:
            rolled = np.roll(new, 2, axis=0)          # same corners, starting from the other end
            if self._distance(rolled) < self._distance(new):
                new = rolled

            if self._distance(new) <= MAX_CORNER_JUMP:
                self.smoothed = CORNER_SMOOTHING * new + (1 - CORNER_SMOOTHING) * self.smoothed
            else:
                self.rejected += 1                    # keep last frame's corners

        # Hand back the corners in the same shape and type the rest of the pipeline expects.
        result = self.smoothed
        if np.issubdtype(original_dtype, np.integer):
            result = np.round(result)
        return result.reshape(original_shape).astype(original_dtype)

    def _distance(self, corners):
        return np.max(np.linalg.norm(corners - self.smoothed, axis=1))


# ---------------------------------------------------------------- detection
def detect_frame(frame, stabiliser):
    """Steps 1-5 from run_photo.detect(), with the corners stabilised."""
    click_x, click_y = find_cloth_center(frame)
    image_hsv, cloth_mask = table_mask.find_cloth_mask(frame, click_x, click_y)
    table_contour, table_hull = table_mask.find_table_outline(cloth_mask, click_x, click_y)

    corners = table_corners.find_corners(table_hull)
    corners = table_corners.put_long_side_first(corners, table_contour)
    corners = stabiliser.update(corners)

    to_map, to_photo = top_down.make_map_transform(corners)
    balls = ball_detection.find_balls(image_hsv, cloth_mask, table_contour, table_hull, to_map, to_photo)
    return balls, table_hull, corners, to_photo


def count_colours(balls):
    counts = {colour: 0 for colour in FULL_RACK}
    for b in balls:
        counts[b['colour']] = counts.get(b['colour'], 0) + 1
    return counts


# ---------------------------------------------------------------- drawing
def draw_tracking(overlay, tracked_balls, matches, to_photo):
    """Label every matched ball with its ID, and show LOST balls at their last known spot."""
    for ball_id, d in matches.items():
        centre = (int(d['photo_x']), int(d['photo_y']))
        colour = DRAW_COLOURS.get(d['colour'], (255, 0, 255))
        cv2.putText(overlay, ball_id, (centre[0] + 12, centre[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
        cv2.putText(overlay, ball_id, (centre[0] + 12, centre[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

    for ball_id, info in tracked_balls.items():
        if info['state'] != LOST:
            continue
        photo_x, photo_y = move_point(to_photo, info['x'], info['y'])
        centre = (int(photo_x), int(photo_y))
        cv2.circle(overlay, centre, 14, LOST_COLOUR, 1)
        cv2.putText(overlay, ball_id + '?', (centre[0] + 12, centre[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, LOST_COLOUR, 1)


def draw_status(image, lines):
    y = 30
    for line in lines:
        cv2.putText(image, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 5)
        cv2.putText(image, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 30


def side_by_side(overlay, balls):
    """Same layout as run_photo: photo overlay on the left, top-down map on the right."""
    top_down_map = draw.draw_map(balls)
    scale = overlay.shape[0] / top_down_map.shape[0]
    map_resized = cv2.resize(top_down_map, (int(top_down_map.shape[1] * scale), overlay.shape[0]))
    return np.hstack([overlay, map_resized])


# ---------------------------------------------------------------- main loop
def parse_arguments():
    parser = argparse.ArgumentParser(description='Track pool balls through a video.')
    parser.add_argument('video', help='path to the .mp4')
    parser.add_argument('--step', type=int, default=3, help='process every Nth frame (default 3)')
    parser.add_argument('--start', type=int, default=0, help='first frame to process')
    parser.add_argument('--end', type=int, default=None, help='last frame to process')
    parser.add_argument('--output', default=None, help='annotated video to write '
                        '(default results/<name>_tracked.mp4)')
    parser.add_argument('--no-video', action='store_true', help="don't write an output video")
    parser.add_argument('--show', action='store_true', help='show frames live (press q to stop)')
    parser.add_argument('--full-rack', action='store_true',
                        help='only start tracking once exactly 1 cue, 1 black, 7 red, 7 yellow are seen')
    parser.add_argument('--detect-only', action='store_true',
                        help='run ball detection only (no tracking) and report the counts per frame')
    return parser.parse_args()


def main():
    args = parse_arguments()

    video = cv2.VideoCapture(args.video)
    if not video.isOpened():
        raise SystemExit(f'Could not open {args.video}')
    fps = video.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.start:
        video.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    # The tracker counts missing *processed* frames, so convert seconds to processed frames.
    sunk_after = max(1, round(SUNK_SECONDS * fps / args.step))
    if hasattr(update_game_state, 'SUNK_AFTER'):
        update_game_state.SUNK_AFTER = sunk_after
    # Don't call a shot finished until a ball pocketed at the very end has had time to be marked sunk.
    rest_frames_needed = max(MIN_REST_FRAMES, sunk_after + 1)

    output_path = None
    if not args.no_video:
        name = os.path.splitext(os.path.basename(args.video))[0]
        suffix = 'detected' if args.detect_only else 'tracked'
        output_path = args.output or os.path.join('results', f'{name}_{suffix}.mp4')
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    print(f'{args.video}: {total_frames} frames at {fps:.1f} fps, processing every {args.step}')
    if args.detect_only:
        print('Detection only: no tracking')
    else:
        print(f'Sunk after {sunk_after} missing frames, shot over after {rest_frames_needed} still frames')
    print('-' * 60)

    stabiliser = CornerStabiliser()
    tracked_balls = None
    writer = None
    last_counts = None
    full_rack_frames = 0

    def output_frame(combined):
        """Write and/or show one annotated frame. Returns False if q was pressed."""
        nonlocal writer
        if output_path:
            if writer is None:
                height, width = combined.shape[:2]
                writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                         fps / args.step, (width, height))
            writer.write(combined)
        # if args.show:
        #     cv2.imshow('tracking', combined)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         return False
        if args.show:
            preview = combined
            max_width = 1600                      # roughly your screen width
            if preview.shape[1] > max_width:
                scale = max_width / preview.shape[1]
                preview = cv2.resize(preview, None, fx=scale, fy=scale,
                                     interpolation=cv2.INTER_AREA)
            cv2.imshow('tracking', preview)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False
        return True
    moving = False
    still_frames = 0
    shot_number = 0
    shot_events = []
    processed = failed = 0
    detection_time = 0.0
    frame_index = args.start - 1

    while True:
        ok, frame = video.read()
        if not ok:
            break
        frame_index += 1
        if args.end is not None and frame_index > args.end:
            break
        if (frame_index - args.start) % args.step != 0:
            continue
        seconds = frame_index / fps

        # --- detection. A failure here skips the frame; tracker bugs below still crash.
        started = time.perf_counter()
        try:
            balls, table_hull, corners, to_photo = detect_frame(frame, stabiliser)
        except Exception as error:
            failed += 1
            print(f'[{seconds:7.2f}s] frame {frame_index}: detection failed: {error!r}')
            continue
        detection_time += time.perf_counter() - started
        processed += 1

        if processed % 50 == 0:
            print(f'  ... frame {frame_index}/{total_frames}, '
                  f'{1000 * detection_time / processed:.0f} ms per frame for detection')

        # --- detection only: report the counts, draw, and move on to the next frame
        if args.detect_only:
            counts = count_colours(balls)
            if counts != last_counts:                 # only print when the counts change
                print(f'[{seconds:7.2f}s] frame {frame_index}: {len(balls)} balls {counts}')
                last_counts = counts
            if counts == FULL_RACK:
                full_rack_frames += 1

            overlay = draw.draw_overlay(frame, table_hull, corners, balls)
            draw_status(overlay, [
                f'frame {frame_index}  {seconds:.1f}s  {len(balls)} balls',
                '  '.join(f'{colour} {counts.get(colour, 0)}' for colour in FULL_RACK),
            ])
            if not output_frame(side_by_side(overlay, balls)):
                break
            continue

        # --- tracker
        events = []
        movement = 0.0
        if tracked_balls is None:
            if args.full_rack and count_colours(balls) != FULL_RACK:
                print(f'[{seconds:7.2f}s] waiting for a full rack, saw {count_colours(balls)}')
                continue
            tracked_balls = start_tracking(balls)
            matches = match_detections(tracked_balls, balls)
            print(f'[{seconds:7.2f}s] started tracking {len(tracked_balls)} balls: '
                  f'{sorted(tracked_balls)}')
        else:
            previous = {ball_id: (info['x'], info['y']) for ball_id, info in tracked_balls.items()}
            matches = match_detections(tracked_balls, balls)
            events = update_states(tracked_balls, matches)

            for ball_id in matches:
                old_x, old_y = previous[ball_id]
                new = tracked_balls[ball_id]
                movement = max(movement, np.hypot(new['x'] - old_x, new['y'] - old_y))

        for event in events:
            print(f'[{seconds:7.2f}s]   {event}')
        if moving:
            shot_events.extend(events)

        # --- shots: balls start moving -> shot starts; everything still for a while -> shot over
        if movement > MOVING_THRESHOLD:
            still_frames = 0
            if not moving:
                moving = True
                shot_number += 1
                shot_events = list(events)
                print(f'[{seconds:7.2f}s] shot {shot_number} started')
        else:
            still_frames += 1
            if moving and still_frames >= rest_frames_needed:
                moving = False
                sunk = [ball_id for kind, ball_id in shot_events if kind == 'sunk']
                print(f'[{seconds:7.2f}s] shot {shot_number} over. sunk: {sunk or "nothing"}')

        # --- annotated frame
        overlay = draw.draw_overlay(frame, table_hull, corners, balls)
        draw_tracking(overlay, tracked_balls, matches, to_photo)
        states = [info['state'] for info in tracked_balls.values()]
        draw_status(overlay, [
            f'frame {frame_index}  {seconds:.1f}s  {"MOVING" if moving else "at rest"}',
            f'on table {states.count(ACTIVE)}  lost {states.count(LOST)}  sunk {states.count(SUNK)}',
        ])
        if not output_frame(side_by_side(overlay, balls)):
            break

    video.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    # --- summary
    print('-' * 60)
    print(f'Processed {processed} frames, {failed} failed detection, '
          f'{stabiliser.rejected} corner jumps ignored')
    if processed:
        print(f'Detection took {1000 * detection_time / processed:.0f} ms per frame on average')
    if args.detect_only:
        if processed:
            print(f'All 16 balls (1 cue, 1 black, 7 red, 7 yellow) found in '
                  f'{full_rack_frames}/{processed} frames')
        if output_path:
            print(f'Saved {output_path}')
        return
    if tracked_balls is None:
        print('Tracking never started.')
        return
    for state in (ACTIVE, LOST, SUNK):
        ids = sorted(i for i, info in tracked_balls.items() if info['state'] == state)
        print(f'{state:>7}: {ids}')
    if output_path:
        print(f'Saved {output_path}')


if __name__ == '__main__':
    main()