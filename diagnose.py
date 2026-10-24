"""
Diagnostic script - saves frames with debug overlays to diagnose detection issues.
Saves every Nth frame with state info overlaid.
"""

import cv2
import os
import numpy as np
import table_mask
import board_reading
import config
from shot_detection import shrink_to_gray, moving_pixels, view_is_blocked

def diagnose_video(video_path, output_dir="diagnostics", save_every=30, max_frames=300):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_path}")
    print(f"FPS: {fps}, Total frames: {total_frames}")

    # Read first frame for calibration
    ok, first_frame = cap.read()
    if not ok:
        print("ERROR: Cannot read first frame")
        return

    print(f"Frame size: {first_frame.shape}")

    # Find table
    try:
        click_x, click_y = table_mask.find_cloth_center(first_frame)
        print(f"Cloth center: ({click_x}, {click_y})")
        reference = board_reading.make_reference(first_frame, click_x, click_y)
        print(f"Table found with {len(reference['corners'])} corners")
        print(f"Table area (small): {reference['table_area_small']}")
    except Exception as e:
        print(f"ERROR finding table: {e}")
        return

    # Process frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    previous_gray = None
    frame_idx = 0
    saved = 0

    stats = {'settled': 0, 'shot': 0, 'blocked': 0, 'reading': 0}

    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        gray = shrink_to_gray(frame)
        moving = moving_pixels(gray, previous_gray, reference)
        blocked = view_is_blocked(frame, reference)
        previous_gray = gray

        # Determine state
        if blocked:
            state = "BLOCKED"
            stats['blocked'] += 1
        elif moving > config.MOVE_HIGH_FRACTION:
            state = "SHOT"
            stats['shot'] += 1
        elif moving < config.MOVE_LOW_FRACTION:
            state = "SETTLED"
            stats['settled'] += 1
        else:
            state = "READING"
            stats['reading'] += 1

        # Save diagnostic frame
        if frame_idx % save_every == 0:
            # Draw debug info on frame
            diag_frame = frame.copy()

            # Draw table outline
            cv2.drawContours(diag_frame, [reference['table_hull']], -1, (0, 255, 0), 2)

            # Draw corners
            for cx, cy in reference['corners']:
                cv2.circle(diag_frame, (int(cx), int(cy)), 10, (0, 0, 255), -1)

            # Try to detect balls
            try:
                balls = board_reading.read_board(frame, reference, click_x, click_y)
                ball_count = len(balls)
                for ball in balls:
                    bx, by = int(ball['photo_x']), int(ball['photo_y'])
                    cv2.circle(diag_frame, (bx, by), int(ball['radius']), (255, 255, 0), 2)
            except Exception as e:
                ball_count = f"ERR: {e}"

            # Add text overlay
            info_lines = [
                f"Frame: {frame_idx} / {total_frames}",
                f"State: {state}",
                f"Moving: {moving:.6f}",
                f"Blocked: {blocked}",
                f"Balls: {ball_count}",
                f"MOVE_HIGH: {config.MOVE_HIGH_FRACTION}",
                f"MOVE_LOW: {config.MOVE_LOW_FRACTION}",
            ]

            y = 30
            for line in info_lines:
                cv2.putText(diag_frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 0, 0), 3)
                cv2.putText(diag_frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 255, 255), 2)
                y += 25

            # Save frame
            out_path = os.path.join(output_dir, f"frame_{frame_idx:05d}_{state}.jpg")
            cv2.imwrite(out_path, diag_frame)
            saved += 1
            print(f"Saved {out_path}")

        frame_idx += 1

    cap.release()

    print(f"\n=== SUMMARY ===")
    print(f"Frames processed: {frame_idx}")
    print(f"Frames saved: {saved}")
    print(f"State counts: {stats}")
    print(f"Diagnostics saved to: {output_dir}/")

if __name__ == "__main__":
    import sys
    video = sys.argv[1] if len(sys.argv) > 1 else "videos/IMG_0711.mp4"
    diagnose_video(video)
