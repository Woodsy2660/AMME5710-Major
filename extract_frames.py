"""
Extract frames from video for Claude to analyze.
Saves frames as images that can be read and viewed.
"""
import cv2
import os
import sys

def extract_frames(video_path, output_dir="frames", frame_indices=None, interval=30):
    """
    Extract specific frames or every Nth frame from a video.

    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        frame_indices: List of specific frame numbers, or None for interval-based
        interval: Extract every Nth frame (if frame_indices is None)
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_path}")
    print(f"FPS: {fps}, Total frames: {total}, Duration: {total/fps:.1f}s")

    if frame_indices is None:
        # Extract every Nth frame
        frame_indices = list(range(0, min(total, 900), interval))  # Cap at 900 frames

    saved = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            time_sec = idx / fps
            filename = f"frame_{idx:05d}_t{time_sec:.1f}s.jpg"
            path = os.path.join(output_dir, filename)
            cv2.imwrite(path, frame)
            saved.append(path)
            print(f"Saved: {path}")

    cap.release()
    print(f"\nExtracted {len(saved)} frames to {output_dir}/")
    return saved

if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "videos/IMG_0711.mp4"

    # Extract frames at: start, 2s, 5s, 10s, 20s, 30s, 60s
    # Adjust based on your video length
    extract_frames(video, output_dir="frames", interval=60)
