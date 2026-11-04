"""
Find which camera number is the iPhone (Camo / Iriun / Continuity Camera).

    python find_camera.py

Tries camera numbers 0-4, asks each for 1920x1080, prints the size it actually gives,
and saves a picture camera_<n>.jpg. The picture showing the pool table is the iPhone:
use that number with gui.py --camera <n>.
Keep Camo Studio open while running this (and gui.py).
"""

import os
import time

import cv2

backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY

for i in range(5):
    cam = cv2.VideoCapture(i, backend)
    if not cam.isOpened():
        continue
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    time.sleep(1)
    ok, frame = False, None
    for _ in range(10):  # the first frames are often black while the camera starts
        ok, frame = cam.read()
    if ok:
        cv2.imwrite(f'camera_{i}.jpg', frame)
        print(f'camera {i}: {frame.shape[1]}x{frame.shape[0]}  saved camera_{i}.jpg')
    else:
        print(f'camera {i}: opened but gave no picture')
    cam.release()
