"""
Simple ball detector for a video
Process the video frame by frame, detecting balls on each frame.

Find the table once in the first frame, then use this repeatedly.
"""
import cv2
import numpy as np
TABLE_HSV_MIN = (90, 50, 50)  # HSV range for table colour
TABLE_HSV_MAX = (120, 255, 255)

def pick_table_corners(frame, max_display=1200):
    """Manually click the 4 table corners.
    Left-click: add point | Right-click / 'u': undo | 'r': reset
    Enter: accept (needs 4 points) | Esc: cancel
    """
    # Scale down big images so they fit on screen
    scale = min(1.0, max_display / max(frame.shape[:2]))
    display = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame.copy()

    points = []
    win = 'Click 4 table corners'

    def redraw():
        img = display.copy()
        for i, (x, y) in enumerate(points):
            cv2.circle(img, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(img, str(i + 1), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if len(points) > 1:
            cv2.polylines(img, [np.array(points, np.int32)],
                          len(points) == 4, (0, 255, 0), 2)
        msg = 'Enter = accept' if len(points) == 4 else f'{len(points)}/4 points'
        cv2.putText(img, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow(win, img)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()
            redraw()

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10) and len(points) == 4:      # Enter
            break
        elif key == ord('u') and points:
            points.pop(); redraw()
        elif key == ord('r'):
            points.clear(); redraw()
        elif key == 27 or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyWindow(win)                    # Esc or window closed
            raise RuntimeError('Corner selection cancelled')

    cv2.destroyWindow(win)
    pts = np.array(points, dtype=np.float32) / scale   # back to full-res coords
    return order_corners(pts)

def order_corners(pts):
    # Order corners in clockwise order starting from top-left
    pts = pts.astype(np.float32)
    centre = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    pts = pts[np.argsort(angles)]          # clockwise on screen (y points down)
    start = np.argmin(pts.sum(axis=1))     # top-left = smallest x + y
    return np.roll(pts, -start, axis=0)    # TL, TR, BR, BL

def find_table_corners(frame):
    # Convert to HSV and threshold for table colour
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, TABLE_HSV_MIN, TABLE_HSV_MAX)          

    # Remove small noise and fill holes using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours and keep the largest one
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError('No contours found')
    contour = max(contours, key=cv2.contourArea) # assume the table is the largest contour
    hull = cv2.convexHull(contour)

    # Approximate the contour to a polygon and get its corners
    for eps in np.linspace(0.01, 0.1, 10):
        approx = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
        if len(approx) == 4:
            break
    
    corners = order_corners(approx.reshape((4, 2)))
    return corners

def table_transform(corners, rail=40):
    # Determine the transform for normalised perspective view of the table
    w, h = 800, 400
    corners = np.roll(corners, -1, axis=0)
    dst = np.array([[-rail, -rail], [w + rail, -rail],
                    [w + rail, h + rail], [-rail, h + rail]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)   # Compute the perspective transform matrix
    return M

def find_balls(frame, M):
    # Apply the perspective transform to get a top-down view of the table
    warped = cv2.warpPerspective(frame, M, (800, 400))

    # Convert to HSV and threshold out blue table to find anything that is not blue
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, TABLE_HSV_MIN, TABLE_HSV_MAX)
    mask = cv2.bitwise_not(mask)  # Invert the mask to find everything that is not blue

    # Find contours of the balls
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    balls = []

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area < 50 or perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
            
        if circularity > 0.5:  # filter out non-circular contours
            (x, y), radius = cv2.minEnclosingCircle(contour)
            balls.append((int(x), int(y), int(radius)))

    return balls, warped

# Typical BGR colour of each ball type. Sample these from your own footage.
REF_COLOURS = {
    'red':    (40, 30, 200),
    'yellow': (30, 160, 230),
    'white':  (220, 225, 225),
    'black':  (30, 30, 30),
}

# TO DO: Add a more robust classification method that can handle variations in lighting (also should i use HSV?)
def classify_balls(balls, warped):
    # use BGR colour space to classify balls based on their colour, since HSV wraps around for red
    classified = []
    for (x, y, r) in balls:
        rr = max(1, r // 2)   # inner half of the ball only
        patch = warped[max(0, y - rr):y + rr + 1, max(0, x - rr):x + rr + 1]
        colour = np.median(patch.reshape(-1, 3), axis=0)
        label = min(REF_COLOURS, key=lambda k: np.linalg.norm(colour - REF_COLOURS[k]))
        classified.append((x, y, r, label))
    return classified

# REMOVE THIS LATER: This function is just for visualisation and debugging
def check_table(frame, M, w=800, h=400):
    # Project the warp's edges back onto the original image
    outline = np.float32([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]])
    outline = cv2.perspectiveTransform(outline, np.linalg.inv(M))
    overlay = frame.copy()
    cv2.polylines(overlay, [outline.astype(np.int32)], True, (0, 0, 255), 3)

    # Mark where the six pockets should be in the top-down view
    warped = cv2.warpPerspective(frame, M, (w, h))
    for x, y in [(0, 0), (w // 2, 0), (w, 0), (0, h), (w // 2, h), (w, h)]:
        cv2.circle(warped, (x, y), 15, (0, 0, 255), 2)

    return overlay, warped

def main():
    img = cv2.imread(r'Table images/839999399_2303128927107538_4571142254927940165_n.jpg')      # horizontal table
    # img = cv2.imread(r'Table images/840599825_1837650447578857_2846709159424431200_n.jpg')    # diagonal table view
    corners = pick_table_corners(img)
    M = table_transform(corners)
    balls, warped = find_balls(img, M)
    # overlay, warped = check_table(img, M)
    # cv2.imshow('Overlay', overlay)
    # cv2.imshow('Warped', warped)
    # cv2.waitKey(0)
    print(f"Detected {len(balls)} balls")
    classified = classify_balls(balls, warped)
    for (x, y, radius, color) in classified:
        print(f"Ball at ({x}, {y}) with radius {radius} classified as {color}")

if __name__ == '__main__':
    main()
    