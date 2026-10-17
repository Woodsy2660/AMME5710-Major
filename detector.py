"""
Simple pool ball detector for a fixed camera.

  python pool_simple.py video.mov [--out out_dir] [--corners x1 y1 x2 y2 x3 y3 x4 y4]

Once: table outline and 4 corners from the first frame (nobody over the table) -> top-down map.
Per frame: non-cloth pixels inside the table, split into red / yellow / white / dark masks;
ball centres = distance-transform peaks. A board reading is taken once per window in which the
view is clear and every ball has stayed put for HOLD seconds.
Outputs in --out: annotated.mp4, reading_NN.jpg, readings.json, signals.csv.
"""
import argparse, json, os
from collections import deque
import cv2
import numpy as np

CLOTH_HUE = (90, 115)   # blue cloth hue in OpenCV HSV (0-180)
TABLE_SAT = 120         # strict saturation floor for finding the table outline (chrome trim is bluish too)
CLOTH_SAT = 90          # lenient floor inside the table (worn cloth patches drop to ~120)
MIN_R = 6               # px: colour patches thinner than this are not balls (cue stick, seams)
SAME_PX = 4             # px: a ball this close to where it was HOLD seconds ago has not moved
OCCLUDED_FRAC = 0.06    # non-cloth fraction of the table above which something is over it
EDGE_LUMP_FRAC = 0.003  # a non-cloth lump touching the table edge bigger than this is a hand/arm
                        # (hand on the rail measured 0.004-0.006; a far ball against a cushion ~0.002)
HOLD = 0.5              # seconds still before a reading is taken
MAP_W, MAP_H = 800, 400 # top-down map, long side horizontal
EDGE_MARGIN = 20        # map px: centres closer than this to the table edge are rail/pocket, not balls
COLOURS = {'cue': (235, 235, 235), 'black': (30, 30, 30), 'red': (40, 40, 230), 'yellow': (0, 200, 255)}


def cloth_mask(hsv, min_sat=CLOTH_SAT):
    mask = cv2.inRange(hsv, (CLOTH_HUE[0], min_sat, 60), (CLOTH_HUE[1], 255, 255))  # value >= 60 keeps shadows
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def touching(mask, ring):
    """Connected components of mask that touch the ring, as (component image, labels touching, stats)."""
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    return labels, set(np.unique(labels[(ring > 0) & (labels > 0)])), stats


def find_table(frame):
    """Table region (cloth hull minus the dark pocket mouths), its inner edge ring, and 4 corners.
    Run once on a frame with nobody over the table."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    cloth = cloth_mask(hsv, TABLE_SAT)
    contours, _ = cv2.findContours(cloth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))  # assumes our table is the biggest blue thing
    region = np.zeros(cloth.shape, np.uint8)
    cv2.fillConvexPoly(region, hull, 255)
    region = cv2.erode(region, np.ones((5, 5), np.uint8))
    ring = region - cv2.erode(region, np.ones((9, 9), np.uint8))

    # Pocket mouths: dark, not cloth, and cutting in from the edge. Removed from the region for good.
    dark = (region > 0) & (cloth == 0) & (hsv[..., 2] < 90)
    labels, pockets, _ = touching(dark, ring)
    pocket_mask = cv2.dilate(np.isin(labels, list(pockets)).astype(np.uint8) * 255, np.ones((15, 15), np.uint8))
    region[pocket_mask > 0] = 0
    ring = region - cv2.erode(region, np.ones((9, 9), np.uint8))

    corners = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True).reshape(-1, 2)
    if len(corners) != 4:
        raise SystemExit('found %d corners, pass --corners' % len(corners))
    return region, ring, corners.astype(np.float32)


def is_occluded(frame, region, ring, area):
    """Something other than balls over the table: lots of non-cloth, or a big lump coming in from the edge."""
    hsv = cv2.cvtColor(cv2.resize(frame, None, fx=0.5, fy=0.5), cv2.COLOR_BGR2HSV)
    small_region, small_ring = region[::2, ::2], ring[::2, ::2]
    not_cloth = (small_region > 0) & (cloth_mask(hsv) == 0)
    if np.count_nonzero(not_cloth) / (area / 4) > OCCLUDED_FRAC:
        return True
    _, edge_lumps, stats = touching(not_cloth, small_ring)
    return any(stats[i, cv2.CC_STAT_AREA] / (area / 4) > EDGE_LUMP_FRAC for i in edge_lumps)


def map_transform(corners):
    """Homography to the top-down map. Starts the corner order on the longest image edge."""
    edges = [np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]
    corners = np.roll(corners, -int(np.argmax(edges)), axis=0)
    target = np.float32([[0, 0], [MAP_W, 0], [MAP_W, MAP_H], [0, MAP_H]])
    return cv2.getPerspectiveTransform(corners, target)


def find_balls(frame, region, H):
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    not_cloth = (region > 0) & (cloth_mask(hsv) == 0)
    red = not_cloth & ((hue >= 165) | (hue <= 8)) & (sat >= 120)
    yellow = not_cloth & (hue >= 12) & (hue <= 35) & (sat >= 120)
    other = not_cloth & ~red & ~yellow
    cue = other & (val >= 150)                           # white: bright and unsaturated
    black = other & (val < 150)                          # its small white highlight is filled by the closing

    balls = []
    for colour, mask in (('red', red), ('yellow', yellow), ('cue', cue), ('black', black)):
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        peaks = (dist == cv2.dilate(dist, np.ones((5, 5), np.uint8))) & (dist >= MIN_R)
        ys, xs = np.nonzero(peaks)
        kept = []
        for i in np.argsort(-dist[ys, xs]):              # strongest first; drop peaks inside a kept ball
            x, y, r = int(xs[i]), int(ys[i]), float(dist[ys[i], xs[i]])
            if all(np.hypot(x - kx, y - ky) > 1.5 * kr for kx, ky, kr in kept):
                kept.append((x, y, r))
        for x, y, r in kept:
            mx, my = cv2.perspectiveTransform(np.float32([[[x, y]]]), H)[0, 0]
            if not (EDGE_MARGIN < mx < MAP_W - EDGE_MARGIN and EDGE_MARGIN < my < MAP_H - EDGE_MARGIN):
                continue                                 # on the rail or in a pocket jaw (a ball on the cushion is ~30 in)
            balls.append({'colour': colour, 'x': x, 'y': y, 'r': round(r, 1),
                          'map_x': round(float(mx), 1), 'map_y': round(float(my), 1)})
    return balls


def counts(balls):
    return {c: sum(b['colour'] == c for b in balls) for c in COLOURS}


def same_board(a, b):
    """Same balls in the same places: equal counts, and every ball has a same-colour partner within SAME_PX."""
    return counts(a) == counts(b) and all(
        any(o['colour'] == p['colour'] and np.hypot(o['x'] - p['x'], o['y'] - p['y']) <= SAME_PX for o in b)
        for p in a)


def draw(frame, balls, corners, status):
    out = frame.copy()
    cv2.polylines(out, [corners.astype(np.int32)], True, (0, 255, 0), 2)
    for b in balls:
        cv2.circle(out, (b['x'], b['y']), int(b['r']) + 4, (0, 0, 0), 4)
        cv2.circle(out, (b['x'], b['y']), int(b['r']) + 4, COLOURS[b['colour']], 2)
    top = np.full((MAP_H + 40, MAP_W + 40, 3), (60, 60, 60), np.uint8)
    cv2.rectangle(top, (20, 20), (MAP_W + 20, MAP_H + 20), (200, 120, 30), -1)
    for b in balls:
        cv2.circle(top, (int(b['map_x']) + 20, int(b['map_y']) + 20), 10, COLOURS[b['colour']], -1)
    top = cv2.resize(top, (420, 220))
    out[10:230, out.shape[1] - 430:out.shape[1] - 10] = top
    cv2.putText(out, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6)
    cv2.putText(out, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video')
    ap.add_argument('--out', default='out')
    ap.add_argument('--corners', nargs=8, type=float)
    ap.add_argument('--step', type=int, default=1, help='detect on every Nth frame only')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    ok, frame = cap.read()
    region, ring, corners = find_table(frame)
    if args.corners:
        corners = np.float32(args.corners).reshape(4, 2)
    H = map_transform(corners)
    area = np.count_nonzero(region)
    writer = cv2.VideoWriter(os.path.join(args.out, 'annotated.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                             fps, (frame.shape[1], frame.shape[0]))

    hold = max(1, round(HOLD * fps / args.step))
    history = deque(maxlen=hold + 1)                     # ball lists of the last HOLD seconds (None = occluded)
    read_this_window, readings, signals = False, [], []
    n = 0
    while ok:
        balls = find_balls(frame, region, H)
        c = counts(balls)                                # a hand or cue reads as extra white/black "balls"
        occluded = is_occluded(frame, region, ring, area) or c['cue'] > 1 or c['black'] > 1 \
            or c['red'] > 7 or c['yellow'] > 7
        history.append(None if occluded else balls)
        # Still = clear for HOLD seconds and every ball where it was HOLD seconds ago.
        # (Comparing across the whole window catches a slow roller that moves ~1 px per frame.)
        readable = len(history) == hold + 1 and all(h is not None for h in history) \
            and same_board(history[0], balls)

        if not readable:
            read_this_window = False
        elif not read_this_window:                       # one reading per still window
            read_this_window = True
            readings.append({'frame': n, 'time': round(n / fps, 2), 'counts': counts(balls), 'balls': balls})
            print('t=%5.1fs  %s' % (n / fps, counts(balls)))
            cv2.imwrite(os.path.join(args.out, 'reading_%02d.jpg' % len(readings)), draw(frame, balls, corners, ''))

        state = 'READING' if readable else 'occluded' if occluded else 'moving'
        signals.append((n, occluded, readable, len(balls)))
        writer.write(draw(frame, balls, corners, 't=%.1fs %s  %s' % (n / fps, state, counts(balls))))
        for _ in range(args.step - 1):
            cap.grab()                  # skip a frame without converting it to an image
        ok, frame = cap.read()
        n += args.step

    writer.release()
    json.dump(readings, open(os.path.join(args.out, 'readings.json'), 'w'), indent=1)
    np.savetxt(os.path.join(args.out, 'signals.csv'), np.array(signals, dtype=float), delimiter=',',
               header='frame,occluded,readable,n_balls', fmt='%g')
    print('%d frames, %d readings' % (n, len(readings)))


if __name__ == '__main__':
    main()