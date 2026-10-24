"""
step 4: find the balls on the table (blob detector). 

- size tests. top-down map tells us how big a ball should look at each spot, thus blobs that are too small or thin are dropped. 
- balls resting against a cushion. they overlap the edge of the cloth, showing up as bites out of the outline instead of holes 
in it, red and yellow are trusted here as holes are dark too
- Touching balls. Merged blob is plit by colour first, then by counting. 

1. Take candidate blobs = anything inside table that it not cloth-coloured, filtered by the expected
ball size
2. Inside each blob, make one mask per colour family (red, yellow, dark/white), black and cue share a blob 
as the black ball carries a bright highlight. A closing fills the highlight hole 
3. Ball centers are the peaks of the distance transform of each colour mask. The shadow seam between two touching balls is
not that colour, thus same-colour neighbours seperate. 
4. A ball that is mostly hidden behind another shows only a sliver. Its peak is weak and sits in the middle of the silver, 
so a center is pushed inward, away from the cloth. 
5. The same ball must not be counted twice, two centers closer than 1.5 radii on the map are one ball (real balls are noted as 2 radii apart on the table),
and two weak peaks close together in the photo are one sliver. 
6. The colour is the family the mask came from: the dark/white family is split into cue (mostly white), and black (mostly dark).


To include: subtracting a photo of the empty table (the backdrop). That is the proper 
fix for cue and black balls on a cushion, and needs a fixed camera. 
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans

import config
import ball_colour
from top_down import move_point, ball_radius_in_photo

def is_at_pocket_mouth(map_x, map_y):
    """
    True if a map position is in the mouth of one of the six pockets (where the opening is dark).
    """
    width = config.MAP_WIDTH
    height = config.MAP_HEIGHT
    near_a_pocket_column = min(map_x, width - map_x, abs(map_x - width / 2)) < 0.08 * width
    near_the_long_edge = min(map_y, height - map_y) < 0.12 * height
    return near_a_pocket_column and near_the_long_edge
    
def make_ball(colour, photo_x, photo_y, radius, to_map, visible):
    """
    Builds the record for one ball.
    """
    map_x, map_y = move_point(to_map, photo_x, photo_y)
    return {'colour': colour, 'photo_x': float(photo_x), 'photo_y': float(photo_y),
            'radius': float(radius), 'map_x': float(map_x), 'map_y': float(map_y),
            'visible': float(visible)}

def colour_at(image_hsv, centre_x, centre_y, radius, family):
    """
    Checks the colour in a disc round a candidate centre.
    inputs:
    - image_hsv: the frame in HSV
    - centre_x, centre_y, radius: the disc (about half the ball radius)
    - family: the mask the centre came from ('red', 'yellow', 'darkwhite')
    Output:
    - 'red', 'yellow', 'cue', 'black', or None if the disc does not hold enough colour

    For darkwhite: a shadow on the cloth is dark but keeps the cloth's saturation (~190);
    the black ball has a grey core (S < 60). Measured colourless share of dark pixels:
    0.28-0.76 on the black ball, 0.00 on every shadow.
    """
    image_height, image_width = image_hsv.shape[:2]
    y0 = max(0, int(centre_y - radius))
    y1 = min(image_height, int(centre_y + radius) + 1)
    x0 = max(0, int(centre_x - radius))
    x1 = min(image_width, int(centre_x + radius) + 1)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    inside = (xs - centre_x) ** 2 + (ys - centre_y) ** 2 <= radius * radius

    pixels = image_hsv[ys[inside], xs[inside]]
    labels = ball_colour.label_pixels(pixels)
    white_fraction = np.mean(labels == ball_colour.WHITE)
    dark_fraction = np.mean(labels == ball_colour.DARK)
    red_fraction = np.mean(labels == ball_colour.RED)
    yellow_fraction = np.mean(labels == ball_colour.YELLOW)

    if family == 'red':
        return 'red' if red_fraction > config.MIN_COLOUR_FRACTION else None
    if family == 'yellow':
        return 'yellow' if yellow_fraction > config.MIN_COLOUR_FRACTION else None

    # darkwhite branch: distinguish cue ball, black ball, and shadows
    if white_fraction + dark_fraction < config.MIN_COLOUR_FRACTION:
        return None
    if dark_fraction < config.BLACK_MIN_DARK_FRACTION:
        return 'cue'                                      # bright, no dark pixels
    dark = labels == ball_colour.DARK
    colourless = dark & (pixels[:, 1] < config.BLACK_MAX_SATURATION)
    colourless_share = np.sum(colourless) / max(1, np.sum(dark))
    if colourless_share < config.BLACK_MIN_COLOURLESS_SHARE:
        return None                                       # dark but still cloth-coloured: a shadow
    return 'black'

def find_centres(colour_mask, radius, to_map, offset_x, offset_y):
    """
    Finds the ball centers in one colour mask. 
    Inputs:
    - colour_mask: 0/1 image of one colour inside one blob (crop of the frame)
    - radius: expected ball radius here, in photo pixels 
    - to_map: photo -> map matrix 
    - offset_x, offset_y: where the crop sits in the full frame 
    Output: 
    - List of (x, y, peak_height) in full-frame pixels. peak_height is how far the centre is from the edge 
    of its colour patch. About the radius of a whole ball, less for a sliver. 

    distanceTransform used here to give every pixel its distance to the edge of the patch. A ball is a round 
    patch, thus its centre is a peak of that distance. Peaks are taken strongest first, a weaker peak is thrown away if it
    belongs to a ball already found. 
    """
    distance = cv2.distanceTransform(colour_mask, cv2.DIST_L2, 5)
    if distance.max() < config.PEAK_MIN_RATIO * radius:
        return []

    # A pixel is a peak if it equals the largest value in the 5 x 5 square round it.
    neighbourhood_max = cv2.dilate(distance, np.ones((5, 5), np.uint8))
    is_peak = (distance == neighbourhood_max) & (distance > config.PEAK_MIN_RATIO * radius)
    peak_ys, peak_xs = np.nonzero(is_peak)
    strongest_first = np.argsort(-distance[peak_ys, peak_xs])

    kept = []           # each entry: (x, y, height, map_x, map_y)
    for index in strongest_first:
        x = int(peak_xs[index]) + offset_x
        y = int(peak_ys[index]) + offset_y
        height = float(distance[peak_ys[index], peak_xs[index]])
        map_x, map_y = move_point(to_map, x, y)

        same_ball = False
        for kept_x, kept_y, kept_height, kept_map_x, kept_map_y in kept:
            map_distance = np.hypot(map_x - kept_map_x, map_y - kept_map_y)
            if map_distance < config.SAME_BALL_ON_MAP * config.BALL_RADIUS_ON_MAP:
                same_ball = True
                break
            both_slivers = height < config.WEAK_PEAK_RATIO * radius and kept_height < config.WEAK_PEAK_RATIO * radius
            if both_slivers and np.hypot(x - kept_x, y - kept_y) < config.SAME_SLIVER_IN_PHOTO * radius:
                same_ball = True                      # a thin sliver gives two peaks along its ridge
                break
        if not same_ball:
            kept.append((x, y, height, map_x, map_y))

    return [(x, y, height) for x, y, height, _, _ in kept]


def shift_sliver_inward(x, y, peak_height, radius, cloth_mask):
    """
    Moves the center of a sliver to where the ball centre really is. 

    The peak of a sliver sits in the middle of the visible part, peak_height away from the 
    cloth on the outside. ball centre is (radius - peak_height) further in, straight away from 
    the nearest cloth pixel. Whole balls, and slivers boxed in by other balls, are left alone. 
    """
    if peak_height >= config.WEAK_PEAK_RATIO * radius:
        return x, y

    reach = int(1.5 * radius)
    y0, y1 = max(0, y - reach), min(cloth_mask.shape[0], y + reach + 1)
    x0, x1 = max(0, x - reach), min(cloth_mask.shape[1], x + reach + 1)
    cloth_ys, cloth_xs = np.nonzero(cloth_mask[y0:y1, x0:x1])
    if len(cloth_ys) == 0:
        return x, y

    distances = np.hypot(cloth_xs + x0 - x, cloth_ys + y0 - y)
    nearest = np.argmin(distances)
    if distances[nearest] > reach:
        return x, y

    away_x = x - (cloth_xs[nearest] + x0)
    away_y = y - (cloth_ys[nearest] + y0)
    length = np.hypot(away_x, away_y)
    if length < 1:
        return x, y
    step = radius - peak_height
    return int(round(x + step * away_x / length)), int(round(y + step * away_y / length))


def find_balls(image_hsv, cloth_mask, table_contour, table_hull, to_map, to_photo):
    """
    Inputs
        image_hsv, cloth_mask : from table_mask
        table_contour, table_hull : from find_table_outline
        to_map, to_photo : from make_map_transform
    Output
        balls : a list of records, one per ball, each with
                colour, photo_x, photo_y, radius, map_x, map_y, visible
    """
    image_height, image_width = cloth_mask.shape

    # Two filled shapes: the exact cloth outline, and the hull stretched round it.
    cloth_region = np.zeros((image_height, image_width), dtype=np.uint8)
    cv2.drawContours(cloth_region, [table_contour], -1, 255, -1)
    hull_region = np.zeros((image_height, image_width), dtype=np.uint8)
    cv2.fillConvexPoly(hull_region, table_hull, 255)

    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # Holes: inside the cloth outline but not cloth-coloured. These are balls on the bed.
    holes = cv2.bitwise_and(cloth_region, cv2.bitwise_not(cloth_mask))
    holes = cv2.morphologyEx(holes, cv2.MORPH_OPEN, small_kernel)
    # Bites: inside the hull but outside the cloth outline. Pockets, rails, and balls on a cushion.
    bites = cv2.bitwise_and(hull_region, cv2.bitwise_not(cloth_region))
    bites = cv2.morphologyEx(bites, cv2.MORPH_OPEN, small_kernel)

    balls = []
    for mask, at_edge in ((holes, False), (bites, True)):
        blob_count, blob_image, blob_stats, blob_centres = cv2.connectedComponentsWithStats(mask)

        for blob in range(1, blob_count):
            left, top, width, height, area = blob_stats[blob]
            if area < 40:
                continue

            centre_x, centre_y = blob_centres[blob]
            expected_radius = ball_radius_in_photo(to_map, to_photo, centre_x, centre_y)
            expected_area = np.pi * expected_radius * expected_radius
            if area < config.MIN_AREA_RATIO * expected_area:
                continue                                  # too small
            if at_edge and area > config.MAX_EDGE_BLOB_AREA_RATIO * expected_area:
                continue                                  # a pocket or a length of rail

            blob_mask = (blob_image == blob).astype(np.uint8)
            # thickness = cv2.distanceTransform(blob_mask, cv2.DIST_L2, 5).max()
            # if thickness < config.MIN_THICKNESS_RATIO * expected_radius:
            #     continue                                  # too thin: cue stick, seam or rail sliver

            # Work on a crop round the blob (a bit bigger than the blob) to keep it quick.
            pad = int(expected_radius) + 2
            y0, y1 = max(0, top - pad), min(image_height, top + height + pad)
            x0, x1 = max(0, left - pad), min(image_width, left + width + pad)
            blob_crop = blob_mask[y0:y1, x0:x1]

            thickness = cv2.distanceTransform(blob_crop, cv2.DIST_L2, 5).max()
            if thickness < config.MIN_THICKNESS_RATIO * expected_radius:
                continue                                  # too thin: cue stick, seam or rail sliver

            crop_pixels = image_hsv[y0:y1, x0:x1].reshape(-1, 3)
            labels = ball_colour.label_pixels(crop_pixels).reshape(blob_crop.shape)

            kernel_size = max(3, int(config.CLOSE_KERNEL_RATIO * expected_radius)) | 1
            close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

            families = [('red', labels == ball_colour.RED),
                        ('yellow', labels == ball_colour.YELLOW),
                        ('darkwhite', (labels == ball_colour.DARK) | (labels == ball_colour.WHITE))]
            if at_edge:
                families = families[:2]                   # pockets and rails are dark too

            for family, family_pixels in families:
                colour_mask = (family_pixels & (blob_crop > 0)).astype(np.uint8)
                colour_mask = cv2.morphologyEx(colour_mask, cv2.MORPH_CLOSE, close_kernel)

                for x, y, peak_height in find_centres(colour_mask, expected_radius, to_map, x0, y0):
                    # The colour is checked at the peak, before any shift, so a hidden ball is
                    # not judged on the pixels of the ball in front of it.
                    colour = colour_at(image_hsv, x, y, 0.55 * expected_radius, family)
                    if colour is None:
                        continue
                    x, y = shift_sliver_inward(x, y, peak_height, expected_radius, cloth_mask)
                    map_x, map_y = move_point(to_map, x, y)

                    on_the_map = -0.03 * config.MAP_WIDTH < map_x < 1.03 * config.MAP_WIDTH \
                        and -0.06 * config.MAP_HEIGHT < map_y < 1.06 * config.MAP_HEIGHT
                    if not on_the_map:
                        continue
                    if colour == 'black' and is_at_pocket_mouth(map_x, map_y):
                        continue                          # a pocket opening, not the black ball
                    balls.append(make_ball(colour, x, y, expected_radius, to_map, peak_height / expected_radius))

    # Last tidy-up across blobs: the same colour closer than 1.5 radii on the map, or any
    # colour closer than half a radius, is one ball. The first one found is kept.
    tidy = []
    for ball in balls:
        duplicate = False
        for other in tidy:
            distance = np.hypot(ball['map_x'] - other['map_x'], ball['map_y'] - other['map_y'])
            same_colour = ball['colour'] == other['colour']
            if distance < 0.5 * config.BALL_RADIUS_ON_MAP \
                    or (same_colour and distance < config.SAME_BALL_ON_MAP * config.BALL_RADIUS_ON_MAP):
                duplicate = True
                break
        if not duplicate:
            tidy.append(ball)
    return tidy
