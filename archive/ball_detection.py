"""
step 4: find the balls on the table (blob detector). 

- size tests. top-down map tells us how big a ball should look at each spot, thus blobs that are too small or thin are dropped. 
- balls resting against a cushion. they overlap the edge of the cloth, showing up as bites out of the outline instead of holes 
in it, red and yellow are trusted here as holes are dark too
- Touching balls. Merged blob is plit by colour first, then by counting. 


To include: subtracting a photo of the empty table (the backdrop). That is the proper 
fix for cue and black balls on a cushion, and needs a fixed camera. 
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans

import config
import archive.ball_colour as ball_colour
from archive.top_down import move_point, ball_radius_in_photo

def is_near_pocket(map_x, map_y):
    """
    True if a map position is close to one of the six pockets.
    """

    width = config.MAP_WIDTH
    height = config.MAP_HEIGHT
    pockets = [(0, 0), (width / 2, 0), (width, 0), (0, height), (width / 2, height), (width, height)]

    nearest = None
    for pocket_x, pocket_y in pockets:
        distance = np.hypot(map_x - pocket_x, map_y - pocket_y)
        if nearest is None or distance < nearest:
            nearest = distance

    close_to_a_pocket = nearest < 0.24 * height
    close_to_the_edge = map_y < 0.2 * height or map_y > 0.8 * height \
        or map_x < 0.1 * width or map_x > 0.9 * width
    return close_to_a_pocket and close_to_the_edge

def make_ball(colour, xs, ys, to_map):
    """
    builds the record for one ball from the pixels that belong to it.
    """
    photo_x = float(np.mean(xs))
    photo_y = float(np.mean(ys))
    map_x, map_y = move_point(to_map, photo_x, photo_y)
    # Radius of a circle with the same area as the pixel group: area = pi * r^2.
    radius = float(np.sqrt(len(xs) / np.pi))
    return {'colour': colour, 'photo_x': photo_x, 'photo_y': photo_y,
            'radius': radius, 'map_x': float(map_x), 'map_y': float(map_y)}

def split_blob_by_colour(image_hsv, xs, ys, expected_radius, at_edge, blob_near_pocket, to_map):
    """
    Splits one merged or odd-shapped blob into balls.

    Inputs
    - image_hsv : the photo in HSV
    - xs, ys : pixel coordinates of the blob
    - expected_radius : how big a ball should look here, in photo pixels
    - at_edge : True if this blob is a bite out of the cloth outline
    - blob_near_pocket : True if the blob sits at a 
    Output 
    - a list of ball records 
    """

    balls = []
    expected_area = np.pi * expected_radius * expected_radius
    labels = ball_colour.label_pixels(image_hsv[ys, xs])

    if at_edge:
        colours_to_try = [ball_colour.RED, ball_colour.YELLOW]
    else:
        colours_to_try = [ball_colour.WHITE, ball_colour.DARK, ball_colour.RED, ball_colour.YELLOW]

    for colour_label in colours_to_try:
        # Draw only this colour's pixels, then find the separate groups of them.
        colour_mask = np.zeros(image_hsv.shape[:2], dtype=np.uint8)
        colour_mask[ys[labels == colour_label], xs[labels == colour_label]] = 1
        group_count, group_image, group_stats, _ = cv2.connectedComponentsWithStats(colour_mask)

        for group in range(1, group_count):
            area = group_stats[group][4]
            if area < 0.4 * expected_area:
                continue                                  # a highlight or a shadow, not a ball

            group_mask = (group_image == group).astype(np.uint8)

            # distanceTransform gives, for every pixel, its distance to the edge of the group.
            # Its largest value is the radius of the biggest circle that fits inside.
            distance_to_edge = cv2.distanceTransform(group_mask, cv2.DIST_L2, 5)
            if distance_to_edge.max() < 0.5 * expected_radius:
                continue                                  # too thin to be a ball

            if colour_label == ball_colour.DARK and blob_near_pocket:
                continue                                  # dark at a pocket is the pocket
            if colour_label == ball_colour.WHITE and area > config.MAX_SINGLE_BALL_AREA_RATIO * expected_area:
                continue                                  # there is only one cue ball

            # How many balls are in this group? Two estimates, take the larger.
            #   1. By area: total area divided by the area of one ball.
            #   2. By centres: keep only pixels deep inside the group. Each ball leaves
            #      its own island, so count the islands.
            count_by_area = int(round(area / (0.92 * expected_area)))
            deep_inside = (distance_to_edge > 0.6 * expected_radius).astype(np.uint8)
            island_count, _ = cv2.connectedComponents(deep_inside)
            count_by_centres = island_count - 1           # minus one for the background
            ball_count = max(1, count_by_area, count_by_centres)

            group_ys, group_xs = np.nonzero(group_mask)
            colour_name = ball_colour.LABEL_NAMES[colour_label]

            if ball_count == 1:
                balls.append(make_ball(colour_name, group_xs, group_ys, to_map))
            else:
                # k-means on the pixel positions cuts the group into ball_count compact pieces.
                positions = np.column_stack([group_xs, group_ys])
                kmeans = KMeans(n_clusters=ball_count, n_init=5, random_state=0).fit(positions)
                for piece in range(ball_count):
                    in_piece = kmeans.labels_ == piece
                    balls.append(make_ball(colour_name, group_xs[in_piece], group_ys[in_piece], to_map))
    return balls


def find_balls(image_hsv, cloth_mask, table_contour, table_hull, to_map, to_photo):
    """
    Inputs 
    - image_hsv, cloth_mask : from find_cloth_mask()
    - table_contour, table_hull : from find_table_outline()
    - to_map, to_photo : from make_map_transform()
    Output
    - balls : a list of records, one per ball, each with colour, 
        photo_x, photo_y, radius, map_x, map_y
    """

    image_height, image_width = cloth_mask.shape

    # Two filled shapes: the exact cloth outline, and the hull stretched round it.
    cloth_region = np.zeros((image_height, image_width), dtype=np.uint8)
    cv2.drawContours(cloth_region, [table_contour], -1, 255, -1)
    hull_region = np.zeros((image_height, image_width), dtype=np.uint8)
    cv2.fillConvexPoly(hull_region, table_hull, 255)

    not_cloth = cv2.bitwise_not(cloth_mask)

    # Holes: inside the cloth outline but not cloth-coloured. These are balls on the bed.
    holes = cv2.bitwise_and(cloth_region, not_cloth)
    # Bites: inside the hull but outside the cloth outline. Pockets, rails, and balls on a cushion.
    bites = cv2.bitwise_and(hull_region, cv2.bitwise_not(cloth_region))

    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    holes = cv2.morphologyEx(holes, cv2.MORPH_OPEN, small_kernel)
    bites = cv2.morphologyEx(bites, cv2.MORPH_OPEN, small_kernel)

    balls = []
    for mask, at_edge in ((holes, False), (bites, True)):
        blob_count, blob_image, blob_stats, blob_centres = cv2.connectedComponentsWithStats(mask)

        for blob in range(1, blob_count):
            area = blob_stats[blob][4]
            if area < 40:
                continue

            centre_x, centre_y = blob_centres[blob]
            expected_radius = ball_radius_in_photo(to_map, to_photo, centre_x, centre_y)
            expected_area = np.pi * expected_radius * expected_radius

            if area < config.MIN_AREA_RATIO * expected_area:
                continue                                  # too small

            blob_mask = (blob_image == blob).astype(np.uint8)
            thickness = cv2.distanceTransform(blob_mask, cv2.DIST_L2, 5).max()
            if thickness < config.MIN_THICKNESS_RATIO * expected_radius:
                continue                                  # too thin: cue stick, seam or rail sliver

            ys, xs = np.nonzero(blob_mask)
            map_x, map_y = move_point(to_map, centre_x, centre_y)
            near_pocket = is_near_pocket(map_x, map_y)

            # Roundness = 4 * pi * area / perimeter^2. It is 1 for a circle and
            # drops towards 0 for long or ragged shapes.
            outlines, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            perimeter = cv2.arcLength(max(outlines, key=cv2.contourArea), True)
            roundness = 4 * np.pi * area / (perimeter * perimeter + 1e-9)

            looks_like_one_ball = (not at_edge) \
                and area <= config.MAX_SINGLE_BALL_AREA_RATIO * expected_area \
                and roundness >= config.MIN_ROUNDNESS

            if looks_like_one_ball:
                colour = ball_colour.classify_ball(image_hsv[ys, xs])
                if colour == 'unknown':
                    continue
                if colour == 'black' and near_pocket:
                    continue                              # a pocket opening, not the black ball
                balls.append(make_ball(colour, xs, ys, to_map))
            else:
                pieces = split_blob_by_colour(image_hsv, xs, ys, expected_radius,
                                              at_edge, at_edge or near_pocket, to_map)
                for piece in pieces:
                    balls.append(piece)

    return balls