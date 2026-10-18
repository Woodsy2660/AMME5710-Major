"""
Step 2: turn thee table hull into four corners, work out which sides are the long ones 

- treat each hull edge as a line 
- k-means into four groups (four sides)
- intersect neighbouring sides to get the corners 
"""

import cv2 
import numpy as np 
from sklearn.cluster import KMeans

import config 

def find_corners(table_hull):
    """
    Input 
    - table_hull : from find_table_outline() 
    Output
    - corners : 4 x 2 array of (x, y) pixels, in order going roud the table
    """

    points = table_hull.reshape(-1, 2).astype(float)
    number_of_points = len(points)

    # The centre of the hull. Every side will be described relative to this point.
    centre = points.mean(axis=0)
    perimeter = cv2.arcLength(table_hull, True)

    edge_normals = []      # which way each edge faces
    edge_ends = []         # the two end points of each edge
    votes = []             # one copy of the normal for every 5 pixels of edge length

    for i in range(number_of_points):
        start = points[i]
        end = points[(i + 1) % number_of_points]
        direction = end - start
        length = np.hypot(direction[0], direction[1])

        # Skip the short edges that cut across the corner pockets.
        if length < 0.04 * perimeter:
            continue

        # The normal is the edge direction turned 90 degrees, scaled to length 1.
        normal = np.array([direction[1], -direction[0]]) / length
        # Make it point away from the centre of the table.
        if np.dot(normal, start - centre) < 0:
            normal = -normal

        edge_normals.append(normal)
        edge_ends.append((start, end))
        for _ in range(max(1, int(length / 5))):
            votes.append(normal)

    # The four sides of the table face four different ways, about 90 degrees apart,
    # so k-means with k = 4 on the normals separates the edges into the four sides.
    kmeans = KMeans(n_clusters=4, n_init=10, random_state=0)
    kmeans.fit(np.array(votes))
    edge_side = kmeans.predict(np.array(edge_normals))

    # Fit one straight line to all the end points belonging to each side.
    # A line is stored as (a, b, c), meaning  a*x + b*y + c = 0.
    sides = []
    for side in range(4):
        side_points = []
        for edge_index in range(len(edge_ends)):
            if edge_side[edge_index] == side:
                side_points.append(edge_ends[edge_index][0])
                side_points.append(edge_ends[edge_index][1])
        side_points = np.array(side_points, dtype=np.float32)

        vx, vy, x0, y0 = cv2.fitLine(side_points, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        a = vy
        b = -vx
        c = -(a * x0 + b * y0)

        # The angle the side faces, used to put the sides in order round the table.
        facing = kmeans.cluster_centers_[side]
        angle = np.arctan2(facing[1], facing[0])
        sides.append((angle, np.array([a, b, c])))

    sides.sort(key=lambda item: item[0])

    # Neighbouring sides meet at a corner.
    # For lines l1 and l2 written as (a, b, c), the cross product l1 x l2 gives a
    # point (X, Y, W) that lies on both lines. The pixel is (X / W, Y / W).
    corners = []
    for i in range(4):
        line_1 = sides[i][1]
        line_2 = sides[(i + 1) % 4][1]
        meeting_point = np.cross(line_1, line_2)
        corners.append([meeting_point[0] / meeting_point[2], meeting_point[1] / meeting_point[2]])

    return np.array(corners, dtype=np.float32)
 
def put_long_side_first(corners, table_contour):
    """
    Reorders the corners so that corners[0] -> corners[1] is a long side 

    this is decided from the middle pocket of the table (deep bite out of the cloth outline)

    Inputs
    - corners : from find_corners()
    - table_contour " from find_table_outline() 
    Output
    - corners : the same four corners, shifted to have long side first if needed
    """

    if config.FIRST_SIDE_IS_LONG is True:
        return corners
    if config.FIRST_SIDE_IS_LONG is False:
        return np.roll(corners, -1, axis=0)

    width = config.MAP_WIDTH
    height = config.MAP_HEIGHT
    map_corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

    # convexityDefects lists every bite out of the outline: where it is and how deep.
    hull_indices = cv2.convexHull(table_contour, returnPoints=False)
    defects = cv2.convexityDefects(table_contour, hull_indices).reshape(-1, 4)

    deepest = 0
    for defect in defects:
        depth = defect[3] / 256.0
        if depth > deepest:
            deepest = depth

    best_score = None
    best_corners = corners
    for shift in (0, 1):
        candidate = np.roll(corners, -shift, axis=0)
        transform = cv2.getPerspectiveTransform(candidate, map_corners)

        score = 0
        for defect in defects:
            depth = defect[3] / 256.0
            if depth < 0.3 * deepest:
                continue                      # a shallow dent, not a pocket

            # The deepest point of this bite, moved onto the map.
            x, y = table_contour[int(defect[2])][0]
            moved = transform @ np.array([x, y, 1.0])
            map_x = moved[0] / moved[2]
            map_y = moved[1] / moved[2]

            near_long_middle = abs(map_x - width / 2) < 0.15 * width \
                and (map_y < 0.3 * height or map_y > 0.7 * height)
            near_short_middle = abs(map_y - height / 2) < 0.3 * height \
                and (map_x < 0.15 * width or map_x > 0.85 * width)

            if near_long_middle:
                score = score + depth         # a pocket where one should be
            if near_short_middle:
                score = score - depth         # a pocket where there should be none

        if best_score is None or score > best_score:
            best_score = score
            best_corners = candidate

    return best_corners
