"""
Contains every number for tuning, other files stay readable 
colour values use OpenCV's HSV scale: Hue 0-180, Saturation 0-255, Value 0-255.

"""
# Rectangle for table mask
MAP_WIDTH = 800
MAP_HEIGHT = 400 

# ball radius measure in map pixels in comparision to iphone position
# (ball radius / width of the pool table rectangle)
BALL_RADIUS_ON_MAP = 10.2

# Pool Background colour 
CLOTH_HUE_TOLERANCE = 12 
CLOTH_VALUE_TOLERANCE = 90 
CLOTH_SATURATION_DROP = 90 

# Ball colour detection (HSV hues: 0-180 scale)
RED_HUE = 170
YELLOW_HUE = 22
RED_HUE_TOLERANCE = 22
YELLOW_HUE_TOLERANCE = 16
MIN_COLOUR_SATURATION = 100
WHITE_MAX_SATURATION = 100
WHITE_MIN_VALUE = 150
DARK_MAX_VALUE = 120              # a pixel darker than this (V) counts as dark; the black ball body is V 50-120
WHITE_FRACTION_FOR_CUE = 0.45     # used by ball_colour.classify_ball (photo pipeline only)
DARK_FRACTION_FOR_BLACK = 0.45    # used by ball_colour.classify_ball (photo pipeline only)

# Blob shape filters
MIN_AREA_RATIO = 0.35
MIN_THICKNESS_RATIO = 0.55
MAX_EDGE_BLOB_AREA_RATIO = 6.0    # a bite out of the cloth outline bigger than this is a pocket or rail, not balls

# Finding ball centres inside a blob (ball_detection.py)
MIN_COLOUR_FRACTION = 0.5         # CHANGED (was 0.35). A real ball has 0.85+ of its colour in the disc; the dark pocket between
                                  # three touching balls has under 0.5, so this stops it becoming a black ball
PEAK_MIN_RATIO = 0.30             # CHANGED (was 0.20). This camera is high (26 deg at the centre, 58 deg near): a ball hidden
                                  # behind another still shows 0.9 r, so 0.3 is safe and drops the thin shadow slivers (0.2-0.3 r)
WEAK_PEAK_RATIO = 0.70            # thinner than this = a "sliver": a ball mostly hidden behind another
SAME_BALL_ON_MAP = 1.5            # two centres closer than this many radii ON THE MAP are one ball
SAME_SLIVER_IN_PHOTO = 1.5        # two sliver centres closer than this many radii in the photo are one sliver
CLOSE_KERNEL_RATIO = 0.5          # closing kernel (fraction of a radius) that fills the highlight on a ball

# Black ball vs cue ball vs shadows (ball_detection.py colour_at)
BLACK_MIN_DARK_FRACTION = 0.15    # a dark-or-white ball with at least this much dark in its disc is the black ball, else the cue
BLACK_MAX_SATURATION = 60         # a dark pixel this colourless (grey) belongs to the black ball; a shadow on blue cloth stays saturated
BLACK_MIN_COLOURLESS_SHARE = 0.15 # a dark patch is the black ball only if at least this share of its dark pixels is colourless

# Corner orientation override (None = auto-detect)
FIRST_SIDE_IS_LONG = None

# Video: when can the board be read?  (run_video.py)
# The four "fraction" numbers are fractions of the table area in the picture, so they do not
# depend on the video size.
SIGNAL_SCALE = 0.5                # frames are shrunk by this before measuring motion (speed)
MOVING_PIXEL_DIFF = 25            # grey-level change between two frames that counts as "moving"
MOVE_HIGH_FRACTION = 0.002        # moving pixels above this: a shot is in progress
MOVE_LOW_FRACTION = 0.0005        # CHANGED (was 1e-5 = 1 px). Measured on Video 4: a still table gives 0 px at the median but
                                  # up to 71 px at the 99th percentile (sensor noise). 0.0005 = 68 px here; a shot gives 375+
HOLD_FRAMES = 15                  # still for this many frames in a row before the table counts as settled
TABLE_EDGE_TRIM = 15             # CHANGED (was 41). px (at SIGNAL_SCALE) trimmed off the table edge for the motion / blocked tests.
                                  # Video 4 is clear at 15 (checked on four clear frames); the only non-cloth near the edge is the
                                  # pocket mouths, which obstruction.py drops as ball-sized. Keep 41 only for IMG_0711.
# OCCLUDER_BIG_FRACTION / OCCLUDER_EDGE_FRACTION: no longer used once obstruction.py is in (delete when the old view_is_blocked goes)
OCCLUDER_BIG_FRACTION = 0.024
OCCLUDER_EDGE_FRACTION = 0.012
READ_EVERY = 15                   # frames between two readings of the board (0.5 s at 30 fps)
REFIND_AREA_TOLERANCE = 0.25      # moving camera: a re-found table more than 25% bigger/smaller than the first is a failed detection
REFIND_MAX_CORNER_SHIFT = 120     # moving camera: a re-found corner further than this (px) from the last one is a failed detection
CAMERA_SHIFT_PX = 1.5             # the whole picture shifting more than this (px, at SIGNAL_SCALE) in one frame = camera knocked/moved

AGREE_TOLERANCE = 1.0             # two readings agree if every ball is within this many map radii
WINDOW_BASE_RADII = 3.0     # search window half-size for a found ball, in ball radii
WINDOW_SPEED_FACTOR = 1.5   # extra half-size per map px the ball is expected to move this frame
WINDOW_MAX_RADII = 15.0     # the window doubles every frame a ball is unseen, up to this size
VELOCITY_BLEND = 0.6        # weight of a new velocity measurement (the rest is the old velocity): TO DO - remove this? set to 1?
MAX_COAST_FRAMES = 8        # a moving ball unseen for more frames than this is dropped
DROP_MIN_SPEED = 20.0       # map px/s; slower than this counts as still, and is never dropped
POCKET_ZONE_RADII = 4.0     # dropped within this many radii of a pocket centre = potted there
MAP_BORDER = 36             # must match the border in draw.draw_map
NO_MATCH = 1e6              # cost for a pairing outside a ball's window
STILL_WINDOW_RADII = 1.0    # tracking: search half-size for a still ball that has not been hit (it can only be where it is)
STILL_WINDOW_MAX_RADII = 2.5 # tracking: ... growing to at most this while the ball is not found (camera shake)
HIT_RADII = 3.0             # tracking: a still ball is free to move once a moving ball comes this close (it was hit)
TRACK_MAX_SPEED = 2500.0    # tracking: map px/s; a match further than a ball could roll since it was last seen is refused
MIN_TRACK_PEAK_RATIO = 0.5  # tracking: a colour patch thinner than this many radii is the cue stick or its tip, not a ball
MAX_TRACK_PEAK_RATIO = 1.4  # tracking: a colour patch thicker than this many radii is a hand or arm, not a ball
CUE_RELEASE_FRAMES = 2      # tracking: the cue ball is free to move once it is missing from its small window this many frames
CUE_MAX_SATURATION = 55     # tracking: cue-ball pixels are whiter than this (HSV saturation); skin is about 65-90
TRACK_PAUSE_OCCLUDER_FRACTION = 0.10  # no longer used once obstruction.hidden_balls is in (per-ball hidden flags replace the pause)

POCKETS = {'top-left': (0, 0), 'top-middle': (MAP_WIDTH / 2, 0), 'top-right': (MAP_WIDTH, 0),
           'bottom-left': (0, MAP_HEIGHT), 'bottom-middle': (MAP_WIDTH / 2, MAP_HEIGHT),
           'bottom-right': (MAP_WIDTH, MAP_HEIGHT)}

# Game rules and scoring (game_state.py, board_watcher.py)
BALLS_PER_COLOUR = 7              # blackball pool: 7 red, 7 yellow, 1 black, 1 cue
FOUL_GIVES_TWO_VISITS = True      # after a foul the other player gets two visits (pub rules)

# -----------------------------------------------------------------------------
# Obstruction detection (obstruction.py)
# -----------------------------------------------------------------------------
# Lump classification thresholds (in ball radii / ball areas)
THICK_RADII = 1.5                 # thickness > this = arm/hand/body
THIN_RADII = 0.6                  # opening kernel radius for thin part detection
THIN_MIN_BALLS = 1.0              # thin part area > this = cue/fingers
EDGE_CONTACT_RADII = 4.0          # edge contact length > this = hand reaching in
EDGE_MIN_BALLS = 3.0              # edge contact area > this many ball areas
EDGE_RING_WIDTH = 7               # px width of edge detection ring at SIGNAL_SCALE. CHECK: must equal the real width of
                                  # reference['edge_small'] = count_nonzero(edge_small) / hull perimeter. Built with a 7x7
                                  # erosion the ring is about 3.5 px wide, and 7 here halves every edge-contact measurement.
HUGE_BALLS = 15.0                 # area > this = huge obstruction
HIDDEN_DILATE_PX = 10             # dilate lump mask by this many px for hidden ball check
CLEAR_FRAMES = 5                  # blocked->clear requires this many clear frames
SHADOW_MIN_SATURATION = 40        # shadows keep cloth hue with at least this saturation
CAMERA_SHIFT_FRAMES = 3           # camera shift must persist this many frames