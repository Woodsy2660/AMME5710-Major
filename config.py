"""
Contains every number for tuning, other files stay readable 
colour values use OpenCV's HLS scale: Hue 0-180, Lightness 0-255, Saturation 0-255.

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
DARK_MAX_VALUE = 70
WHITE_FRACTION_FOR_CUE = 0.45
DARK_FRACTION_FOR_BLACK = 0.45

# Blob shape filters
MIN_AREA_RATIO = 0.35
MIN_THICKNESS_RATIO = 0.55
MAX_EDGE_BLOB_AREA_RATIO = 6.0    # a bite out of the cloth outline bigger than this is a pocket or rail, not balls

# Finding ball centres inside a blob (ball_detection.py)
MIN_COLOUR_FRACTION = 0.35        # a centre needs this much of its colour in the disc round it
PEAK_MIN_RATIO = 0.20             # a colour patch thinner than this fraction of a radius is not a ball
WEAK_PEAK_RATIO = 0.70            # thinner than this = a "sliver": a ball mostly hidden behind another
SAME_BALL_ON_MAP = 1.5            # two centres closer than this many radii ON THE MAP are one ball
SAME_SLIVER_IN_PHOTO = 1.5        # two sliver centres closer than this many radii in the photo are one sliver
CLOSE_KERNEL_RATIO = 0.5          # closing kernel (fraction of a radius) that fills the highlight on a ball

# Corner orientation override (None = auto-detect)
FIRST_SIDE_IS_LONG = None

# Video: when can the board be read?  (run_video.py)
# The four "fraction" numbers are fractions of the table area in the picture, so they do not
# depend on the video size.
SIGNAL_SCALE = 0.5                # frames are shrunk by this before measuring motion (speed)
MOVING_PIXEL_DIFF = 25            # grey-level change between two frames that counts as "moving"
MOVE_HIGH_FRACTION = 0.002        # moving pixels above this: a shot is in progress
MOVE_LOW_FRACTION = 1e-5          # moving pixels below this: still
HOLD_FRAMES = 15                  # still for this many frames in a row before the table counts as settled
OCCLUDER_BIG_FRACTION = 0.024     # a non-cloth lump bigger than this is a person, arm or cue
OCCLUDER_EDGE_FRACTION = 0.012    # ... or bigger than this and touching the table edge (arms come in from outside)
READ_EVERY = 15                   # frames between two readings of the board (0.5 s at 30 fps)

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

POCKETS = {'top-left': (0, 0), 'top-middle': (MAP_WIDTH / 2, 0), 'top-right': (MAP_WIDTH, 0),
           'bottom-left': (0, MAP_HEIGHT), 'bottom-middle': (MAP_WIDTH / 2, MAP_HEIGHT),
           'bottom-right': (MAP_WIDTH, MAP_HEIGHT)}

AGREE_TOLERANCE = 1.0             # two readings agree if every ball is within this many map radii
# Game rules and scoring (game_state.py, board_watcher.py)
BALLS_PER_COLOUR = 7              # blackball pool: 7 red, 7 yellow, 1 black, 1 cue
FOUL_GIVES_TWO_VISITS = True      # after a foul the other player gets two visits (pub rules)

