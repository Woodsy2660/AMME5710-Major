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
MOVE_LOW_FRACTION = 0.0005        # moving pixels below this: still
HOLD_FRAMES = 15                  # still for this many frames in a row before the table counts as settled
OCCLUDER_BIG_FRACTION = 0.024     # a non-cloth lump bigger than this is a person, arm or cue
OCCLUDER_EDGE_FRACTION = 0.012    # ... or bigger than this and touching the table edge (arms come in from outside)
READ_EVERY = 15                   # frames between two readings of the board (0.5 s at 30 fps)
AGREE_TOLERANCE = 1.0             # two readings agree if every ball is within this many map radii