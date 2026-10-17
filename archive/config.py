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

# True/False to force answer if the automatic long-side test gets it wrong.
FIRST_SIDE_IS_LONG = None

# Pool Background colour 

CLOTH_HUE_TOLERANCE = 12 
CLOTH_VALUE_TOLERANCE = 90 
CLOTH_SATURATION_DROP = 90 

# Ball colours 
RED_HUE = 170
YELLOW_HUE = 22
RED_HUE_TOLERANCE = 22
YELLOW_HUE_TOLERANCE = 16 
MIN_COLOUR_SATURATION = 100 

WHITE_MAX_SATURATION = 100
WHITE_MIN_VALUE = 150
DARK_MAX_VALUE = 70 

# A ball is called "cue" or "black" when this fraction of its pixels is white or dark
WHITE_FRACTION_FOR_CUE = 0.45 
DARK_FRACTION_FOR_BLACK = 0.45

# ball size and shape tests 

MIN_AREA_RATIO = 0.35 
MIN_THICKNESS_RATIO = 0.55 
MAX_SINGLE_BALL_AREA_RATIO = 1.7 
MIN_ROUNDNESS = 0.6 
