# Pool Table Ball Detection & Scoring System - Architecture Report

## Overview

This is a computer vision system for detecting pool balls, tracking them during shots, and scoring a game of blackball pool. It processes video from a camera overlooking a pool table and automatically determines ball positions, tracks movement during shots, and applies scoring rules.

---

## Repository Structure

```
AMME5710-Major/
├── gui.py                 # Main entry point - game window with live view & scoring
├── board_watcher.py       # Video orchestrator - turns frames into shots
├── board_reading.py       # Reference setup and board state reading
├── ball_tracking.py       # Frame-by-frame ball tracking during shots
├── tracking_helpers.py    # Search windows and candidate detection for tracking
├── shot_detection.py      # Motion detection state machine (SHOT/SETTLED)
├── ball_detection.py      # Static ball detection using blob analysis
├── ball_colour.py         # Pixel colour classification (red/yellow/cue/black)
├── table_mask.py          # Cloth detection and table outline extraction
├── table_corners.py       # Four corner detection using k-means clustering
├── top_down.py            # Perspective transform (photo <-> map)
├── draw.py                # Visualization utilities
├── game_state.py          # Blackball pool rules engine
├── config.py              # All tunable parameters (centralized)
├── run_photo.py           # Batch processing for still images
└── test_game_state.py     # Unit tests for scoring rules
```

---

## Module Dependency Graph

```
                         ┌──────────────┐
                         │    gui.py    │  (Entry Point)
                         └──────┬───────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
              v                 v                 v
     ┌────────────────┐  ┌─────────────┐  ┌──────────────┐
     │ board_watcher  │  │ game_state  │  │     draw     │
     └───────┬────────┘  └─────────────┘  └──────────────┘
             │
    ┌────────┴────────┬─────────────────┐
    │                 │                 │
    v                 v                 v
┌──────────────┐ ┌─────────────┐ ┌────────────────┐
│board_reading │ │ball_tracking│ │shot_detection  │
└──────┬───────┘ └──────┬──────┘ └────────────────┘
       │                │
       │     ┌──────────┴──────────┐
       │     │                     │
       v     v                     v
  ┌──────────────┐        ┌─────────────────┐
  │ball_detection│        │tracking_helpers │
  └──────┬───────┘        └────────┬────────┘
         │                         │
    ┌────┴────┐                    │
    │         │                    │
    v         v                    v
┌────────┐ ┌──────────┐     ┌─────────────┐
│ball_   │ │table_    │     │ball_colour  │
│colour  │ │corners   │     │ball_detection│
└────────┘ └──────────┘     └─────────────┘
                 │
                 v
          ┌────────────┐
          │ table_mask │
          └────────────┘
                 │
                 v
          ┌────────────┐
          │  top_down  │
          └────────────┘
                 │
                 v
          ┌────────────┐
          │   config   │ (used by all modules)
          └────────────┘
```

---

## Data Flow Pipeline

### Static Detection (Single Frame)

```
Image Frame
    │
    v
┌─────────────────────────────────────────────────────────────────────┐
│ Step 1: table_mask.find_cloth_mask()                                │
│   - Convert to HSV                                                  │
│   - Sample cloth colour from click point (or auto-detect)          │
│   - LUT-based threshold for fast cloth detection                   │
│   Output: cloth_mask (binary), image_hsv                           │
└─────────────────────────────────────────────────────────────────────┘
    │
    v
┌─────────────────────────────────────────────────────────────────────┐
│ Step 2: table_mask.find_table_outline()                            │
│   - Morphological cleanup (open + close)                           │
│   - Find largest contour containing the click                      │
│   - Compute convex hull (bridges pockets)                          │
│   Output: table_contour, table_hull                                │
└─────────────────────────────────────────────────────────────────────┘
    │
    v
┌─────────────────────────────────────────────────────────────────────┐
│ Step 3: table_corners.find_corners()                               │
│   - Extract edge normals from hull                                 │
│   - K-means cluster into 4 sides                                   │
│   - Fit lines to each side                                         │
│   - Intersect adjacent lines for corners                           │
│   Output: 4x2 array of corner coordinates                          │
└─────────────────────────────────────────────────────────────────────┘
    │
    v
┌─────────────────────────────────────────────────────────────────────┐
│ Step 4: top_down.make_map_transform()                              │
│   - getPerspectiveTransform from corners to 800x400 map            │
│   Output: to_map (3x3 matrix), to_photo (inverse)                  │
└─────────────────────────────────────────────────────────────────────┘
    │
    v
┌─────────────────────────────────────────────────────────────────────┐
│ Step 5: ball_detection.find_balls()                                │
│   - Find blobs: holes (inside cloth) + bites (outside contour)    │
│   - Filter by expected ball size at each position                  │
│   - Separate by colour family (red, yellow, dark/white)            │
│   - Distance transform peaks = ball centres                        │
│   - Shift sliver centres inward for partially hidden balls         │
│   - Colour check and deduplication                                 │
│   Output: list of ball records                                     │
└─────────────────────────────────────────────────────────────────────┘
    │
    v
┌─────────────────────────────────────────────────────────────────────┐
│ Step 6: draw.draw_overlay() + draw.draw_map()                      │
│   - Annotate photo with circles and labels                         │
│   - Render 800x400 top-down view                                   │
│   Output: annotated images                                         │
└─────────────────────────────────────────────────────────────────────┘
```

### Video Processing (Live Tracking)

```
Video Frame Stream
    │
    ┌───────────────────────────────────────────────────────────────┐
    │           BoardWatcher.update() - Main Loop                   │
    └───────────────────────────────────────────────────────────────┘
    │
    v
┌─────────────────────────────────────────────────────────────────────┐
│ Motion Detection: shot_detection                                   │
│   - shrink_to_gray(): downscale frame for speed                   │
│   - moving_pixels(): count changed pixels inside table            │
│   - update_detector(): state machine                              │
│       SHOT: moving > MOVE_HIGH_FRACTION                           │
│       SETTLED: still_count >= HOLD_FRAMES                         │
│   - view_is_blocked(): detect arms/cues over table                │
└─────────────────────────────────────────────────────────────────────┘
    │
    ├──── If SHOT ────┐
    │                 v
    │   ┌─────────────────────────────────────────────────────────┐
    │   │ Ball Tracking: ball_tracking.track_frame()             │
    │   │   - Initialise tracks from last committed state        │
    │   │   - free_hit_balls(): determine which balls can move   │
    │   │   - For each colour:                                   │
    │   │       - find_candidates() in search windows            │
    │   │       - Hungarian assignment (linear_sum_assignment)   │
    │   │       - see(): update found balls                      │
    │   │       - coast(): predict unseen balls                  │
    │   │   - Detect pots (ball lost near pocket)                │
    │   │   Output: windows dict, events list                    │
    │   └─────────────────────────────────────────────────────────┘
    │
    └──── If SETTLED & readable ────┐
                                    v
    ┌─────────────────────────────────────────────────────────────────┐
    │ Board Reading: board_reading.read_board()                      │
    │   - Run static ball detection on current frame                 │
    │   - Compare with previous reading (same_reading check)         │
    │   - If two consecutive readings agree → commit()               │
    └─────────────────────────────────────────────────────────────────┘
    │
    v
┌─────────────────────────────────────────────────────────────────────┐
│ Shot Result: board_watcher.shot_result()                           │
│   - Compare ball counts before vs after                            │
│   - Determine what was potted                                      │
│   - Identify closest pocket for each gone ball                     │
│   Output: {'potted': {...}, 'note': 'red TL, yellow BM'}          │
└─────────────────────────────────────────────────────────────────────┘
    │
    v
┌─────────────────────────────────────────────────────────────────────┐
│ Game Rules: game_state.apply_shot()                                │
│   - Apply blackball pool rules                                     │
│   - Detect fouls (cue potted, opponent ball)                       │
│   - Assign colours on first legal pot                              │
│   - Handle black ball (win/lose)                                   │
│   - Determine turn continuation                                    │
│   Output: event dict + updated game state                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Data Structures

### Ball Record (from ball_detection.find_balls)
```python
{
    'colour': 'red' | 'yellow' | 'black' | 'cue',
    'photo_x': float,      # centre in original image (px)
    'photo_y': float,
    'radius': float,       # expected radius at this position (px)
    'map_x': float,        # centre on 800x400 top-down map
    'map_y': float,
    'visible': float       # peak_height / radius (1.0 = fully visible)
}
```

### Track Record (from ball_tracking.initialise_tracking)
```python
{
    'colour': str,
    'x': float,            # current map position
    'y': float,
    'vx': float,           # velocity (map px/s)
    'vy': float,
    'seen_x': float,       # last actually-seen position
    'seen_y': float,
    'seen_t': float,       # time of last sighting
    'missed': int,         # frames since last sighting
    'vanish': tuple,       # first predicted position when lost
    'lost': bool,          # dropped from tracking
    'free': bool,          # allowed to move (hit by another ball)
    'path': [(x, y, found), ...]  # trajectory this shot
}
```

### Reference Dictionary (from board_reading.make_reference)
```python
{
    'cloth_colour': (h, s, v),         # median HSV of cloth
    'table_hull': np.array,            # convex hull points
    'corners': np.array (4x2),         # four corners, long side first
    'to_map': np.array (3x3),          # photo → map homography
    'to_photo': np.array (3x3),        # map → photo homography
    'inner_small': np.array,           # eroded hull at SIGNAL_SCALE
    'edge_small': np.array,            # thin ring at table edge
    'table_area_small': int,           # pixel count at SIGNAL_SCALE
    'hull_mask': np.array              # full-res filled hull
}
```

### GameState
```python
{
    'players': ['Player 1', 'Player 2'],
    'current': 0 | 1,                  # index of player at table
    'visits_left': 1 | 2,              # 2 after opponent fouls
    'groups': [None | 'red' | 'yellow', ...],  # assigned colours
    'remaining': {'red': 7, 'yellow': 7},
    'fouls': [0, 0],
    'winner': None | 0 | 1,
    'shot_number': int,
    'log': [str, ...]                  # event descriptions
}
```

---

## Configuration Parameters (config.py)

### Map Dimensions
| Parameter | Value | Description |
|-----------|-------|-------------|
| MAP_WIDTH | 800 | Top-down map width (px) |
| MAP_HEIGHT | 400 | Top-down map height (px) |
| BALL_RADIUS_ON_MAP | 10.2 | Ball radius in map pixels |
| MAP_BORDER | 36 | Border around map for display |

### Cloth Detection
| Parameter | Value | Description |
|-----------|-------|-------------|
| CLOTH_HUE_TOLERANCE | 12 | HSV hue tolerance (0-180) |
| CLOTH_VALUE_TOLERANCE | 90 | Brightness tolerance |
| CLOTH_SATURATION_DROP | 90 | Min saturation below cloth |

### Ball Colour Detection
| Parameter | Value | Description |
|-----------|-------|-------------|
| RED_HUE | 170 | Target hue for red balls |
| YELLOW_HUE | 22 | Target hue for yellow balls |
| RED_HUE_TOLERANCE | 22 | Red hue matching range |
| YELLOW_HUE_TOLERANCE | 16 | Yellow hue matching range |
| MIN_COLOUR_SATURATION | 100 | Min saturation for coloured |
| WHITE_MAX_SATURATION | 100 | Max saturation for white |
| WHITE_MIN_VALUE | 150 | Min brightness for white |
| DARK_MAX_VALUE | 70 | Max brightness for black |

### Motion Detection
| Parameter | Value | Description |
|-----------|-------|-------------|
| SIGNAL_SCALE | 0.5 | Downscale factor for speed |
| MOVING_PIXEL_DIFF | 25 | Pixel change threshold |
| MOVE_HIGH_FRACTION | 0.002 | Fraction → SHOT state |
| MOVE_LOW_FRACTION | 1e-5 | Fraction → counting as still |
| HOLD_FRAMES | 15 | Still frames before SETTLED |

### Ball Tracking
| Parameter | Value | Description |
|-----------|-------|-------------|
| WINDOW_BASE_RADII | 3.0 | Base search window size |
| WINDOW_MAX_RADII | 15.0 | Max window size |
| VELOCITY_BLEND | 0.6 | New vs old velocity weight |
| MAX_COAST_FRAMES | 8 | Frames before dropping ball |
| TRACK_MAX_SPEED | 2500.0 | Max ball speed (map px/s) |
| HIT_RADII | 3.0 | Proximity to "free" a ball |
| CUE_RELEASE_FRAMES | 2 | Frames cue missing → struck |

### Game Rules
| Parameter | Value | Description |
|-----------|-------|-------------|
| BALLS_PER_COLOUR | 7 | Red/yellow balls each |
| FOUL_GIVES_TWO_VISITS | True | Two visits after foul |
| POCKET_ZONE_RADII | 4.0 | Proximity to count as potted |

---

## Entry Points

### gui.py - Main Application
```bash
# Live camera
python gui.py

# Video file
python gui.py --video "game.mp4"

# With options
python gui.py --video game.mp4 --speed 2 --start 60 --end 300
python gui.py --video clip.mov --mid-game --record output.mp4 --headless

# Still photos
python gui.py --photos "Table images"
```

**Key Controls:**
- `q` / `esc`: Quit
- `space`: Pause
- `u`: Undo last shot
- `s`: Swap turn
- `n`: New game
- `k`: Re-find table
- `r/y/b/c`: Log ground truth (red/yellow/black/cue potted)
- `enter`: Save ground truth entry

### run_photo.py - Batch Image Processing
```bash
python run_photo.py
```
Processes all images in `Table images/` folder, saves annotated results to `results/`.

---

## Key Algorithms

### 1. Cloth Detection (table_mask.py)
- Sample cloth colour from a 40x40 patch at click point
- Build LUT (lookup table) for fast thresholding
- Hue wraps at 180, so distance = min(diff, 180-diff)
- Morphological open removes noise, close fills gaps
- Largest contour containing click = table

### 2. Corner Detection (table_corners.py)
- Extract edge normals from convex hull
- Weight by edge length (longer edges get more votes)
- K-means (k=4) clusters normals into four sides
- Fit line to each cluster using cv2.fitLine
- Corners = intersections of adjacent lines

### 3. Ball Detection (ball_detection.py)
- Find "holes" (non-cloth inside outline) and "bites" (hull - outline)
- Filter blobs by expected area at each position
- Classify pixels into families: red, yellow, dark, white
- Close masks to fill highlights
- Distance transform peaks = ball centres
- Shift sliver peaks inward for partially hidden balls

### 4. Ball Tracking (ball_tracking.py)
- Each ball gets a track with position, velocity, history
- Balls start "not free" (can't move until hit)
- Cue ball freed after missing from small window for 2 frames
- Other balls freed when a moving ball's path passes within 3 radii
- Hungarian assignment matches detections to tracks
- Coasting: predict position when not found, drop after 8 frames
- Dropped near pocket = potted

### 5. Shot Detection State Machine (shot_detection.py)
```
         moving > HIGH
              ↓
SETTLED ─────────────→ SHOT
    ↑                    │
    │                    │ moving < LOW for
    │                    │ HOLD_FRAMES frames
    └────────────────────┘
```

---

## Testing

```bash
python test_game_state.py
```

Tests the scoring rules engine with simulated shot sequences covering:
- Potting own balls
- Fouls (cue ball, opponent's ball)
- Open table colour assignment
- Two-visit rule
- Black ball win/loss conditions

---

## Performance Notes

- Cloth mask uses LUT for O(1) per-pixel lookup instead of per-pixel arithmetic
- Motion detection runs at 50% scale (SIGNAL_SCALE = 0.5)
- Tracking uses small search windows that grow only when ball is lost
- Board reading only runs every 15 frames (READ_EVERY) when settled
- Two consecutive agreeing readings required before committing state
