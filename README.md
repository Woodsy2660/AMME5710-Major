# AMME5710-Major
This repo contains the source code for our AMME5710 major project

## Pool Table Ball Detection & Tracking

Detects and locates balls on a pool table from photos or video, generates a top-down view, and tracks game events.

## How It Works

### Detection Pipeline

1. **Find the table** (`table_mask.py`)
   - Apply Gaussian blur to reduce noise
   - Convert image from BGR to HSV colour space
   - Sample a 40x40 pixel patch to get the cloth colour
   - Threshold pixels by hue, saturation, and value similarity
   - Clean mask with morphological opening and closing
   - Find the largest contour containing the sample point
   - Compute convex hull to bridge across pocket indentations

2. **Find the corners** (`table_corners.py`)
   - Extract edge segments from the convex hull
   - Skip short edges (these are pocket cuts)
   - Compute outward-facing normal for each edge
   - K-means clustering (k=4) groups edges into four sides
   - Fit a line to each side using least squares
   - Intersect adjacent lines to get the four corners
   - Reorder corners so the long side comes first (using pocket depth)

3. **Create perspective transform** (`top_down.py`)
   - Define target rectangle (800x400 pixels, 2:1 ratio)
   - Compute homography matrix from four corner pairs
   - Create inverse matrix for mapping back to photo
   - Calculate expected ball radius at any position (accounts for perspective)

4. **Detect balls** (`ball_detection.py`)
   - Create "holes" mask: inside cloth outline but not cloth-coloured
   - Create "bites" mask: inside hull but outside contour (balls on cushions)
   - Find connected components in each mask
   - Filter blobs by minimum area and thickness
   - Calculate roundness (4πA/P²) to identify single balls
   - Split irregular blobs by colour, then by k-means clustering
   - Map each ball position to top-down coordinates

5. **Classify colours** (`ball_colour.py`)
   - Label each pixel: white, dark, red, yellow, or other
   - White: low saturation (<100) and high value (>150)
   - Dark: low value (<70)
   - Red: hue near 170 with good saturation
   - Yellow: hue near 22 with good saturation
   - If >45% white pixels → cue ball
   - If >45% dark pixels → black ball
   - Otherwise average coloured pixel hues and pick nearest

6. **Draw results** (`draw.py`)
   - Draw green convex hull outline on the photo
   - Mark corners with red circles
   - Draw coloured rings around each ball with letter labels
   - Create 800x400 top-down map with pockets and ball positions

### Video Tracking (`run_video.py`)

1. **Setup from first frame**
   - Find cloth colour and table corners once
   - Create scaled-down masks for motion detection
   - Store reference for the entire video

2. **Detect motion**
   - Shrink frame to half size and convert to grayscale
   - Compare pixel differences with previous frame
   - Count changed pixels inside the table area
   - High motion = shot in progress, low motion = settling

3. **Check for obstructions**
   - Find non-cloth regions inside the table
   - Measure the largest lump and check if it touches the edge
   - Arms and cues come from outside; ball clusters stay in the middle

4. **Read when settled**
   - Wait for motion to stay low for 15 frames
   - Confirm the view is not blocked
   - Run the full detection pipeline on the frame

5. **Commit states**
   - Take readings every 15 frames while settled
   - Compare ball positions between two readings
   - If all balls match within tolerance, commit the state

6. **Report events**
   - Compare consecutive committed states
   - Ball in old state with no match in new = potted (name nearest pocket)
   - Ball in new state with no match in old = appeared

### Scoring and GUI (`board_watcher.py`, `game_state.py`, `gui.py`)

1. **Shots** (`board_watcher.py`)
   - Uses `run_video.py`'s still / view-blocked / two-readings-agree tests to keep trusted board states
   - A shot = table motion between two kept states AND the balls ended up somewhere else
   - Potted = ball counts per colour before minus after (capped at a full rack), with the nearest pocket named
   - Cue ball put back by hand after a scratch is not a shot

2. **Rules** (`game_state.py`) - blackball pool, 7 red / 7 yellow
   - Open table until a player pots only one colour
   - Potting your own colour keeps your turn
   - Fouls: cue ball potted, opponent's colour potted; the other player gets two visits
   - Black: wins after clearing your colour, loses otherwise (including with your last ball)

3. **Game window** (`gui.py`)
   - Live view with the last kept board state, top-down map, scoreboard, event log
   - Keys: `u` undo shot, `s` swap turn, `n` new game, `k` find table again, `space` pause, `q` quit
   - Ground-truth logger: `r` `y` `b` `c` then `enter` records what really went down; every game writes `logs/game_<time>.csv` with system and truth rows for scoring accuracy

Tests (no video needed): `python test_game_state.py`

## Usage

**Photos:**
```bash
python run_photo.py
```
Processes all images in `Table images/` and saves results to `results/`.

**Video:**
```bash
python run_video.py "path/to/video.mp4"
```
Outputs: `_tracked.mp4` (annotated video), `_state_N.jpg` (each board state), `_events.txt` (game events)

**Game window:**
```bash
python gui.py --video "path/to/video.mp4" --players Ben Natalie
python gui.py                            # webcam / iPhone
python gui.py --photos "Table images"    # still photos, space = next
```

## Requirements

- Python 3
- OpenCV
- NumPy
- scikit-learn
- SciPy

```bash
pip install -r requirements.txt
```
