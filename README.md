# AMME5710-Major
This repo contains the source code for our AMME5710 major project.

## Pool Table Ball Detection, Tracking & Scoring

Finds the table and the balls from a phone camera on a tripod, draws a top-down map of the table, follows every shot and scores the game (blackball pool, 7 red / 7 yellow).

---

## Folder Layout

```
AMME5710-Major/
├── gui.py                 game window: live camera, recorded video or photos (main program)
├── board_watcher.py       turns a video into shots and trusted board states
├── board_reading.py       table reference from the first frame, ball list for a frame
├── table_mask.py          cloth colour and table outline
├── table_corners.py       four table corners from the outline
├── top_down.py            photo <-> top-down map transform, expected ball size
├── ball_detection.py      ball finding on a settled table
├── ball_colour.py         ball colour classification
├── ball_tracking.py       ball tracking during a shot (IDs, first contact, pockets)
├── tracking_helpers.py    helpers for ball_tracking.py
├── shot_detection.py      motion state machine (SHOT / SETTLED)
├── obstruction.py         arms, cues and bodies over the table
├── game_state.py          scoring rules, turns and fouls
├── shot_log.py            CSV log of every scored shot and frame time (gui.py --log)
├── draw.py                drawing the outline, balls and top-down map
├── config.py              every tuning number
├── run_photo.py           batch detection on still photos
├── diagnose.py            saves debug frames from a video
├── extract_frames.py      saves frames from a video as JPEGs
├── test_game_state.py     tests for the scoring rules (no video needed)
├── requirements.txt
├── eval/                  evaluation scripts and hand-made ground truth (see Evaluation)
├── results/eval/          evaluation results (tables, CSVs, figures)
└── Videos/              not included: download from Google Drive (see Videos)
    ├── Game videos at normal diagonal view/   main test videos (tripod at a table corner)
    ├── Game videos at other view angles/      other camera positions
    └── GUI recordings/                        screen recordings of the game window
```

---

## Videos

The videos are too large to include. Download them from Google Drive:

**https://drive.google.com/drive/folders/1PdaQXUE6kI3Szcc3Ps4YSRILCWtXDSFV?usp=drive_link**

Put the three folders from the Drive inside a `Videos/` folder at the repo root, so the paths are:

```
Videos/Game videos at normal diagonal view/
Videos/Game videos at other view angles/
Videos/GUI recordings/
```

The game window, `diagnose.py`, `extract_frames.py` and some evaluation scripts need them (see Evaluation). The scoring tests, `ball_eval.py`, `brightness_sweep.py` and all the results in `results/eval/` do not.

---

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
   - Reject ball shadows: a shadow keeps the cloth's blue saturation, the black ball has a grey core
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

### Video Tracking (`board_reading.py`, `shot_detection.py`, `obstruction.py`, `board_watcher.py`)

1. **Setup from first frame** (`board_reading.make_reference`)
   - Find cloth colour and table corners once
   - Create scaled-down masks for motion detection
   - Store the reference for the entire video (press `k` or use `--moving-camera` to find the table again)

2. **Detect motion** (`shot_detection.py`)
   - Shrink frame to half size and convert to grayscale
   - Compare pixel differences with previous frame
   - Count changed pixels inside the table area
   - High motion = shot in progress, low motion = settling

3. **Check for obstructions** (`obstruction.py`)
   - Measure each non-cloth lump inside the table by its shape in ball units (thickness, area, edge contact)
   - Ball-shaped lumps are ignored; arms, hands, cues and bodies block the view
   - During a shot, balls under an obstruction are marked hidden and keep coasting instead of being called potted

4. **Read when settled** (`board_reading.read_board`)
   - Wait for motion to stay low for 15 frames
   - Confirm the view is not blocked
   - Run the full detection pipeline on the frame

5. **Keep trusted states** (`board_watcher.py`)
   - Two readings half a second apart must agree before a state is kept

6. **Track the shot** (`ball_tracking.py`)
   - Every ball gets an ID and is followed frame by frame on the map
   - Records which balls moved, what the cue ball hit first and which pocket each potted ball went into

### Scoring and GUI (`board_watcher.py`, `game_state.py`, `gui.py`)

1. **Shots** (`board_watcher.py`)
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
   - Keys: `space` pause (photos: next photo), `k` find the table again, `u` undo the last shot, `s` swap turn, `c` swap colours, `f` last shot was not a foul, `x` last shot was a foul, `n` new game, `q` / `esc` quit
   - `--log logs` saves every scored shot and the processing time of every frame as CSV (`shot_log.py`)

---

## Usage

Install the requirements first:
```bash
pip install -r requirements.txt
```

**Game window:**
```bash
python gui.py --video "Videos/Game videos at normal diagonal view/IMG_7429.MOV"
python gui.py --video "<video>" --start 30 --speed 2     # start at 30 s, twice real time
python gui.py --video "<video>" --start 120 --end 200    # only part of a recording
python gui.py --video "<video>" --table-edge-trim 41     # high, far camera angle
python gui.py --video "<video>" --log logs               # save shots and frame times as CSV
python gui.py --video "<video>" --record demo.mp4 --headless   # save the window as a video
python gui.py                                            # webcam 0 (e.g. an iPhone)
python gui.py --photos "Table images"                    # still photos, space = next
```
`python gui.py --help` lists every option.

**Photos:**
```bash
python run_photo.py
```
Processes every image in a `Table images/` folder next to the scripts and saves the results to `results/`. The folder is not included: create it and put the photos in it first.

**Debugging tools:**
```bash
python extract_frames.py "<video>"    # save frames as JPEGs
python diagnose.py "<video>"          # save frames with debug overlays
```

**Tests** (no video needed):
```bash
python test_game_state.py
```

`config.py` is tuned for the diagonal-view videos. Other camera positions may need `--table-edge-trim` (about 41 for a high, far camera).

---

## Evaluation (`eval/`)

Scripts that measure the pipeline for the report. They call the pipeline's own functions and never change `config.py`; settings that are varied are set at run time only. Results go to `results/eval/`, and every script also writes a `<script>_summary.md` table there. Run them from the repo root.

**Ground truth (made by hand, already included):**

| File | Made with | Contents |
|---|---|---|
| `eval/videos.csv` | by hand | The evaluation videos, camera position, start time |
| `eval/gt/<video>_corners.json` | `eval/click_corners.py` | Hand-clicked table corners for each video |
| `eval/frames/` | `eval/pick_frames.py` | Frames exported for ball labelling (`frames.csv` lists them) |
| `eval/gt/<frame>_balls.json` | `eval/annotate_balls.py`, `eval/label_frames.py` | Hand-labelled ball centres and colours |
| `eval/shots_test_4.csv` | `eval/mark_shots.py` | Hand-marked shot end times for test_4 |

**Experiments:**
```bash
python eval/calibration_eval.py                       # table corners vs the clicked corners
python eval/ball_eval.py                              # ball detection vs the hand labels
python eval/ball_eval.py --method hough               # Hough-circle baseline
python eval/ball_eval.py --no-shadow-test             # without the black-vs-shadow test
python eval/brightness_sweep.py                       # darker / brighter frames
python eval/clip_info.py                              # duration, fps, frames of each clip
python eval/timing.py "Videos/Game videos at normal diagonal view/test_4.mp4"
python eval/threshold_sweep.py "Videos/Game videos at normal diagonal view/test_4.mp4" --shots eval/shots_test_4.csv
```
`calibration_eval.py`, `clip_info.py`, `timing.py` and `threshold_sweep.py` need the videos in `Videos/` ([download from Google Drive](https://drive.google.com/drive/folders/1PdaQXUE6kI3Szcc3Ps4YSRILCWtXDSFV?usp=drive_link)). `ball_eval.py` and `brightness_sweep.py` only need `eval/frames/` and `eval/gt/`.

---

## Requirements

- Python 3 (tested with Python 3.12)
- OpenCV
- NumPy
- scikit-learn
- SciPy
- Matplotlib (evaluation figures only)

```bash
pip install -r requirements.txt
```
