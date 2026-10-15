# AMME5710-Major
This repo contains the source code for our AMME5710 major project

## Pool Table Ball Detection

Detects and locates balls on a pool table from photos, then generates a top-down view.

## How It Works

1. **Find the table** (`table_mask.py`) - Detects the cloth colour and creates a mask of the playing surface

2. **Find the corners** (`table_corners.py`) - Fits lines to the table edges and finds where they intersect

3. **Create perspective transform** (`top_down.py`) - Maps between the angled photo and a flat top-down view

4. **Detect balls** (`ball_detection.py`) - Finds blobs that aren't cloth-coloured and filters by size/shape

5. **Classify colours** (`ball_colour.py`) - Labels each ball as cue, black, red, or yellow based on pixel colours

6. **Draw results** (`draw.py`) - Annotates the photo and draws the top-down map

## Usage

```bash
python run_photo.py
```

Processes all images in `Table images/` and saves results to `results/`.

## Requirements

- Python 3
- OpenCV
- NumPy
- scikit-learn

```bash
pip install -r requirements.txt
```
