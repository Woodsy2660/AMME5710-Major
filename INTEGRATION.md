# Obstruction rework: `obstruction.py`

One new file replaces the three scattered functions (`view_is_blocked`, `covers_balls`,
`camera_moved`) and answers the same three questions from one measurement. Tested on both
tripod recordings: states and events unchanged on Video 2 (7 states), one false state removed on
Video 3 (a hand over the far cushion at 62 s had hidden two yellows and the old test let the
reading through).

## The idea

A lump of non-cloth pixels inside the table is judged by its **shape in ball units**, measured
against the ball radius expected at that spot (near balls look bigger than far ones):

| Measure | Balls, clusters, the rack, a blurred moving ball | Obstruction |
|---|---|---|
| thickness = biggest circle that fits in the lump | about 1 radius | arm, hand, body, phone: > 1.5 r (`THICK_RADII`) |
| thin part = what an opening with a 0.6 r disc removes | < 0.3 ball areas (seams) | a cue on the cloth, fingers: > 1 ball area (`THIN_MIN_BALLS`), even when it touches balls |
| edge contact = length of the lump along the table edge | 1 ball at the cushion: ~2 r | a hand reaching in over the cushion: > 4 r and > 3 ball areas (`EDGE_CONTACT_RADII`, `EDGE_MIN_BALLS`) |
| area | 1 to 12 ball areas | anything over 15 (`HUGE_BALLS`) |

Shadows on the cloth (cloth hue, still saturated) count as cloth, so a player's shadow does not
block the reader. No skin colour, no cue model, no knowledge of where the balls are: it works on
a sleeve, a glove, a wooden cue and a black phone alike, in any lighting, and it cannot deadlock
on a ball cluster the way an area threshold can.

`obstruction_examples.jpg` shows the four cases on your own videos.

## Functions (all on the SIGNAL_SCALE frame, using `reference` from `board_reading.make_reference`)

| Function | Replaces | Takes | Gives back |
|---|---|---|---|
| `unexplained_lumps(frame, reference)` | the lump finding inside `view_is_blocked` and `covers_balls` | frame | `lump_mask` (only obstruction lumps), `lumps` (area, thickness, thin part, edge contact, kind) |
| `occluder_mask(lump_mask)` | - | | the lumps grown by `HIDDEN_DILATE_PX` |
| `view_is_blocked(lumps)` | `shot_detection.view_is_blocked` | lumps | True if any obstruction lump exists |
| `new_blocked_filter()`, `update_blocked(filter, blocked_now)` | - (new: temporal smoothing) | | blocked switches on at once, off after `CLEAR_FRAMES` clear frames |
| `hidden_balls(balls, occluder, reference)` | `board_watcher.covers_balls` | tracked balls, occluder | one True/False per ball |
| `new_camera_filter()`, `camera_moved(gray_small, reference, filter)` | `board_watcher.camera_moved` | small grey frame | True while the picture has drifted from the FIRST frame by > `CAMERA_SHIFT_PX` for `CAMERA_SHIFT_FRAMES` frames |

`camera_moved` needs `reference['first_gray_small']` (the first frame, grey, at SIGNAL_SCALE):
add one line to `make_reference`. Comparing with the first frame instead of the previous one
catches a slow drift as well as a bump, and the frame counter stops a passing person faking one.

## Where it plugs in

**`board_watcher.update()` (every frame, before the state machine):**
```python
lump_mask, lumps = obstruction.unexplained_lumps(frame, reference)
blocked = obstruction.update_blocked(self.blocked_filter, obstruction.view_is_blocked(lumps))
moved = obstruction.camera_moved(gray_small, reference, self.camera_filter)
readable = settled and still_count >= HOLD_FRAMES and not blocked and not moved
```

**During a SHOT, before `ball_tracking.track_frame()`** (replaces `covers_balls`, which paused
all tracking when two balls were covered; now each ball is handled on its own):
```python
occluder = obstruction.occluder_mask(lump_mask)
hidden = obstruction.hidden_balls(tracks_as_balls, occluder, reference)   # one flag per track
for track, is_hidden in zip(tracks, hidden):
    track['hidden'] = is_hidden
```

**`ball_tracking`: hidden is not lost.** In the branch where a track is not found this frame:
```python
if track['hidden']:
    coast(track)                     # keep predicting, velocity decays
    continue                         # do NOT count a missed frame, do NOT drop, do NOT call it potted
```
and a pot from the tracker needs the ball to have vanished while **visible**:
```python
potted = track['lost'] and not track['hidden_when_lost'] and near_pocket(track['vanish'])
```
Tracker pots stay hints (which pocket, first contact). The committed-state difference decides.

**`board_reading`: no shot, no count drop.** Before accepting a reading:
```python
if fewer_balls_than(last_committed, reading) and not shot_since_last_commit:
    return None                      # a ball is hidden by something small and static: wait
```

**`board_watcher.shot_result()`: no movement, no shot.** A SHOT episode is only a candidate:
```python
moved = any_ball_moved(before, after, 1.0 * config.BALL_RADIUS_ON_MAP)
if not potted and not moved:
    return None                      # a walk-by, a hand, a cue placed: not a shot, apply_shot not called
```
(cue ball 0 → 1 with nothing else changed = "cue ball replaced", not a shot.)

## Config

Add the block in `config_additions.py`. `OCCLUDER_BIG_FRACTION`, `OCCLUDER_EDGE_FRACTION` and
`TRACK_PAUSE_OCCLUDER_FRACTION` are no longer used.

The other parameters your report lists as missing, with starting values consistent with the
architecture document (check each against how the code uses it; I have not seen
`tracking_helpers.py`):

```python
# board_watcher.py
CAMERA_SHIFT_PX = 3.0             # now in obstruction.py's block
REFIND_AREA_TOLERANCE = 0.15      # re-found table area may differ from the reference by this fraction
REFIND_MAX_CORNER_SHIFT = 40      # px a re-found corner may move from the reference corner
# tracking_helpers.py
STILL_WINDOW_RADII = 1.0          # search window round a ball that has not been freed yet
STILL_WINDOW_MAX_RADII = 3.0      # ... grown to this while it is missing
CUE_MAX_SATURATION = 100          # same meaning as WHITE_MAX_SATURATION
MIN_TRACK_PEAK_RATIO = 0.2        # same meaning as PEAK_MIN_RATIO (sliver allowed)
MAX_TRACK_PEAK_RATIO = 1.3        # a patch fatter than this many radii is not a single ball
TRACK_MAX_SPEED = 2500.0          # map px / s (architecture document)
# ball_tracking.py
CUE_RELEASE_FRAMES = 2            # architecture document
HIT_RADII = 3.0                   # architecture document
# board_reading.py
TABLE_EDGE_TRIM = 15              # px trimmed off the outline at SIGNAL_SCALE (same as before)
```

## Result on the two recordings

| | Video 2 (end view) | Video 3 (side view) |
|---|---|---|
| frames blocked, old / new | 1078 / 1118 | 893 / 1080 |
| frames old said blocked, new says clear | 25 (42.7-43.1 s: checked, the table is clear) | 0 |
| newly caught | the rack at 0 s; a hand at 66 s | the rack; a hand over the cushion at 61-62 s; cue and hand at 80-82 s |
| committed states / events | 7 / unchanged | 5 / unchanged (the false 62 s state is gone) |
| frames with a known ball under an obstruction | 354 | 150 |

The per-frame cost is one cloth mask, one connected-components pass and a distance transform per
lump on the half-size frame: the same order as the old `view_is_blocked`.
