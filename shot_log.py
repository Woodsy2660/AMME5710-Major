"""
Saves what the system decided for every shot, so it can be compared with a hand-made
ground-truth sheet afterwards (scoring / foul accuracy), plus how long each frame took
to process (frame rate results).

Files written (folder given with gui.py --log, e.g. --log logs):
    <name>_shots.csv    one row per shot the system scored, and one per manual correction
    <name>_timing.csv   one row per processed frame: frame number, processing time (ms)
    <name>_summary.txt  frames processed, mean / median / 95th percentile time, frame rate

<name> is the video's file name (e.g. IMG_7429) or 'live_<date>_<time>' for the camera.
"""

import csv
import os
import time

import numpy as np


SHOT_FIELDS = ['row', 'type', 'shot', 'frame', 'time_s', 'shooter',
               'red', 'yellow', 'black', 'cue', 'pockets',
               'first_hit', 'tracked', 'balls_moved', 'fouls', 'turn_kept', 'winner']


class ShotLog:

    def __init__(self, folder, source_name):
        os.makedirs(folder, exist_ok=True)
        base = os.path.join(folder, source_name)
        # Never overwrite an earlier run: add _2, _3 ... to the name.
        n = 1
        while os.path.exists(f"{base}_shots.csv" if n == 1 else f"{base}_{n}_shots.csv"):
            n = n + 1
        if n > 1:
            base = f"{base}_{n}"
        self.shots_path = f"{base}_shots.csv"
        self.timing_path = f"{base}_timing.csv"
        self.summary_path = f"{base}_summary.txt"
        self.rows = 0
        self.frame_ms = []

        self.shots_file = open(self.shots_path, 'w', newline='')
        self.shots = csv.writer(self.shots_file)
        self.shots.writerow(SHOT_FIELDS)
        self.timing_file = open(self.timing_path, 'w', newline='')
        self.timing = csv.writer(self.timing_file)
        self.timing.writerow(['frame', 'process_ms'])

    def shot(self, game, event, result, shooter, frame_number, time_s):
        """
        One shot, straight after game.apply_shot().
        - game : the GameState (after the shot)
        - event : what apply_shot() returned (None if the game had already ended)
        - result : the dict from BoardWatcher.update()
        - shooter : index of the player who took the shot
        """
        potted = result['potted']
        fouls = event['fouls'] if event else []
        winner = event['winner'] if event else None
        self._write(['shot', game.shot_number, frame_number, round(time_s, 2),
                     game.players[shooter],
                     potted['red'], potted['yellow'], int(potted['black']), int(potted['cue']),
                     result['note'],
                     result['first_hit'] if result['first_hit'] else '',
                     int(result['tracked']), int(result['balls_moved']),
                     '; '.join(fouls),
                     int(event['turn_kept']) if event else '',
                     game.players[winner] if winner is not None else ''])

    def correction(self, action, game, frame_number, time_s):
        """A referee correction (key press or --correct), so corrected shots can be found."""
        self._write([f"correction:{action}", game.shot_number, frame_number, round(time_s, 2)]
                    + [''] * (len(SHOT_FIELDS) - 5))

    def frame(self, frame_number, seconds):
        ms = seconds * 1000.0
        self.frame_ms.append(ms)
        self.timing.writerow([frame_number, round(ms, 2)])

    def _write(self, values):
        self.rows = self.rows + 1
        self.shots.writerow([self.rows] + values)
        self.shots_file.flush()        # written straight away, so a crash or Ctrl-C keeps the rows

    def close(self):
        self.shots_file.close()
        self.timing_file.close()
        with open(self.summary_path, 'w') as file:
            if self.frame_ms:
                ms = np.array(self.frame_ms)
                file.write(f"frames processed: {len(ms)}\n"
                           f"mean ms/frame: {ms.mean():.1f}\n"
                           f"median ms/frame: {np.median(ms):.1f}\n"
                           f"95th percentile ms/frame: {np.percentile(ms, 95):.1f}\n"
                           f"max ms/frame: {ms.max():.1f}\n"
                           f"processing rate (fps, from mean): {1000.0 / ms.mean():.1f}\n")
            else:
                file.write("no frames processed\n")
        return self.shots_path
