"""
Step 7: keep score. Applies the rules of blackball pool to what was potted on each shot.

- Two players, 7 reds, 7 yellows, 1 black, 1 cue ball
- Table is open until a player pots only one colour; that colour becomes theirs
- Potting one of your own balls (or any ball on an open table) keeps your turn
- Fouls: potting the cue ball, potting the opponent's colour
- After a foul the other player gets two visits (config.FOUL_GIVES_TWO_VISITS)
- Black: potting it after clearing your colour wins, potting it any other way loses

This file knows nothing about images. It is given a 'potted' record for each shot
(from board_watcher.py, or typed in by hand for testing).
"""

import config

RED = 'red'
YELLOW = 'yellow'


def other_colour(colour):
    if colour == RED:
        return YELLOW
    return RED


def empty_potted():
    """
    A record of nothing potted. Fill it in to describe a shot.
    - red, yellow : how many of each went down
    - black, cue : True if that ball went down
    """
    return {'red': 0, 'yellow': 0, 'black': False, 'cue': False}


class GameState:

    def __init__(self, player_names=('Player 1', 'Player 2')):
        self.players = list(player_names)
        self.current = 0                       # index of the player at the table
        self.visits_left = 1                   # 2 after the opponent fouls
        self.groups = [None, None]             # 'red' or 'yellow' once the table is decided
        self.remaining = {RED: config.BALLS_PER_COLOUR, YELLOW: config.BALLS_PER_COLOUR}
        self.fouls = [0, 0]
        self.winner = None                     # index of the winning player
        self.shot_number = 0
        self.log = []                          # one line of text per shot, for the GUI

    def balls_left(self, player):
        """How many of their own balls a player still has to pot (None if table is open)."""
        if self.groups[player] is None:
            return None
        return self.remaining[self.groups[player]]

    def balls_potted(self, player):
        """How many of their own balls a player has potted so far."""
        if self.groups[player] is None:
            return 0
        return config.BALLS_PER_COLOUR - self.remaining[self.groups[player]]

    def apply_shot(self, potted, note=''):
        """
        Inputs
        - potted : record from empty_potted() describing one shot
        - note : optional extra text for the log, e.g. which pockets
        Output
        - event : dict with what happened, also added to self.log as text
            player, potted, fouls (list of reasons), turn_kept, winner
        """

        if self.winner is not None:
            return None

        self.shot_number = self.shot_number + 1
        shooter = self.current
        opponent = 1 - shooter
        own = self.groups[shooter]

        # Was the shooter already on the black before this shot?
        on_the_black = own is not None and self.remaining[own] == 0

        # Take the potted balls off the table. Never below zero, in case the
        # detector reports more balls than are left.
        red_down = min(potted['red'], self.remaining[RED])
        yellow_down = min(potted['yellow'], self.remaining[YELLOW])
        self.remaining[RED] = self.remaining[RED] - red_down
        self.remaining[YELLOW] = self.remaining[YELLOW] - yellow_down
        down = {RED: red_down, YELLOW: yellow_down}

        fouls = []
        if potted['cue']:
            fouls.append('cue ball potted')
        if own is not None and down[other_colour(own)] > 0:
            fouls.append('potted opponent ball')

        event = {'player': shooter, 'potted': potted, 'fouls': fouls,
                 'turn_kept': False, 'winner': None}

        # The black ends the game either way.
        if potted['black']:
            if on_the_black and not fouls:
                self.winner = shooter
                reason = 'potted the black'
            else:
                self.winner = opponent
                reason = 'black potted early' if not on_the_black else 'foul on the black'
                event['fouls'].append(reason)
            event['winner'] = self.winner
            self.log.append(f"Shot {self.shot_number}: {self.players[shooter]} {reason} "
                            f"- {self.players[self.winner]} wins")
            return event

        # Open table: potting only one colour (without a foul) claims it.
        if own is None and not fouls:
            if red_down > 0 and yellow_down == 0:
                own = RED
            elif yellow_down > 0 and red_down == 0:
                own = YELLOW
            if own is not None:
                self.groups[shooter] = own
                self.groups[opponent] = other_colour(own)

        # Decide who plays next.
        if fouls:
            self.fouls[shooter] = self.fouls[shooter] + 1
            self.current = opponent
            self.visits_left = 2 if config.FOUL_GIVES_TWO_VISITS else 1
        else:
            if own is None:
                potted_something = red_down + yellow_down > 0
            else:
                potted_something = down[own] > 0

            if potted_something:
                event['turn_kept'] = True
            elif self.visits_left > 1:
                self.visits_left = self.visits_left - 1    # use up the free visit
                event['turn_kept'] = True
            else:
                self.current = opponent
                self.visits_left = 1

        self.log.append(self.describe_shot(shooter, down, fouls, note))
        return event

    def describe_shot(self, shooter, down, fouls, note=''):
        """One line of text describing a shot, for the GUI's event log."""
        parts = []
        if down[RED]:
            parts.append(f"{down[RED]} red")
        if down[YELLOW]:
            parts.append(f"{down[YELLOW]} yellow")
        text = f"Shot {self.shot_number}: {self.players[shooter]} potted "
        text = text + (", ".join(parts) if parts else "nothing")
        if note:
            text = text + f" ({note})"
        if fouls:
            text = text + " - FOUL (" + ", ".join(fouls) + ")"
        return text
