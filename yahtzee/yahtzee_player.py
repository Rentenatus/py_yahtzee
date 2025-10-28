"""
<copyright>
Copyright (c) 2025, Janusch Rentenatus. This program and the accompanying materials are made available under the
terms of the Apache License v2.0 which accompanies this distribution, and is available at
https://github.com/Rentenatus/py_yahtzee?tab=Apache-2.0-1-ov-file#readme
</copyright>

"""

from abc import ABC, abstractmethod


class Player(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def choose_dice(self, game, roll_number):
        """
        Decide which dice to keep based on the current game state and roll number.

        Parameters:
            game (YahtzeeGame): The current game instance.
            roll_number (int): The current roll (1, 2, or 3).

        Returns:
            list[int]: A subarray of dice values to keep.
        """
        pass

    @abstractmethod
    def choose_rating(self, game, roll_number):
        """
        Decide which category to score based on the current game state and roll number.

        Parameters:
            game (YahtzeeGame): The current game instance.
            roll_number (int): The current roll (1, 2, or 3).

        Returns:
            str: The name of the category to score.
        """
        pass



class AlwaysKeepPlayer(Player):
    def __init__(self, name):
        super().__init__(name)

    def choose_dice(self, game, roll_number):
        return game.dice  # keep all dice

    def choose_rating(self, game, roll_number):
        open_scores = game.open_score()
        if not open_scores:
            return None
        # Choose category with highest potential score
        return max(open_scores.items(), key=lambda x: x[1])[0]
