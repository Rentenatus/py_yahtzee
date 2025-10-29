"""
<copyright>
Copyright (c) 2025, Janusch Rentenatus. This program and the accompanying materials are made available under the
terms of the Apache License v2.0 which accompanies this distribution, and is available at
https://github.com/Rentenatus/py_yahtzee?tab=Apache-2.0-1-ov-file#readme
</copyright>

"""

from abc import ABC, abstractmethod
import random
from collections import Counter

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
            return open_scores, None
        # Choose category with highest potential score
        return open_scores, max(open_scores.items(), key=lambda x: x[1])[0]



class ChaosPlayer(Player):
    def __init__(self, name):
        super().__init__(name)

    def choose_dice(self, game, roll_number):
        dice = game.dice
        counts = Counter(dice)
        # Priorität: mindestens drei gleiche → obere Kategorie
        for val, count in counts.items():
            if count >= 2:
                category = self._upper_category(val)
                if category and game.scorecard.get(category) is None:
                    return [d for d in dice if d == val]

        # Sonst: zufällige Auswahl
        if roll_number == 1:
            return random.sample(dice, 2)
        elif roll_number == 2:
            return random.sample(dice, 3)
        else:
            return dice


    def choose_rating(self, game, roll_number):
        open_scores = game.open_score()
        if not open_scores:
            return open_scores, None

        # Trenne obere und untere Kategorien
        upper = {"ones", "twos", "threes", "fours", "fives", "sixes"}
        lower = {
            "three_of_a_kind", "four_of_a_kind", "full_house",
            "small_straight", "large_straight", "yahtzee", "chance"
        }

        # Priorität: mindestens drei gleiche → obere Kategorie
        counts = Counter(game.dice)
        for val, count in counts.items():
            if count >= 3:
                category = self._upper_category(val)
                if category and game.scorecard.get(category) is None:
                    return open_scores, category

        # Finde beste untere Kategorie
        lower_scores = {k: v for k, v in open_scores.items() if k in lower}
        if lower_scores:
            best_lower = max(lower_scores.items(), key=lambda x: x[1])
            if best_lower[1] >= 8:
                return open_scores, best_lower[0]

        # Sonst: beste obere Kategorie
        upper_scores = {k: v for k, v in open_scores.items() if k in upper}
        if upper_scores:
            return open_scores, max(upper_scores.items(), key=lambda x: x[1])[0]

        # Fallback: beste Kategorie insgesamt
        return open_scores, max(open_scores.items(), key=lambda x: x[1])[0]

    def _upper_category(self, value):
        mapping = {
            1: "ones", 2: "twos", 3: "threes",
            4: "fours", 5: "fives", 6: "sixes"
        }
        return mapping.get(value)

