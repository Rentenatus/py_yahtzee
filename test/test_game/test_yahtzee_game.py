import os
import unittest

from yahtzee.yahtzee_game import YahtzeeGame


class TestYahtzeeGame(unittest.TestCase):
    """
    Unit test class for YahtzeeGame.
    """

    def setUp(self):
        self.game = YahtzeeGame()

    def test_roll_and_score(self):
        print("First roll:", self.game.roll_dice())
        for meth in self.game._score_functions().keys():
            self.game.score_category(meth)
        self.game.print_scorecard()
