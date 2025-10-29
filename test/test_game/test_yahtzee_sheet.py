"""
<copyright>
Copyright (c) 2025, Janusch Rentenatus. This program and the accompanying materials are made available under the
terms of the Apache License v2.0 which accompanies this distribution, and is available at
https://github.com/Rentenatus/py_yahtzee?tab=Apache-2.0-1-ov-file#readme
</copyright>

"""

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
        for meth in self.game.score_categories():
            self.game.score_category(meth)
        self.game.print_scorecard()

