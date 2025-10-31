"""
<copyright>
Copyright (c) 2025, Janusch Rentenatus. This program and the accompanying materials are made available under the
terms of the Apache License v2.0 which accompanies this distribution, and is available at
https://github.com/Rentenatus/py_yahtzee?tab=Apache-2.0-1-ov-file#readme
</copyright>
"""

import unittest

from yahtzee.yahtzee_game import YahtzeeGame
from yahtzee.yahtzee_player import AlwaysKeepPlayer, ChaosPlayer
from yahtzee.yahtzee_scheduler import Scheduler


class TestYahtzeeGame(unittest.TestCase):
    """
    Unit test class for YahtzeeGame.
    """

    def setUp(self):
        self.game = YahtzeeGame()


    def test_scheduler(self):
        players = [AlwaysKeepPlayer("AlwaysKeepBot"), ChaosPlayer("ChaosBot")]
        scheduler = Scheduler(players)
        scheduler.start()
