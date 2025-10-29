"""
<copyright>
Copyright (c) 2025, Janusch Rentenatus. This program and the accompanying materials are made available under the
terms of the Apache License v2.0 which accompanies this distribution, and is available at
https://github.com/Rentenatus/py_yahtzee?tab=Apache-2.0-1-ov-file#readme
</copyright>

"""

import unittest

from yahtzee.ml_dice_trainer import DiceTrainerRandomForest
from yahtzee.ml_rating_trainer import RatingTrainerRandomForest
from yahtzee.yahtzee_game import YahtzeeGame
from yahtzee.yahtzee_ml_player import ModelPlayer
from yahtzee.yahtzee_player import AlwaysKeepPlayer, ChaosPlayer
from yahtzee.yahtzee_scheduler import Scheduler


class TestYahtzeeGame(unittest.TestCase):
    """
    Unit test class for YahtzeeGame.
    """

    def setUp(self):
        self.game = YahtzeeGame()


    def test_scheduler(self):
        model1_path = "assets/models/dice_model_rf.pkl"
        model2_path = "assets/models/rating_model_rf.pkl"
        dice_model = DiceTrainerRandomForest()
        dice_model.load_model(model1_path)
        rating_model = RatingTrainerRandomForest()
        rating_model.load_model(model2_path)
        players = [ChaosPlayer("ChaosBot"), ModelPlayer("RandomForestBot", dice_model, rating_model)]
        scheduler = Scheduler(players)
        scheduler.start()
