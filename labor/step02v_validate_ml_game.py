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
        dice_model_rf = DiceTrainerRandomForest()
        dice_model_rf.load_model(model1_path)
        rating_model_rf = RatingTrainerRandomForest()
        rating_model_rf.load_model(model2_path)

        model3_path = "assets/models/dice_model_nn.pkl"
        model4_path = "assets/models/rating_model_nn.pkl"
        dice_model_nn = DiceTrainerRandomForest()
        dice_model_nn.load_model(model3_path)
        rating_model_nn = RatingTrainerRandomForest()
        rating_model_nn.load_model(model4_path)

        players = [
            ModelPlayer("RandomForestBot", dice_model_rf, rating_model_rf),
            ModelPlayer("NeuralNetworkBot", dice_model_nn, rating_model_nn)
        ]
        scheduler = Scheduler(players)
        scheduler.start()
