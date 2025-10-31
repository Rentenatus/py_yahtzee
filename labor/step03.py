"""
<copyright>
Copyright (c) 2025, Janusch Rentenatus. This program and the accompanying materials are made available under the
terms of the Apache License v2.0 which accompanies this distribution, and is available at
https://github.com/Rentenatus/py_yahtzee?tab=Apache-2.0-1-ov-file#readme
</copyright>
"""

from labor.step01 import run_batch_games, splitt_and_save
from labor.step02v_validate_ml_game import nn_diff_player, reandom_forest_player
from yahtzee.ml_dice_trainer import  DiceTrainerRandomForest, DiceTrainerNN
from yahtzee.ml_rating_trainer import  RatingTrainerRandomForest, RatingTrainerNN
from yahtzee.yahtzee_logger import PandasLogger
from yahtzee.yahtzee_ml_player import ModelPlayer
from yahtzee.yahtzee_player import ChaosPlayer


def main():


    players = [
        ChaosPlayer("ChaosBot"),
        (reandom_forest_player()),
        (nn_diff_player())
    ]

    df = run_batch_games(players, logger_class=PandasLogger, num_games=28000)

    filename1 = "assets/yahtzee_training_data_11.csv"
    filename2 = "assets/yahtzee_training_data_12.csv"

    splitt_and_save(df, filename1, filename2)

if __name__ == "__main__":
    main()
