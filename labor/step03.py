"""
<copyright>
Copyright (c) 2025, Janusch Rentenatus. This program and the accompanying materials are made available under the
terms of the Apache License v2.0 which accompanies this distribution, and is available at
https://github.com/Rentenatus/py_yahtzee?tab=Apache-2.0-1-ov-file#readme
</copyright>

"""

import pandas as pd

from labor.step01 import run_batch_games
from yahtzee.ml_dice_trainer import DiceTrainer, DiceTrainerRandomForest, DiceTrainerNN
from yahtzee.ml_rating_trainer import RatingTrainer, RatingTrainerRandomForest, RatingTrainerNN
from yahtzee.yahtzee_logger import PandasLogger
from yahtzee.yahtzee_ml_player import ModelPlayer
from yahtzee.yahtzee_player import ChaosPlayer, AlwaysKeepPlayer
from yahtzee.yahtzee_scheduler import Scheduler





def main():
    model1_path = "assets/models/dice_model_rf.pkl"
    model2_path = "assets/models/rating_model_rf.pkl"
    dice_model_rf = DiceTrainerRandomForest()
    dice_model_rf.load_model(model1_path)
    rating_model_rf = RatingTrainerRandomForest()
    rating_model_rf.load_model(model2_path)

    model3_path = "assets/models/dice_model_nn.pkl"
    model4_path = "assets/models/rating_model_nn.pkl"
    dice_model_nn = DiceTrainerNN()
    dice_model_nn.load_model(model3_path)
    rating_model_nn = RatingTrainerNN()
    rating_model_nn.load_model(model4_path)

    players = [
        ChaosPlayer("ChaosBot"),
        ModelPlayer("RandomForestBot", dice_model_rf, rating_model_rf),
        ModelPlayer("NeuralNetworkBot", dice_model_nn, rating_model_nn)
    ]

    df = run_batch_games(players, logger_class=PandasLogger, num_games=1000)

    if "total_score" in df.columns:
        # Globaler Schwellenwert
        threshold = df["total_score"].quantile(1 - 0.1)  # Top 10%
        df = df[df["total_score"] >= threshold]

    print(f"✅ Batch abgeschlossen: {len(df)} Zeilen")
    print(df.head())

    filename = "assets/yahtzee_training_data_02.csv"
    df.to_csv(filename, index=False)
    print(f"📁 Daten gespeichert unter {filename}")

if __name__ == "__main__":
    main()
