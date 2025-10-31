"""
<copyright>
Copyright (c) 2025, Janusch Rentenatus. This program and the accompanying materials are made available under the
terms of the Apache License v2.0 which accompanies this distribution, and is available at
https://github.com/Rentenatus/py_yahtzee?tab=Apache-2.0-1-ov-file#readme
</copyright>

"""
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
from yahtzee.ml_rating_trainer import RatingTrainerRandomForest, RatingTrainer, RatingTrainerNN
from yahtzee.ml_dice_trainer import DiceTrainerRandomForest, DiceTrainer, DiceTrainerNN
import numpy as np


def main():
    # 🔹 Schritt 1: Daten laden
    filename = "assets/yahtzee_training_data_01.csv"
    df = pd.read_csv(filename)
    print(f"📁 Daten geladen aus {filename} mit {len(df)} Zeilen")

    train_dice(df, DiceTrainerRandomForest(),"assets/models/dice_model_rf.pkl")
    train_dice(df, DiceTrainerNN(),"assets/models/dice_model_nn.pkl")
    train_rating(df, RatingTrainerRandomForest(), "assets/models/rating_model_rf.pkl")
    train_rating(df, RatingTrainerNN(), "assets/models/rating_model_nn.pkl")


def train_dice(df: pd.DataFrame, trainer: DiceTrainer, model_path):
    # 🔹 Schritt 2: Trainingsdaten extrahieren
    x_data, y_data, weights = trainer.extract_dice_training_data(df)

    trainer.print_train_data(x_data, y_data)
    trainer.check_train_data(x_data, y_data, weights)
    train_and_save(model_path, trainer, weights, x_data, y_data)

def train_rating(df: pd.DataFrame, trainer: RatingTrainer, model_path):
    # 🔹 Schritt 2: Trainingsdaten extrahieren
    x_data, y_data, weights = trainer.extract_rating_training_data(df)

    trainer.print_train_data(x_data, y_data)
    trainer.check_train_data(x_data, y_data, weights)
    train_and_save(model_path, trainer, weights, x_data, y_data)



def train_and_save(model_path, trainer, weights, x_data, y_data):
    # 🔹 Schritt 3: Split & Training
    x_train, x_test, y_train, y_test, w_train, w_test = train_test_split(
        x_data, y_data, weights, test_size=0.1, random_state=42
    )

    trainer.train_model(x_train, y_train, w_train)

    # 🔹 Schritt 4: Evaluation
    trainer.evaluate_model(x_test, y_test)

    # 🔹 Schritt 5: Modell speichern
    trainer.save_model(model_path)



if __name__ == "__main__":
    main()
