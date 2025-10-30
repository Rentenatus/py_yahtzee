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

    print_train_data(x_data, y_data)
    check_train_data(x_data, y_data, weights)
    train_and_save(model_path, trainer, weights, x_data, y_data)

def train_rating(df: pd.DataFrame, trainer: RatingTrainer, model_path):
    # 🔹 Schritt 2: Trainingsdaten extrahieren
    x_data, y_data, weights = trainer.extract_rating_training_data(df)

    print_train_data(x_data, y_data)
    check_train_data(x_data, y_data, weights)
    train_and_save(model_path, trainer, weights, x_data, y_data)


def print_train_data(x_data, y_data):
    print(f"📊 Trainingsbeispiele: {len(x_data)}")
    print(f"📐 Feature-Spalten: {list(x_data.columns)}")
    print(f"🎯 Ziel-Spalten: {list(y_data.columns)}")

    print(x_data.head())
    print(y_data.head())

def check_train_data(x_data, y_data, weights):
    # Prüfe auf NaN oder None in x_data
    if x_data.isnull().values.any():
        bad_rows = x_data[x_data.isnull().any(axis=1)]
        raise ValueError(f"❌ x_data enthält NaN/None in Zeilen:\n{bad_rows}")

    # Prüfe auf NaN oder None in y_data
    if y_data.isnull().values.any():
        bad_rows = y_data[y_data.isnull().any(axis=1)]
        raise ValueError(f"❌ y_data enthält NaN/None in Zeilen:\n{bad_rows}")

    # Prüfe auf NaN oder None in weights
    if any(w is None or (isinstance(w, float) and np.isnan(w)) for w in weights):
        bad_indices = [i for i, w in enumerate(weights) if w is None or (isinstance(w, float) and np.isnan(w))]
        raise ValueError(f"❌ weights enthält NaN/None an Positionen: {bad_indices}")

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
