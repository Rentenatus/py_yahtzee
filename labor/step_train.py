"""
<copyright>
Copyright (c) 2025, Janusch Rentenatus. This program and the accompanying materials are made available under the
terms of the Apache License v2.0 which accompanies this distribution, and is available at
https://github.com/Rentenatus/py_yahtzee?tab=Apache-2.0-1-ov-file#readme
</copyright>

"""

import pandas as pd
from sklearn.model_selection import train_test_split

from yahtzee.yahtzee_game import YahtzeeGame
from yahtzee.ml_dice_trainer import DiceTrainerRandomForest

def main():
    # 🔹 Schritt 1: Daten laden
    filename = "assets/yahtzee_training_data_01.csv"
    df = pd.read_csv(filename)
    print(f"📁 Daten geladen aus {filename} mit {len(df)} Zeilen")

    # 🔹 Schritt 2: Trainingsdaten extrahieren
    trainer = DiceTrainerRandomForest()
    x_data, y_data = trainer.extract_dice_training_data(df)

    print(f"📊 Trainingsbeispiele: {len(x_data)}")
    print(f"📐 Feature-Spalten: {list(x_data.columns)}")
    print(f"🎯 Ziel-Spalten: {list(y_data.columns)}")

    print(x_data.head())
    print(y_data.head())

    # 🔹 Schritt 3: Split & Training
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=42)
    trainer.train_model(x_train, y_train)

    # 🔹 Schritt 4: Evaluation
    trainer.evaluate_model(x_test, y_test)

    # 🔹 Schritt 5: Modell speichern
    model_path = "assets/models/dice_model_rf.pkl"
    trainer.save_model(model_path)

if __name__ == "__main__":
    main()
