"""
<copyright>
Copyright (c) 2025, Janusch Rentenatus. This program and the accompanying materials are made available under the
terms of the Apache License v2.0 which accompanies this distribution, and is available at
https://github.com/Rentenatus/py_yahtzee?tab=Apache-2.0-1-ov-file#readme
</copyright>
"""


from abc import  abstractmethod
from collections import Counter

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from yahtzee.ml_a_trainer import MlTrainer
from yahtzee.yahtzee_game import YahtzeeGame
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier


class DiceTrainer(MlTrainer):

    @staticmethod
    def extract_dice_training_data(df):
        features = []
        targets = []
        weights = []

        score_categories = YahtzeeGame.score_categories()
        first_print = 17

        for roll_number in [1, 2]:
            roll_prefix = f"roll{roll_number}_dice_"
            stay_prefix = f"roll{roll_number}_stay_"

            for _, row in df.iterrows():
                if not all(pd.notna(row.get(f"{roll_prefix}{i}")) for i in range(1, 6)):
                    continue
                dice_values = sorted([int (row[f"{roll_prefix}{i}"]) for i in range(1, 6)])
                dice_copy =dice_values.copy()
                counter = Counter(dice_values)
                counts = [counter.get(i, 0) for i in range(1, 7)]
                stay_values = [row.get(f"{stay_prefix}{i}") for i in range(1, 6)]

                y = [0] * 5
                for w in stay_values:
                    if w is None:
                        continue
                    try:
                        pos = dice_copy.index(w)
                        y[pos] = 1
                        dice_copy[pos] = -9966  # verhindert Duplikat-Erkennung
                    except ValueError:
                        pass  # falls Wert nicht mehr vorhanden ist

                score_values = [
                    -9 if pd.isna(row.get(f"score_{cat}_before",-9)) else row.get(f"score_{cat}_before",-9)
                    for cat in score_categories
                ]
                x = dice_values + counts + score_values + [roll_number]
                gradient = row.get("gradient_score", 1.0)  # Default: neutrale Lernstärke
                gradient = max(0.92, gradient)
                if first_print >=0:
                    print(dice_values, stay_values, x , y)
                    first_print -= 1
                features.append(x)
                targets.append(y)
                weights.append(gradient)

        x_columns = [f"dice_{i}" for i in range(1, 6)] + \
                    [f"count{i}" for i in range(1, 7)] + \
                    [f"score_{cat}_before" for cat in score_categories] + ["roll_number"]
        y_columns = [f"choosen_{i}" for i in range(1, 6)]

        x_data = pd.DataFrame(features, columns=x_columns)
        y_data = pd.DataFrame(targets, columns=y_columns)
        return x_data, y_data, weights

    def extract_dice_predict_data(self, game: YahtzeeGame, roll_number: int):
        # Extrahiere aktuelle Würfel
        dice = sorted(game.dice)  # z.B. [2, 5, 5, 1, 6] -> [1, 2, 5, 5, 6]
        counter = Counter(dice)
        counts = [counter.get(i, 0) for i in range(1, 7)]

        # Extrahiere Scorecard-Werte
        score_features = [
            -9 if pd.isna(game.scorecard.get(cat, -9)) else game.scorecard.get(cat, -9)
            for cat in YahtzeeGame.score_categories()
        ]

        # Kombiniere Features
        feature_vector = dice + counts + score_features + [roll_number]

        # Spaltennamen müssen zum Training passen
        columns = [f"dice_{i}" for i in range(1, 6)] + \
                  [f"count{i}" for i in range(1, 7)] + \
                  [f"score_{cat}_before" for cat in YahtzeeGame.score_categories()] + ["roll_number"]

        df = pd.DataFrame([feature_vector], columns=columns)
        return dice, df

    @abstractmethod
    def train_model(self, x_data, y_data, w_train):
        pass

    @abstractmethod
    def evaluate_model(self, x_data, y_data):
        pass

    @abstractmethod
    def predict_from_game(self, game: YahtzeeGame, roll_number: int):
        pass

class DiceTrainerRandomForest(DiceTrainer):
    def __init__(self):
        self.model = None

    def train_model(self, x_data, y_data, w_train):
        base_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model = MultiOutputClassifier(base_model)
        self.model.fit(x_data, y_data)
        print("✅ DiceModel erfolgreich trainiert")

    def evaluate_model(self, x_test, y_test):
        if self.model is None:
            print("⚠️ Kein Modell vorhanden")
            return
        score = self.model.score(x_test, y_test)
        print(f"📊 Genauigkeit: {score:.3f}")
        return score

    def predict_from_game(self, game: YahtzeeGame, roll_number: int):
        if self.model is None:
            raise ValueError("Modell ist nicht trainiert")
        extract_dice, df = self.extract_dice_predict_data(game, roll_number)
        prediction = self.model.predict(df)[0]
        return extract_dice, prediction



    def save_model(self, path):
        import joblib
        joblib.dump(self.model, path)
        print(f"💾 Modell gespeichert unter {path}")

    def load_model(self, path):
        import joblib
        self.model = joblib.load(path)
        print(f"📂 Modell geladen aus {path}")


class DiceTrainerNN(DiceTrainer):
    def __init__(self):
        self.model = None

    def train_model(self, x_data, y_data, w_train):
        base_model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42
        )
        self.model = MultiOutputClassifier(base_model)
        self.model.fit(x_data, y_data, sample_weight=w_train)
        print("🧠 NN DiceModel erfolgreich trainiert")

    def evaluate_model(self, x_test, y_test):
        if self.model is None:
            print("⚠️ Kein Modell vorhanden")
            return
        score = self.model.score(x_test, y_test)
        print(f"📊 Genauigkeit: {score:.3f}")
        return score

    def predict_from_game(self, game: YahtzeeGame, roll_number: int):
        if self.model is None:
            raise ValueError("Modell ist nicht trainiert")

        extract_dice, df = self.extract_dice_predict_data(game, roll_number)

        # Optional: Schwellenwertlogik mit Wahrscheinlichkeiten
        probas = self.model.predict_proba(df)
        keep = [ probas[i][0][1]  for i in range(5) ]  # Wahrscheinlichkeit für Klasse '1' (Behalten)
        return  extract_dice, keep

    def save_model(self, path):
        import joblib
        joblib.dump(self.model, path)
        print(f"💾 NN-Modell gespeichert unter {path}")

    def load_model(self, path):
        import joblib
        self.model = joblib.load(path)
        print(f"📂 NN-Modell geladen aus {path}")
