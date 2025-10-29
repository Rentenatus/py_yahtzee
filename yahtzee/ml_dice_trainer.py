from abc import ABC, abstractmethod

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

from yahtzee.yahtzee_game import YahtzeeGame


class DiceTrainer(ABC):

    @staticmethod
    def extract_dice_training_data(df):
        features = []
        targets = []

        score_categories = YahtzeeGame.score_categories()

        for roll_number in [1, 2]:
            roll_prefix = f"roll{roll_number}_dice_"
            stay_prefix = f"roll{roll_number}_stay_"

            for _, row in df.iterrows():
                if not all(pd.notna(row.get(f"{roll_prefix}{i}")) for i in range(1, 6)):
                    continue
                dice_values = [row[f"{roll_prefix}{i}"] for i in range(1, 6)]
                dice_copy = dice_values.copy()
                stay_values = [row.get(f"{stay_prefix}{i}") for i in range(1, 6)]

                y = [0] * 5
                for w in stay_values:
                    if w is None:
                        continue
                    try:
                        pos = dice_copy.index(w)
                        y[pos] = 1
                        dice_copy[pos] = -99  # verhindert Duplikat-Erkennung
                    except ValueError:
                        pass  # falls Wert nicht mehr vorhanden ist

                score_values = [row.get(f"score_{cat}_before", 0) for cat in score_categories]
                x = dice_values + score_values + [roll_number]
                features.append(x)
                targets.append(y)

        x_columns = [f"dice_{i}" for i in range(1, 6)] + \
                    [f"score_{cat}_before" for cat in score_categories] + ["roll_number"]
        y_columns = [f"choosen_{i}" for i in range(1, 6)]

        x_data = pd.DataFrame(features, columns=x_columns)
        y_data = pd.DataFrame(targets, columns=y_columns)
        return x_data, y_data

    @abstractmethod
    def train_model(self, x_data, y_data):
        pass

    @abstractmethod
    def evaluate_model(self, x_data, y_data):
        pass

    @abstractmethod
    def save_model(self, path):
       pass

    @abstractmethod
    def load_model(self, path):
         pass

class DiceTrainerRandomForest(DiceTrainer):
    def __init__(self):
        self.model = None

    def train_model(self, x_data, y_data):
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

        # Extrahiere aktuelle Würfel
        dice = game.dice  # z.B. [2, 5, 5, 1, 6]

        # Extrahiere Scorecard-Werte
        score_features = [game.scorecard.get(cat, 0) for cat in YahtzeeGame.score_categories()]

        # Kombiniere Features
        feature_vector = dice + score_features + [roll_number]

        # Spaltennamen müssen zum Training passen
        columns = [f"dice_{i}" for i in range(1, 6)] + \
                  [f"score_{cat}_before" for cat in YahtzeeGame.score_categories()] + ["roll_number"]

        df = pd.DataFrame([feature_vector], columns=columns)
        prediction = self.model.predict(df)[0]
        return prediction  # z.B. [1, 0, 1, 1, 0]

    def save_model(self, path):
        import joblib
        joblib.dump(self.model, path)
        print(f"💾 Modell gespeichert unter {path}")

    def load_model(self, path):
        import joblib
        self.model = joblib.load(path)
        print(f"📂 Modell geladen aus {path}")
