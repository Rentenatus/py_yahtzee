from abc import ABC, abstractmethod
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from yahtzee.yahtzee_game import YahtzeeGame

class RatingTrainer(ABC):

    @staticmethod
    def extract_rating_training_data(df):
        features = []
        targets = []

        score_categories = YahtzeeGame.score_categories()

        for _, row in df.iterrows():
            if pd.isna(row.get("category")):
                continue

            # Eingabefeatures: score_before + score_simulated + roll_number
            score_before = [row.get(f"score_{cat}_before", 0) for cat in score_categories]
            score_simulated = [row.get(f"score_{cat}_simulated", 0) for cat in score_categories]
            roll_number = row.get("round", 1)

            x = score_before + score_simulated + [roll_number]

            # Ziel: flache One-Hot-Kodierung der gewählten Kategorie
            y = [0] * len(score_categories)
            try:
                index = score_categories.index(row["category"])
                y[index] = 1
            except ValueError:
                continue  # ungültige Kategorie überspringen

            features.append(x)
            targets.append(y)

        x_columns = [f"score_{cat}_before" for cat in score_categories] + \
                    [f"score_{cat}_simulated" for cat in score_categories] + ["roll_number"]
        y_columns = [f"chosen_{cat}" for cat in score_categories]

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

class RatingTrainerRandomForest(RatingTrainer):
    def __init__(self):
        self.model = None

    def train_model(self, x_data, y_data):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(x_data, y_data.values.argmax(axis=1))  # Klassenziel: Index der Kategorie
        print("✅ RatingModel erfolgreich trainiert")

    def evaluate_model(self, x_test, y_test):
        if self.model is None:
            print("⚠️ Kein Modell vorhanden")
            return
        score = self.model.score(x_test, y_test.values.argmax(axis=1))
        print(f"📊 Genauigkeit: {score:.3f}")
        return score

    def predict_from_game(self, game: YahtzeeGame, simulated_scores: dict, roll_number: int):
        if self.model is None:
            raise ValueError("Modell ist nicht trainiert")

        score_categories = YahtzeeGame.score_categories()
        score_before = [game.scorecard.get(cat, 0) for cat in score_categories]
        score_simulated = [simulated_scores.get(cat, 0) for cat in score_categories]

        x = score_before + score_simulated + [roll_number]
        columns = [f"score_{cat}_before" for cat in score_categories] + \
                  [f"score_{cat}_simulated" for cat in score_categories] + ["roll_number"]

        df = pd.DataFrame([x], columns=columns)
        predicted_index = self.model.predict(df)[0]
        return score_categories[predicted_index]

    def save_model(self, path):
        import joblib
        joblib.dump(self.model, path)
        print(f"💾 Modell gespeichert unter {path}")

    def load_model(self, path):
        import joblib
        self.model = joblib.load(path)
        print(f"📂 Modell geladen aus {path}")
