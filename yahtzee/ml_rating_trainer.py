from abc import ABC, abstractmethod
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from yahtzee.ml_a_trainer import MlTrainer
from yahtzee.yahtzee_game import YahtzeeGame

class RatingTrainer(MlTrainer):

    @staticmethod
    def extract_rating_training_data(df):
        features = []
        targets = []
        weights = []

        score_categories = YahtzeeGame.score_categories()

        for _, row in df.iterrows():
            if pd.isna(row.get("category")):
                continue

            # Eingabefeatures: score_before + score_simulated + roll_number
            score_before = [
                -9 if pd.isna(row.get(f"score_{cat}_before",-9)) else row.get(f"score_{cat}_before",-9)
                for cat in score_categories
            ]
            score_simulated = [
                -9 if pd.isna(row.get(f"score_{cat}_simulated",-9)) else row.get(f"score_{cat}_simulated",-9)
                for cat in score_categories
            ]
            roll_number = row.get("round", 1)

            x = score_before + score_simulated + [roll_number]

            # Ziel: flache One-Hot-Kodierung der gewählten Kategorie
            y = [0] * len(score_categories)
            try:
                index = score_categories.index(row["category"])
                y[index] = 1
            except ValueError:
                continue  # ungültige Kategorie überspringen

            gradient = row.get("gradient_score", 1.0)  # Default: neutrale Lernstärke

            features.append(x)
            targets.append(y)
            weights.append(gradient)

        x_columns = [f"score_{cat}_before" for cat in score_categories] + \
                    [f"score_{cat}_simulated" for cat in score_categories] + ["roll_number"]
        y_columns = [f"chosen_{cat}" for cat in score_categories]

        x_data = pd.DataFrame(features, columns=x_columns)
        y_data = pd.DataFrame(targets, columns=y_columns)
        return x_data, y_data, weights

    @staticmethod
    def _predict_from_game(ml_model, game: YahtzeeGame, simulated_scores: dict, roll_number: int):
        if ml_model is None:
            raise ValueError("Modell ist nicht trainiert")

        score_categories = YahtzeeGame.score_categories()
        score_before = [
            -9 if pd.isna(game.scorecard.get(cat, -9)) else game.scorecard.get(cat, -9)
            for cat in score_categories
        ]
        score_simulated = [
            -9 if pd.isna(simulated_scores.get(cat, -9)) else simulated_scores.get(cat, -9)
            for cat in score_categories
        ]

        x = score_before + score_simulated + [roll_number]
        columns = [f"score_{cat}_before" for cat in score_categories] + \
                  [f"score_{cat}_simulated" for cat in score_categories] + ["roll_number"]

        df = pd.DataFrame([x], columns=columns)
        probas = ml_model.predict_proba(df)[0]

        # 🔒 Maskierung: Kategorien, die in simulated_scores None sind, bekommen -999
        masked_probas = [
            probas[i] if simulated_scores.get(cat) is not None else -999
            for i, cat in enumerate(score_categories)
        ]

        best_index = masked_probas.index(max(masked_probas))
        return score_categories[best_index]

    @abstractmethod
    def train_model(self, x_data, y_data, w_train):
        pass

    @abstractmethod
    def evaluate_model(self, x_data, y_data):
        pass

    @abstractmethod
    def predict_from_game(self, game: YahtzeeGame, simulated_scores: dict, roll_number: int):
        pass

class RatingTrainerRandomForest(RatingTrainer):
    def __init__(self):
        self.model = None

    def train_model(self, x_data, y_data, w_train):
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
        return self._predict_from_game(self.model, game, simulated_scores, roll_number)

    def save_model(self, path):
        import joblib
        joblib.dump(self.model, path)
        print(f"💾 Modell gespeichert unter {path}")

    def load_model(self, path):
        import joblib
        self.model = joblib.load(path)
        print(f"📂 Modell geladen aus {path}")



class RatingTrainerNN(RatingTrainer):
    def __init__(self):
        self.model = None

    def train_model(self, x_data, y_data, w_train):
        self.model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42
        )
        # Klassenziel: Index der Kategorie
        y_target = y_data.values.argmax(axis=1)
        self.model.fit(x_data, y_target, sample_weight=w_train)
        print("🧠 NN RatingModel erfolgreich trainiert")

    def evaluate_model(self, x_test, y_test):
        if self.model is None:
            print("⚠️ Kein Modell vorhanden")
            return
        y_target = y_test.values.argmax(axis=1)
        score = self.model.score(x_test, y_target)
        print(f"📊 Genauigkeit: {score:.3f}")
        return score

    def predict_from_game(self, game: YahtzeeGame, simulated_scores: dict, roll_number: int):
        return self._predict_from_game(self.model, game, simulated_scores, roll_number)

    def save_model(self, path):
        import joblib
        joblib.dump(self.model, path)
        print(f"💾 NN-Modell gespeichert unter {path}")

    def load_model(self, path):
        import joblib
        self.model = joblib.load(path)
        print(f"📂 NN-Modell geladen aus {path}")
