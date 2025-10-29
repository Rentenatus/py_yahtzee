"""
<copyright>
Copyright (c) 2025, Janusch Rentenatus. This program and the accompanying materials are made available under the
terms of the Apache License v2.0 which accompanies this distribution, and is available at
https://github.com/Rentenatus/py_yahtzee?tab=Apache-2.0-1-ov-file#readme
</copyright>

"""
from abc import ABC, abstractmethod
import pandas as pd
from yahtzee.yahtzee_game import YahtzeeGame

class YahtzeeLogger(ABC):
    @abstractmethod
    def log_game(self, player_name, game: YahtzeeGame, round_number):
        pass

    @abstractmethod
    def log_roll1(self, dice, stay):
        pass

    @abstractmethod
    def log_roll2(self, dice, stay):
        pass

    @abstractmethod
    def log_roll3(self, dice):
        pass

    @abstractmethod
    def log_category(self, open_scores, category, score):
        pass

    @abstractmethod
    def print_final_results(self):
        pass

    @abstractmethod
    def log_results(self, player_name, game: YahtzeeGame):
        pass

class TextLogger:
    def __init__(self):
        self.text=""

    def log_game(self, player_name, game: YahtzeeGame, round_number):
        self.text=""
        print(f"\n🎯 {player_name} is playing round {round_number + 1}")

    def log_roll1(self, dice, stay):
        self.text = f"{self.text}dice: {' '.join([str(die) for die in dice])}, "
        self.text = f"{self.text}choose: {' '.join([str(die) for die in stay])}, "

    def log_roll2(self, dice, stay):
        self.text = f"{self.text}dice: {' '.join([str(die) for die in dice])}, "
        self.text = f"{self.text}choose: {' '.join([str(die) for die in stay])}, "

    def log_roll3(self, dice):
        self.text = f"{self.text}dice: {' '.join([str(die) for die in dice])}, "

    def log_category(self, open_scores, category, score):
        print(self.text,category,score)

    def print_final_results(self):
        print("\n🏁 Final Results:")

    def log_results(self, player_name, game: YahtzeeGame):
        print(f"\n📋 {player_name}'s Scorecard:")
        game.print_scorecard()




class PandasLogger:
    def __init__(self):
        self.rows = []
        self.current = {}

    def log_game(self, player_name, game: YahtzeeGame, round_number):
        self.current = {
            "player": player_name,
            "round": round_number + 1
        }
        for key, val in game.scorecard.items():
            self.current[f"score_{key}_before"] = val

    def log_roll1(self, dice, stay):
        for i in range(5):
            self.current[f"roll1_dice_{i + 1}"] = dice[i] if i < len(dice) else None
            self.current[f"roll1_stay_{i + 1}"] = stay[i] if i < len(stay) else None
        self.current["roll_number"] = 1

    def log_roll2(self, dice, stay):
        for i in range(5):
            self.current[f"roll2_dice_{i + 1}"] = dice[i] if i < len(dice) else None
            self.current[f"roll2_stay_{i + 1}"] = stay[i] if i < len(stay) else None
        self.current["roll_number"] = 2

    def log_roll3(self, dice):
        for i in range(5):
            self.current[f"roll3_dice_{i + 1}"] = dice[i] if i < len(dice) else None
        self.current["roll_number"] = 3

    def log_category(self, open_scores, category, score):
        for key, val in open_scores.items():
            self.current[f"score_{key}_simulated"] = val
        self.current["category"] = category
        self.current["score"] = score
        self.rows.append(self.current.copy())

    def print_final_results(self):
        pass

    def log_results(self, player_name, game: YahtzeeGame):
        upper_score = game.calculate_upper_score()
        lower_score = game.calculate_lower_score()
        upper_bonus = game.calculate_upper_bonus()
        total_score  = upper_score + lower_score + upper_bonus
        gradient = total_score / 242  # Normalisierung für Lerngewicht
        for row in self.rows:
            row["upper_score"] = upper_score
            row["lower_score"] = lower_score
            row["upper_bonus"] = upper_bonus
            row["total_score"] = total_score
            row["gradient_score"] = gradient

    def to_dataframe(self):
        return pd.DataFrame(self.rows)
