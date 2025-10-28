"""
<copyright>
Copyright (c) 2025, Janusch Rentenatus. This program and the accompanying materials are made available under the
terms of the Apache License v2.0 which accompanies this distribution, and is available at
https://github.com/Rentenatus/py_yahtzee?tab=Apache-2.0-1-ov-file#readme
</copyright>

"""
from abc import ABC, abstractmethod
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
    def log_category(self, category, score):
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

    def log_category(self, category, score):
        print(self.text,category,score)

    def log_results(self, player_name, game: YahtzeeGame):
        print(f"\n📋 {player_name}'s Scorecard:")
        game.print_scorecard()

