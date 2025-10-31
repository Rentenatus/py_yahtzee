"""
<copyright>
Copyright (c) 2025, Janusch Rentenatus. This program and the accompanying materials are made available under the
terms of the Apache License v2.0 which accompanies this distribution, and is available at
https://github.com/Rentenatus/py_yahtzee?tab=Apache-2.0-1-ov-file#readme
</copyright>
"""

from yahtzee.ml_dice_trainer import DiceTrainer
from yahtzee.ml_rating_trainer import RatingTrainer
from yahtzee.yahtzee_game import YahtzeeGame
from yahtzee.yahtzee_player import Player


class ModelPlayer(Player):
    def __init__(self, name, dice_model: DiceTrainer, rating_model: RatingTrainer):
        super().__init__(name)
        self.dice_model = dice_model
        self.rating_model = rating_model

    def choose_dice(self, game: YahtzeeGame, roll_number: int):
        extract_dice, keep  =  self.dice_model.predict_from_game(game, roll_number)
        return [extract_dice[i] for i in range(5) if keep[i] > 0.5]

    def choose_rating(self, game: YahtzeeGame, roll_number: int):
        open_scores = game.open_score()
        if not open_scores:
            return open_scores, None

        predicted_category = self.rating_model.predict_from_game(
            game=game,
            simulated_scores=open_scores,
            roll_number=roll_number
        )
        return open_scores, predicted_category


class ModelDifferenzmPlayer(ModelPlayer):

    def __init__(self, name, dice_model: DiceTrainer, dice_bad: DiceTrainer, rating_model: RatingTrainer):
        super().__init__(name, dice_model, rating_model)
        self.dice_bad = dice_bad

    def choose_dice(self, game: YahtzeeGame, roll_number: int):
        extract_dice1, keep1  =  self.dice_model.predict_from_game(game, roll_number)
        extract_dice2, keep2  =  self.dice_bad.predict_from_game(game, roll_number)
        if extract_dice1 == extract_dice2:
            return [extract_dice1[i] for i in range(5) if (keep1[i]>0.96) or (keep1[i] > keep2[i])]
        else:
            print("Warning: Different dices extracted from both models!")
            return [extract_dice1[i] for i in range(5) if keep1[i] > 0.5]
