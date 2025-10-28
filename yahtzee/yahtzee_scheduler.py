"""
<copyright>
Copyright (c) 2025, Janusch Rentenatus. This program and the accompanying materials are made available under the
terms of the Apache License v2.0 which accompanies this distribution, and is available at
https://github.com/Rentenatus/py_yahtzee?tab=Apache-2.0-1-ov-file#readme
</copyright>

"""
from yahtzee.yahtzee_game import YahtzeeGame
from yahtzee.yahtzee_logger import TextLogger

class Scheduler:

    def __init__(self, players, logger_class = TextLogger):
        self.players = players
        self.games = {player.name: YahtzeeGame() for player in players}
        self.loggers = {player.name: logger_class() for player in players}

    def start(self):
        for round_number in range(13):  # 13 Kategorien pro Spieler
            for player in self.players:
                game = self.games[player.name]
                logger = self.loggers[player.name]
                logger.log_game(player.name, game, round_number)

                # Roll 1
                game.roll_dice()
                stay = player.choose_dice(game, 1)
                logger.log_roll1(game.dice, stay)
                if len(stay) == 5:
                    category = player.choose_rating(game, 1)
                    game.score_category(category)
                    logger.log_category(category, game.scorecard[category])
                    continue

                # Roll 2
                game.reroll_dice(stay)
                stay = player.choose_dice(game, 2)
                logger.log_roll2(game.dice, stay)
                if len(stay) == 5:
                    category = player.choose_rating(game, 2)
                    game.score_category(category)
                    logger.log_category(category, game.scorecard[category])
                    continue

                # Roll 3
                game.reroll_dice(stay)
                logger.log_roll3(game.dice)
                category = player.choose_rating(game, 3)
                game.score_category(category)
                logger.log_category(category, game.scorecard[category])

        # Final scorecards
        next(iter(self.loggers.values())).print_final_results()
        for player in self.players:
            game = self.games[player.name]
            logger = self.loggers[player.name]
            logger.log_results(player.name, game)
