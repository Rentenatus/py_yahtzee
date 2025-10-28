"""
<copyright>
Copyright (c) 2025, Janusch Rentenatus. This program and the accompanying materials are made available under the
terms of the Apache License v2.0 which accompanies this distribution, and is available at
https://github.com/Rentenatus/py_yahtzee?tab=Apache-2.0-1-ov-file#readme
</copyright>

"""
from yahtzee.yahtzee_game import YahtzeeGame


class Scheduler:
    def __init__(self, players):
        self.players = players
        self.games = {player.name: YahtzeeGame() for player in players}

    def start(self):
        for round_number in range(13):  # 13 Kategorien pro Spieler
            for player in self.players:
                game = self.games[player.name]
                print(f"\n🎯 {player.name} is playing round {round_number + 1}")

                # Roll 1
                game.roll_dice()
                text = f"dice: {' '.join([str(die) for die in game.dice])}, "
                stay = player.choose_dice(game, 1)
                text = f"{text}choose: {' '.join([str(die) for die in stay])}, "
                if len(stay) == 5:
                    category = player.choose_rating(game, 1)
                    game.score_category(category)
                    print(text,category,game.scorecard[category])
                    continue

                # Roll 2
                game.reroll_dice(stay)
                text = f"{text}dice: {' '.join([str(die) for die in game.dice])}, "
                stay = player.choose_dice(game, 2)
                text = f"{text}choose: {' '.join([str(die) for die in stay])}, "
                if len(stay) == 5:
                    category = player.choose_rating(game, 2)
                    game.score_category(category)
                    print(text, category, game.scorecard[category])
                    continue

                # Roll 3
                game.reroll_dice(stay)
                text = f"{text}dice: {' '.join([str(die) for die in game.dice])}, "
                category = player.choose_rating(game, 3)
                game.score_category(category)
                print(text, category, game.scorecard[category])

        # Final scorecards
        print("\n🏁 Final Results:")
        for player in self.players:
            game = self.games[player.name]
            print(f"\n📋 {player.name}'s Scorecard:")
            game.print_scorecard()
