"""
<copyright>
Copyright (c) 2025, Janusch Rentenatus. This program and the accompanying materials are made available under the
terms of the Apache License v2.0 which accompanies this distribution, and is available at
https://github.com/Rentenatus/py_yahtzee?tab=Apache-2.0-1-ov-file#readme
</copyright>
"""

import pandas as pd
from yahtzee.yahtzee_logger import PandasLogger
from yahtzee.yahtzee_player import ChaosPlayer, AlwaysKeepPlayer
from yahtzee.yahtzee_scheduler import Scheduler


def run_batch_games(players, logger_class, num_games=1000):
    all_dfs = []

    for i in range(num_games):
        scheduler = Scheduler(players, logger_class=logger_class)
        scheduler.start()

        for player in players:
            logger = scheduler.loggers[player.name]
            if hasattr(logger, "to_dataframe"):
                df = logger.to_dataframe()
                if df.empty:
                    print(f"⚠️  Leerer DataFrame in Spiel {i}, Spieler {player.name}")
                else:
                    df["game_id"] = i
                    all_dfs.append(df)
            else:
                print(f"⚠️  Logger von {player.name} hat keine to_dataframe()-Methode")

    return pd.concat(all_dfs, ignore_index=True)


def main():
    players = [ChaosPlayer("ChaosBot"),AlwaysKeepPlayer("AlwaysKeepBot")]
    df = run_batch_games(players, logger_class=PandasLogger, num_games=38000)

    filename1 = "assets/yahtzee_training_data_01.csv"
    filename2 = "assets/yahtzee_training_data_02.csv"

    splitt_and_save(df, filename1, filename2)


def splitt_and_save(df: pd.DataFrame, filename1: str, filename2: str):
    if "total_score" in df.columns:
        # Globaler Schwellenwert
        threshold_good = df["total_score"].quantile(1 - 0.2)  # Top 20%
        df_good = df[df["total_score"] >= threshold_good]
        threshold_bad = df["total_score"].quantile(0.2)  # Bottom 20%
        df_bad = df[df["total_score"] <= threshold_bad]

        print(f"✅ Batch abgeschlossen: {len(df)} Zeilen")
        print(df.head())

        df_good.to_csv(filename1, index=False)
        print(f"📁 Top Daten gespeichert unter {filename1}")

        df_bad.to_csv(filename2, index=False)
        print(f"📁 Bad Daten gespeichert unter {filename2}")

    else:
        print("Colum total_score not found.")


if __name__ == "__main__":
    main()
