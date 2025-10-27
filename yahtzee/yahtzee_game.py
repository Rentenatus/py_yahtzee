"""
<copyright>
Copyright (c) 2025, Janusch Rentenatus. This program and the accompanying materials are made available under the
terms of the Apache License v2.0 which accompanies this distribution, and is available at
https://github.com/Rentenatus/py_yahtzee?tab=Apache-2.0-1-ov-file#readme
</copyright>

"""

import random
from collections import Counter

class YahtzeeGame:
    def __init__(self):
        self.scorecard = {
            "ones": None, "twos": None, "threes": None,
            "fours": None, "fives": None, "sixes": None,
            "three_of_a_kind": None, "four_of_a_kind": None,
            "full_house": None, "small_straight": None,
            "large_straight": None, "yahtzee": None, "chance": None
        }
        self.dice = []

    def roll_dice(self):
        self.dice = [random.randint(1, 6) for _ in range(5)]
        return self.dice

    def reroll_dice(self, stay):
        new_roll = stay.copy()
        while len(new_roll) < 5:
            new_roll.append(random.randint(1, 6))
        self.dice = new_roll
        return self.dice

    def score_category(self, category):
        if self.scorecard.get(category) is None:
            score = self._score_functions()[category](self.dice)
            self.scorecard[category] = score
            return score
        return None

    def calculate_upper_bonus(self):
        upper_keys = ["ones", "twos", "threes", "fours", "fives", "sixes"]
        upper_total = sum(self.scorecard[k] for k in upper_keys if self.scorecard[k] is not None)
        return 35 if upper_total >= 63 else 0

    def calculate_total_score(self):
        upper_keys = ["ones", "twos", "threes", "fours", "fives", "sixes"]
        lower_keys = [
            "three_of_a_kind", "four_of_a_kind", "full_house",
            "small_straight", "large_straight", "yahtzee", "chance"
        ]
        upper_total = sum(self.scorecard[k] for k in upper_keys if self.scorecard[k] is not None)
        lower_total = sum(self.scorecard[k] for k in lower_keys if self.scorecard[k] is not None)
        bonus = self.calculate_upper_bonus()
        return upper_total + bonus + lower_total

    def print_scorecard(self):
        upper = ["ones", "twos", "threes", "fours", "fives", "sixes"]
        lower = [
            "three_of_a_kind", "four_of_a_kind", "full_house",
            "small_straight", "large_straight", "yahtzee", "chance"
        ]

        def format_line(name):
            label = name.replace("_", " ").title().ljust(20)
            value = self.scorecard[name]
            return f"{label}: {value if value is not None else '—'}"

        print("\n🎲 YAHTZEE SCORECARD")
        print("─" * 32)
        print("Upper Section:")
        for cat in upper:
            print("  " + format_line(cat))
        print("\nLower Section:")
        for cat in lower:
            print("  " + format_line(cat))
        print(f"\nUpper Bonus: {self.calculate_upper_bonus()}")
        print(f"Total Score: {self.calculate_total_score()}")
        print("─" * 32)

    def _score_functions(self):
        return {
            "ones": lambda d: d.count(1) * 1,
            "twos": lambda d: d.count(2) * 2,
            "threes": lambda d: d.count(3) * 3,
            "fours": lambda d: d.count(4) * 4,
            "fives": lambda d: d.count(5) * 5,
            "sixes": lambda d: d.count(6) * 6,
            "three_of_a_kind": lambda d: sum(d) if any(v >= 3 for v in Counter(d).values()) else 0,
            "four_of_a_kind": lambda d: sum(d) if any(v >= 4 for v in Counter(d).values()) else 0,
            "full_house": lambda d: 25 if sorted(Counter(d).values()) == [2, 3] else 0,
            "small_straight": lambda d: 30 if any(s.issubset(set(d)) for s in [{1,2,3,4}, {2,3,4,5}, {3,4,5,6}]) else 0,
            "large_straight": lambda d: 40 if sorted(set(d)) in [[1,2,3,4,5], [2,3,4,5,6]] else 0,
            "yahtzee": lambda d: 50 if len(set(d)) == 1 else 0,
            "chance": lambda d: sum(d)
        }
