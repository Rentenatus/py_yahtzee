"""
<copyright>
Copyright (c) 2025, Janusch Rentenatus. This program and the accompanying materials are made available under the
terms of the Apache License v2.0 which accompanies this distribution, and is available at
https://github.com/Rentenatus/py_yahtzee?tab=Apache-2.0-1-ov-file#readme
</copyright>
"""


from abc import ABC, abstractmethod
import numpy as np

class MlTrainer(ABC):

    @staticmethod
    def print_train_data(x_data, y_data):
        print(f"📊 Trainingsbeispiele: {len(x_data)}")
        print(f"📐 Feature-Spalten: {list(x_data.columns)}")
        print(f"🎯 Ziel-Spalten: {list(y_data.columns)}")

        print(x_data.head())
        print(y_data.head())

    @staticmethod
    def check_train_data(x_data, y_data, weights):
        # Prüfe auf NaN oder None in x_data
        if x_data.isnull().values.any():
            bad_rows = x_data[x_data.isnull().any(axis=1)]
            raise ValueError(f"❌ x_data enthält NaN/None in Zeilen:\n{bad_rows}")

        # Prüfe auf NaN oder None in y_data
        if y_data.isnull().values.any():
            bad_rows = y_data[y_data.isnull().any(axis=1)]
            raise ValueError(f"❌ y_data enthält NaN/None in Zeilen:\n{bad_rows}")

        # Prüfe auf NaN oder None in weights
        if any(w is None or (isinstance(w, float) and np.isnan(w)) for w in weights):
            bad_indices = [i for i, w in enumerate(weights) if w is None or (isinstance(w, float) and np.isnan(w))]
            raise ValueError(f"❌ weights enthält NaN/None an Positionen: {bad_indices}")


    @abstractmethod
    def save_model(self, path):
       pass

    @abstractmethod
    def load_model(self, path):
         pass

