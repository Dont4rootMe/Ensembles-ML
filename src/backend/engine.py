from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Tuple
import sys
sys.path.insert(0, './src')

from schemes import Configuration
import ensembles

from sklearn.metrics import mean_squared_error

def make_syntetic_dataset(sample_size: int, feature_size: int, percent: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = make_regression(sample_size, feature_size)

    return train_test_split(X, y, test_size=(percent / 100))

def train_random_forest(X_train, X_test, y_train, y_test, config: Configuration, history: bool):
    print(config)

    model: ensembles.RandomForestMSE = ensembles.RandomForestMSE(
        n_estimators=config.estimators,
        max_depth=config.depth,
        feature_subsample_size=config.fetSubsample,
        splitter='best' if not config.useRandomSplit else 'random',
        bootstrap=config.bootstrapCoef,
    )
    history_obj = {
        'X_val': X_test,
        'y_val': y_test
    } if history else {}

    model.fit(X_train, y_train, **history_obj)

    print(mean_squared_error(y_test, model.predict(X_test)))