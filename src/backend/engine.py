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

def train_random_forest(X_train, X_test, y_train, y_test, config: Configuration, trace: bool):
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
    } if trace else {}

    history = model.fit(X_train, y_train, **history_obj)
    mse, r2, mape  = model.make_metrics(X_test, y_test)

    response = {
        'model': 'Random forest',
        'mse': mse,
        'r2': r2,
        'mape': mape,
        'history': history if trace else None
    }
    return response


def train_grad_boost(X_train, X_test, y_train, y_test, config: Configuration, trace: bool):
    model: ensembles.GradientBoostingMSE = ensembles.GradientBoostingMSE(
        n_estimators=config.estimators,
        learning_rate=config.learningRate,
        max_depth=config.depth,
        feature_subsample_size=config.fetSubsample,
        splitter='best' if not config.useRandomSplit else 'random',
        bootstrap=config.bootstrapCoef,
    )
    history_obj = {
        'X_val': X_test,
        'y_val': y_test
    } if trace else {}

    history = model.fit(X_train, y_train, **history_obj)
    mse, r2, mape  = model.make_metrics(X_test, y_test)

    response = {
        'model': 'Random forest',
        'mse': mse,
        'r2': r2,
        'mape': mape,
        'history': history if trace else None
    }
    return response