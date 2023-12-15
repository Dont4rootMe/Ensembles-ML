from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from fastapi.exceptions import HTTPException
import settings
import pandas as pd
import numpy as np
from typing import Tuple
import ensembles as ensembles
import pickle
import os

from schemes import Configuration

from sklearn.metrics import mean_squared_error


def make_syntetic_dataset(sample_size: int, feature_size: int, percent: int, randomState: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = make_regression(sample_size, feature_size, random_state=randomState)

    return train_test_split(X, y, test_size=(percent / 100))


def make_train_test_dataset(dataset: pd.DataFrame, target: str, test_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        X = dataset.drop(columns=[target])
        y = dataset[target]
    except:
        raise HTTPException(
            status_code=500, detail=f'Отсутсвует колонка {target}')

    return train_test_split(X, y, test_size=(test_size / 100))


def proccess_file(file):
    return pd.read_csv(file.file)


def train_random_forest(X_train, X_test, y_train, y_test, config: Configuration, trace: bool, return_model: bool = False):
    try:
        model: ensembles.RandomForestMSE = ensembles.RandomForestMSE(
            n_estimators=config.estimators,
            max_depth=config.depth,
            feature_subsample_size=config.fetSubsample,
            splitter='best' if not config.useRandomSplit else 'random',
            bootstrap=config.bootstrapCoef,
        )
    except:
        HTTPException(
            status_code=500, detail='Ошибка при построении Random Forrest. Перепроверьте параметры')

    history_obj = {
        'X_val': X_test,
        'y_val': y_test
    } if trace else {}

    try:
        history = model.fit(X_train, y_train, **history_obj)
    except:
        raise HTTPException(status_code=500, detail='Ошибка в обучении модели')
    mse, mae, r2, mape = model.make_metrics(X_test, y_test)

    print(type(model))

    if return_model:
        with open(
                f'./src/backend/local_model_storage/model_serialization_{settings.MODEL_NUMBER}.db', 'wb') as f:
            pickle.dump(model, f)

    response = {
        'number': settings.MODEL_NUMBER,
        'model': 'Random forest',
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'history': history if trace else None
    }

    settings.INC_MODEL_NUMBER()

    return response


def train_grad_boost(X_train, X_test, y_train, y_test, config: Configuration, trace: bool, return_model: bool = False):
    try:
        model: ensembles.GradientBoostingMSE = ensembles.GradientBoostingMSE(
            n_estimators=config.estimators,
            learning_rate=config.learningRate,
            max_depth=config.depth,
            feature_subsample_size=config.fetSubsample,
            splitter='best' if not config.useRandomSplit else 'random',
            bootstrap=config.bootstrapCoef,
        )
    except:
        HTTPException(
            status_code=500, detail='Ошибка при построении Градиентного бустинга. Перепроверьте параметры')

    history_obj = {
        'X_val': X_test,
        'y_val': y_test
    } if trace else {}

    try:
        history = model.fit(X_train, y_train, **history_obj)
    except:
        raise HTTPException(status_code=500, detail='Ошибка в обучении модели')
    mse, mae, r2, mape = model.make_metrics(X_test, y_test)

    if return_model:
        print('inside')
        with open(
                f'./src/backend/local_model_storage/model_serialization_{settings.MODEL_NUMBER}.db', 'wb') as f:
            pickle.dump(model, f)

    response = {
        'number': settings.MODEL_NUMBER,
        'model': 'Gradient Boosting',
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'history': history if trace else None
    }

    settings.INC_MODEL_NUMBER()

    return response


def delete_model(model: int):
    try:
        os.remove(
            f'./src/backend/local_model_storage/model_serialization_{model}.db')
    except OSError:
        pass


def delete_models():
    for i in range(settings.MODEL_NUMBER):
        try:
            os.remove(
                f'./src/backend/local_model_storage/model_serialization_{i}.db')
        except OSError:
            pass
