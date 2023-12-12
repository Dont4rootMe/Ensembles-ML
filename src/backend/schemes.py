from pydantic import BaseModel
from typing import List, Any
from fastapi import Form


class ModelTrainResponse(BaseModel):
    class _ModelHistory(BaseModel):
        mse: List[float]
        mae: List[float]
        r2: List[float]
        mape: List[float]

    history: _ModelHistory | None
    model: str
    mse: float
    mae: float
    r2: float
    mape: float


class Configuration(BaseModel):
    model: str
    estimators: int
    fetSubsample: float | int
    depth: int | None
    randomState: int | None
    bootstrapCoef: float | int | None
    useRandomSplit: bool
    learningRate: float | None


class ConfigurationSyntet(Configuration):
    class _syntet_pref(BaseModel):
        sample_size: int
        feature_size: int
        validation_percent: int

    synt_prefs: _syntet_pref


class ConfigurationDataSet(Configuration):
    test_size: int | None
    target: str


class FormJsonDataset(ConfigurationDataSet):
    class Config:
        orm_mode = True

    @classmethod
    def as_form(cls, model: str,
                estimators: int,
                fetSubsample: float | int,
                depth: int | None,
                randomState: int | None,
                bootstrapCoef: float | int | None,
                useRandomSplit: bool,
                learningRate: float | None,
                test_size: int | None,
                target: str):
        return cls(model=model,
                   estimators=estimators,
                   fetSubsample=fetSubsample,
                   depth=depth,
                   randomState=randomState,
                   bootstrapCoef=bootstrapCoef,
                   useRandomSplit=useRandomSplit,
                   learningRate=learningRate,
                   test_size=test_size,
                   target=target)
