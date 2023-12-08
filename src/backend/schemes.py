from pydantic import BaseModel
from typing import List, Any

class ModelTrainResponse(BaseModel):
    class _ModelHistory(BaseModel):
        mse: List[float]
        r2: List[float]
        mape: List[float]

    history: _ModelHistory | None
    model: str
    mse: float
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
