from pydantic import BaseModel
from typing import List, Any


class ModelStats(BaseModel):
    RMSE: float
    R2_score: float
    smape: float
    mape: float


class ModelTrainDefaultResponse(BaseModel):
    stats: ModelStats


class ModelTrainHistoricResponse(ModelTrainDefaultResponse):
    RMSE_history: List[float]
    R2_history: List[float]
    smape_history: List[float]
    mape_history: List[float]


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
