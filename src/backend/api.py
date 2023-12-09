import uvicorn
from fastapi import FastAPI, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from settings import BACKEND_URL
from fastapi.responses import JSONResponse

import schemes
import engine

from typing import List, Optional, Any

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app = FastAPI(root_path=BACKEND_URL)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/syntet-train', response_model=schemes.ModelTrainResponse)
async def train_on_syntetic_dataset(config: schemes.ConfigurationSyntet, trace: bool):
    X_train, X_test, y_train, y_test = engine.make_syntetic_dataset(config.synt_prefs.sample_size, 
                                                                    config.synt_prefs.feature_size,
                                                                    config.synt_prefs.validation_percent,
                                                                    config.randomState)
    if config.model == 'random-forest':
        return engine.train_random_forest(X_train, X_test, y_train, y_test, config, trace)
    if config.model == 'grad-boosting':
        return engine.train_grad_boost(X_train, X_test, y_train, y_test, config, trace)
    

@app.post('/dataset-train', response_model=schemes.ModelTrainResponse)
async def train_on_specified_data(
        train: UploadFile, 
        test: UploadFile | str,
        trace: bool,
        model: str,
        estimators: int,
        fetSubsample: float | int,
        target: str,
        useRandomSplit: bool,
        test_size: Optional[int] = None,
        depth: Optional[int] = None,
        randomState: Optional[int] = None,
        bootstrapCoef: Optional[float | int] = None,
        learningRate: Optional[float] = None,
        ):

    train = engine.proccess_file(train)
    if test_size is None:
        X_train = train.drop(columns=[target])
        y_train = train[target]

        test = engine.proccess_file(test)
        X_test = test.drop(columns=[target])
        y_test = test[target]
    else:
        X_train, X_test, y_train, y_test = engine.make_train_test_dataset(train, target, test_size)

    config = schemes.Configuration(
        model=model,
        estimators=estimators,
        fetSubsample=fetSubsample,
        depth=depth,
        randomState=randomState,
        bootstrapCoef=bootstrapCoef,
        useRandomSplit=useRandomSplit,
        learningRate=learningRate
    )

    print(config)

    if config.model == 'random-forest':
        return engine.train_random_forest(X_train, X_test, y_train, y_test, config, trace)
    if config.model == 'grad-boosting':
        return engine.train_grad_boost(X_train, X_test, y_train, y_test, config, trace)



if __name__ == '__main__':
    uvicorn.run('api:app', host='0.0.0.0')