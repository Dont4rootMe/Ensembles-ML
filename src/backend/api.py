import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from settings import BACKEND_URL
from typing import Dict

import schemes
import engine

from typing import List

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

@app.post('/syntet-train')   #, response_model=schemes.ModelTrainResponse)
def train_on_syntetic_dataset(config: schemes.ConfigurationSyntet, trace: bool):
    X_train, X_test, y_train, y_test = engine.make_syntetic_dataset(config.synt_prefs.sample_size, 
                                                                    config.synt_prefs.feature_size,
                                                                    config.synt_prefs.validation_percent)
    if config.model == 'random-forest':
        return engine.train_random_forest(X_train, X_test, y_train, y_test, config, trace)

if __name__ == '__main__':
    uvicorn.run('api:app', host='0.0.0.0')