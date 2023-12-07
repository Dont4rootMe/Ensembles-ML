import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from settings import BACKEND_URL

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

if __name__ == '__main__':
    uvicorn.run('api:app', host='0.0.0.0')