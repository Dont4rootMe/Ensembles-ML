import os
from dotenv import load_dotenv
load_dotenv(verbose=True)


BACKEND_URL = os.getenv('BACKEND_URL', default='/')
MODEL_NUMBER = 0


def INC_MODEL_NUMBER():
    global MODEL_NUMBER
    MODEL_NUMBER = MODEL_NUMBER + 1


def RESET_MODEL_NUMBER():
    global MODEL_NUMBER
    MODEL_NUMBER = 0
