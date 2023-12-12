import os
from dotenv import load_dotenv
load_dotenv(verbose=True)


BACKEND_URL = os.getenv('BACKEND_URL', default='/')
