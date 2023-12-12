FROM python:3.10 AS backend
RUN pip install pipenv
WORKDIR /app
COPY ./Pipfile ./Pipfile.lock ./
RUN pipenv install
COPY ./src/backend ./

EXPOSE 8000
CMD ["pipenv", "run", "python3", "api.py"]