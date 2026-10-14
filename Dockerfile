FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
# xgboost ne sert qu'à la comparaison des modèles dans train.py : inutile pour l'API
# (et il télécharge plus de 300 Mo de bibliothèques GPU)
RUN grep -v '^xgboost' requirements.txt > requirements-api.txt     && pip install --no-cache-dir -r requirements-api.txt

COPY app.py credit_lib.py index.html ./
COPY model_artifacts ./model_artifacts

EXPOSE 8000

# Les hébergeurs (Render, Railway...) imposent le port via la variable PORT.
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
