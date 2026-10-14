FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py credit_lib.py index.html ./
COPY model_artifacts ./model_artifacts

EXPOSE 8000

# Les hébergeurs (Render, Railway...) imposent le port via la variable PORT.
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
