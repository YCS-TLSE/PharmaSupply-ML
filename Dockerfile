FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY environment.yml pyproject.toml ./
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    pandas \
    numpy \
    xgboost \
    joblib \
    scikit-learn \
    python-dotenv \
    "psycopg[binary]"

COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
