# 1. Official lightweight Python runtime
FROM python:3.11-slim

# 2. Prevent Python output buffering
ENV PYTHONUNBUFFERED=1

# 3. Set container working directory
WORKDIR /app

# 4. Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy all project source code, models, and data
COPY data_pipeline.py .
COPY monitor.py .
COPY retrain.py .
COPY evaluator.py .
COPY server.py .
COPY data/ ./data/
COPY models/ ./models/

# 6. Expose port 8000 for FastAPI
EXPOSE 8000

# 7. Start the API
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]