FROM python:3.11-slim
WORKDIR /app

# Install Python dependencies
COPY fastapi_app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy FastAPI app
COPY fastapi_app /app/fastapi_app

EXPOSE 8000
CMD ["uvicorn", "fastapi_app.main:app", "--host", "0.0.0.0", "--port", "8000"]
