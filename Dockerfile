FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DETECTOR_CONF_THRESHOLD=0.25

# System libraries needed by OpenCV / video processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch
RUN pip install --no-cache-dir \
    torch==2.14.0+cpu \
    torchvision==0.29.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy application and models
COPY app ./app
COPY models ./models

# Create directory for runtime database / files
RUN touch alerts.db

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
