# syntax=docker/dockerfile:1
FROM python:3.10-slim

# Sistem bağımlılıkları (Pillow, OpenCV için)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Kaynak kodu kopyala
COPY . /app

# Entrypoint
ENV PORT=8000
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
