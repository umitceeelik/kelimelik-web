# syntax=docker/dockerfile:1
FROM python:3.10-slim

# Sistem bağımlılıkları (Pillow, OpenCV için)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Çalışma dizini (inner klasör kökü)
WORKDIR /app

# --- Sadece requirements'ı kur (cache dostu) ---
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# --- Tüm proje (frontend + backend) ---
COPY . /app

# Uygulama
ENV PORT=8000
EXPOSE 8000
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
