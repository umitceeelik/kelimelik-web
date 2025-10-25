# syntax=docker/dockerfile:1
FROM python:3.10-slim

# Sistem bağımlılıkları (Pillow, OpenCV için)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Çalışma dizini (inner klasör)
WORKDIR /app/kelimelik-web

# --- Yalnızca requirements'ı kur (cache dostu) ---
# DİKKAT: Kaynak yollar build context'e (repo kökü: ".") göre verilir.
COPY kelimelik-web/backend/requirements.txt /app/kelimelik-web/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/kelimelik-web/backend/requirements.txt

# --- Tüm projenin kopyası (frontend + backend) ---
COPY kelimelik-web /app/kelimelik-web

# Uygulama
ENV PORT=8000
EXPOSE 8000
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
