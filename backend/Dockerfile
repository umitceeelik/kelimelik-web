# syntax=docker/dockerfile:1
FROM python:3.10-slim

# Sistem bağımlılıkları (Pillow, OpenCV için)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Çalışma dizinimiz İÇTEKİ klasör olsun
# (repo kökünde bir 'kelimelik-web/' klasörü var)
WORKDIR /app/kelimelik-web

# Sadece requirements'ı kopyala ve kur (cache dostu)
COPY kelimelik-web/backend/requirements.txt /app/kelimelik-web/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/kelimelik-web/backend/requirements.txt

# Tüm repo (frontend + backend) image'a girsin
COPY . /app

# Uygulama başlatma
ENV PORT=8000
EXPOSE 8000
# Artık çalışma dizini /app/kelimelik-web, modül yolu: backend/app.py -> backend.app:app
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
