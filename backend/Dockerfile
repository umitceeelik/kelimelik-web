# syntax=docker/dockerfile:1
FROM python:3.10-slim

# Sistem bağımlılıkları (Pillow, OpenCV için)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Çalışma dizinini backend'e ayarlayacağız ama context repo kökü olacak
WORKDIR /app/backend

# Önce sadece requirements'ı kopyala ve kur (cache dostu)
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Sonra tüm repo içeriğini kopyala (frontend de dahil)
COPY . /app

# Uygulama başlatma
ENV PORT=8000
EXPOSE 8000
# Artık modul yolu backend altında: backend/app.py -> backend.app:app
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
