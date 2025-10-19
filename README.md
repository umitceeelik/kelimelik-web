# Kelimelik Web

Bu proje, **Kelime Oyunu (Scrabble benzeri)** tahtasından ekran görüntüsü alıp otomatik olarak OCR + Template Matching ile harfleri tanıyabilen ve en iyi hamleleri öneren bir web uygulamasıdır.

## 🚀 Özellikler
- Scrabble tahtasını otomatik algılama (Sobel + Adaptive Thresholding)
- Hücrelerdeki harfleri OCR + Template Matching ile tanıma
- Türkçe dil desteği (özel karakterler: Ç, Ğ, İ, Ö, Ş, Ü)
- Kullanıcı rafındaki harfleri okuma
- JSON formatında çıktı (board + rack)
- Frontend üzerinden en yüksek puanlı kelime önerileri

## 📂 Proje Yapısı

```
KELIMELIK-WEB/
│── backend/               # Python backend (OCR, işleme)
│   ├── app.py             # FastAPI backend
│   ├── processing/        # Görüntü işleme modülleri
│   └── third_party/       # OCR / Template dosyaları
│
│── frontend/              # Web UI (HTML, CSS, JS)
│   ├── index.html
│   ├── style.css
│   ├── main.js
│   ├── kelimelik-solver.js
│   └── kelimelik-config.json
│
│── tr-dictionary.json     # Türkçe sözlük (kelime önerileri için)
│── requirements.txt       # Python bağımlılıkları
│── .gitignore
│── README.md
```

## ⚙️ Kurulum

### 1. Backend (Python)
```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Frontend (Statik)
Sadece `index.html` dosyasını tarayıcıda açman yeterli.

### 3. Çalıştırma
```bash
cd backend
uvicorn app:app --reload
```

Backend çalıştıktan sonra `http://localhost:8000` üzerinden API’lere erişebilirsin.

## 📦 Bağımlılıklar
- Python 3.9+
- OpenCV
- Numpy
- Pytesseract / EasyOCR
- FastAPI
- Uvicorn

Tüm bağımlılıklar `requirements.txt` içinde listelenmiştir.

## 🛠️ Katkı
PR ve issue açabilirsin. Kod stilini korumak için PEP8’e uymaya dikkat et.

## 📜 Lisans
MIT
