# backend/app.py
import os, shutil, uuid, subprocess, sys, glob, json
from pathlib import Path
from typing import List
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from processing.gemini_batch import classify_with_gemini

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY .env dosyasında yok.")

BASE_DIR = Path(__file__).resolve().parent
THIRD = BASE_DIR / "third_party"
AUTOBOX = THIRD / "kelimelik_autobox_fixed_v22t5.py"  # tahta & rack dilimleme  :contentReference[oaicite:3]{index=3}

app = FastAPI(title="Kelimelik OCR")
app.mount("/static", StaticFiles(directory=BASE_DIR.parent / "frontend"), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse((BASE_DIR.parent / "frontend" / "index.html"))

@app.post("/api/solve")
async def solve(file: UploadFile = File(...)):
    job_dir = Path("/tmp") / f"kel_{uuid.uuid4().hex}"
    out_dir = job_dir / "out"
    job_dir.mkdir(parents=True, exist_ok=True)
    img_path = job_dir / "upload.png"

    # 1) resmi kaydet
    with img_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # 2) autobox ile 15x15 cell ve rack segmentasyonu (OCR yapmadan)
    cmd = [sys.executable, str(AUTOBOX), "--image", str(img_path),
           "--out", str(out_dir), "--export-only"]
    try:
        subprocess.run(cmd, check=True, cwd=str(THIRD.parent))
    except subprocess.CalledProcessError as e:
        return JSONResponse({"error": f"autobox çalışırken hata: {e}"}, status_code=500)

    cells_occ = sorted(glob.glob(str(out_dir / "cells_occupied" / "*.png")))
    rack_imgs = sorted(glob.glob(str(out_dir / "rack" / "*.png")))

    # 3) Gemini ile toplu sınıflandırma
    board_preds = classify_with_gemini(cells_occ, api_key=GOOGLE_API_KEY)
    rack_preds  = classify_with_gemini(rack_imgs,  api_key=GOOGLE_API_KEY)

    # 4) 15x15 tabloyu doldur
    board = [["." for _ in range(15)] for _ in range(15)]
    for p, (ch, _conf) in board_preds.items():
        name = Path(p).stem  # r00_c00
        try:
            r = int(name.split("_")[0][1:])
            c = int(name.split("_")[1][1:])
            board[r][c] = ch if ch else "?"
        except Exception:
            continue

    # 5) raf: r0.png.. sıralı
    def idx_of(path: str) -> int:
        try: return int(Path(path).stem[1:])
        except: return 1_000_000
    rack = "".join([rack_preds[p][0] for p in sorted(rack_imgs, key=idx_of)])
    rack = rack or ""

    # 6) cevap
    return {
        "board": board,                # 15x15
        "rack": rack,                  # string
        "debug": {
            "occupied_count": len(cells_occ),
            "rack_tiles": len(rack_imgs)
        }
    }
