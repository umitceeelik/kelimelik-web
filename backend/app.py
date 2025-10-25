# backend/app.py
# =============================================================================
# - GET  /            : index.html
# - POST /api/solve   : OCR pipeline (autobox + Gemini) → {board,rack,threeStar}
# - POST /api/words   : Backend solver (kelimelik_solver.py) ile kelime önerileri
# =============================================================================

import os, io, shutil, uuid, subprocess, sys, glob, json
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from PIL import Image, ImageOps
import cv2
import numpy as np

# yerel import (backend klasöründen çalıştırıyoruz)
from backend import kelimelik_solver
from backend.processing.gemini_batch import classify_all_with_gemini

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY .env dosyasında yok.")

BASE_DIR     = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
THIRD        = BASE_DIR / "third_party"
AUTOBOX      = THIRD / "kelimelik_board_ocr.py"

CFG_PATH = FRONTEND_DIR / "kelimelik-config.json"
with CFG_PATH.open("r", encoding="utf-8") as f:
    CFG = json.load(f)
PREMIUM = [(p["row"], p["col"]) for p in CFG.get("premiumSquares", [])]

app = FastAPI(title="Kelimelik OCR + Solver")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# ---------- UI ----------
@app.get("/", response_class=HTMLResponse)
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")

# ---------- Helpers ----------
def _normalize_image_no_resize(src_path: Path, dst_path: Path) -> None:
    """EXIF düzelt, yeniden boyutlama yapmadan PNG kaydet."""
    im = Image.open(src_path)
    im = ImageOps.exif_transpose(im)
    im.save(dst_path, "PNG")

def _run_autobox_for_device(raw_path: Path, job_dir: Path) -> Dict:
    """
    Tek çağrı: Cihaz tespiti ve crop işlemlerini third_party/kelimelik_board_ocr.py
    içindeki devices.json mantığı yapar. Çoklu genişlik denenmez.
    """
    norm = job_dir / "norm.png"
    out  = job_dir / "out_device"
    out.mkdir(parents=True, exist_ok=True)

    _normalize_image_no_resize(raw_path, norm)

    subprocess.run(
        [sys.executable, str(AUTOBOX), "--image", str(norm), "--out", str(out), "--export-only"],
        check=True,
        cwd=str(THIRD.parent),
    )

    cells_all = sorted(glob.glob(str(out / "cells" / "*.png")))
    cells_occ = sorted(glob.glob(str(out / "cells_occupied" / "*.png")))
    rack_imgs = sorted(glob.glob(str(out / "rack" / "*.png")))

    occ_names   = {Path(x).name for x in cells_occ}
    empty_paths = [p for p in cells_all if Path(p).name not in occ_names]
    return {
        "out_dir": out,
        "cells_occ": cells_occ,
        "rack_imgs": rack_imgs,
        "empty_paths": empty_paths,
    }

def _is_premium_cell(r: int, c: int) -> bool:
    return any((r == pr and c == pc) for pr, pc in PREMIUM)

def _is_clearly_premium_like(img_path: str) -> bool:
    """EMP hücresinde premium renk baskınsa üç yıldız sayma."""
    bgr = cv2.imread(img_path)
    if bgr is None: return False
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    blue   = cv2.inRange(hsv, np.array([90, 40, 50]),  np.array([130, 255, 255]))
    green  = cv2.inRange(hsv, np.array([45, 35, 50]),  np.array([85, 255, 255]))
    purple = cv2.inRange(hsv, np.array([135, 35, 50]), np.array([165, 255, 255]))
    brown  = cv2.inRange(hsv, np.array([10, 60, 40]),  np.array([25, 220, 200]))
    prem_frac = (blue | green | purple | brown).mean() / 255.0
    return prem_frac > 0.06

# ---------- OCR API ----------
@app.post("/api/solve")
async def solve(file: UploadFile = File(...)):
    job = Path("/tmp") / f"kel_{uuid.uuid4().hex}"
    job.mkdir(parents=True, exist_ok=True)
    raw = job / "upload_raw"
    with raw.open("wb") as f:
        f.write(await file.read())

    # TEK ÇAĞRI: devices.json kullanan autobox
    try:
        res = _run_autobox_for_device(raw, job)
    except subprocess.CalledProcessError:
        return JSONResponse({"error": "autobox başarısız"}, status_code=500)

    cells_occ    = res["cells_occ"]
    rack_imgs    = res["rack_imgs"]
    empty_paths  = res["empty_paths"]

    # Gemini toplu sınıflandırma
    board_preds, rack_preds, star_emp_idx = classify_all_with_gemini(
        cells_occ, rack_imgs, empty_paths, api_key=GOOGLE_API_KEY
    )

    # 15x15 tahta
    board: List[List[str]] = [["." for _ in range(15)] for _ in range(15)]
    for p, (ch, _conf) in board_preds.items():
        stem = Path(p).stem  # r00_c00
        try:
            r = int(stem.split("_")[0][1:])
            c = int(stem.split("_")[1][1:])
            board[r][c] = ch if ch else "."
        except Exception:
            pass

    # Raf
    def r_index(path: str) -> int:
        try: return int(Path(path).stem[1:])
        except Exception: return 10_000

    rack_chars = ["?" for _ in rack_imgs]
    for p in rack_imgs:
        ch, _ = rack_preds.get(p, ("?", 0.0))
        if ch == "?": ch = "*"
        idx = r_index(p)
        if 0 <= idx < len(rack_chars):
            rack_chars[idx] = ch
    rack = "".join(rack_chars) or ""

    # Üç yıldız
    three_star: Optional[Dict[str, int]] = None
    if star_emp_idx is not None and 0 <= star_emp_idx < len(empty_paths):
        emp_path = empty_paths[star_emp_idx]
        stem = Path(emp_path).stem
        try:
            rr = int(stem.split("_")[0][1:])
            cc = int(stem.split("_")[1][1:])
            if (not _is_premium_cell(rr, cc)) and (not _is_clearly_premium_like(emp_path)):
                three_star = {"row": rr, "col": cc}
        except Exception:
            pass

    return {"board": board, "rack": rack, "threeStar": three_star}

# ---------- BACKEND SOLVER API ----------
class SolveWordsRequest(BaseModel):
    board: list[list[str]]
    rack: str

@app.post("/api/words")
async def solve_words(req: SolveWordsRequest):
    try:
        moves = kelimelik_solver.solve_board(req.board, req.rack)
        return {"moves": moves}
    except Exception as e:
        return {"error": str(e)}
