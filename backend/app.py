# backend/app.py
# -----------------------------------------------------------------------------
# FastAPI uçları:
# - GET  /            : Basit arayüz (frontend/index.html)
# - POST /api/solve   : Ekran görüntüsünü alır, autobox ile hücrelere böler,
#                       Gemini toplu sınıflandırması yapar ve {board, rack, threeStar} döner.
# Notlar:
# - I/İ ayrımı yalnızca processing/gemini_batch.py içinde yapılır (burada müdahale yok).
# - EMP hücrelerinden harf alınmaz; 3 yıldız yeri Gemini + HSV fallback ile bulunur.
# - Çeşitli genişlikler denenip premium karesi uyum skoruna göre en iyi çıktı seçilir.
# -----------------------------------------------------------------------------

import os, io, base64, shutil, uuid, subprocess, sys, glob, json
from pathlib import Path
from typing import Dict, List, Optional
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from PIL import Image, ImageOps

# Görsel tabanlı premium/renk kontrolleri (3 yıldız güvenliği)
import cv2
import numpy as np

from processing.gemini_batch import classify_all_with_gemini

# -----------------------------------------------------------------------------#
# Yapılandırma
# -----------------------------------------------------------------------------#
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

app = FastAPI(title="Kelimelik OCR")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# -----------------------------------------------------------------------------#
# UI
# -----------------------------------------------------------------------------#
@app.get("/", response_class=HTMLResponse)
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


# -----------------------------------------------------------------------------#
# Yardımcılar
# -----------------------------------------------------------------------------#
def _normalize_image(src_path: Path, dst_path: Path, target_w: int) -> None:
    """Görüntüyü EXIF düzelt, genişliğe ölçekle ve PNG kaydet."""
    im = Image.open(src_path)
    im = ImageOps.exif_transpose(im)
    w, h = im.size
    if w != target_w:
        s = target_w / float(w)
        im = im.resize((target_w, max(1, int(h * s))), Image.Resampling.LANCZOS)
    im.save(dst_path, "PNG")


def _build_cells_mosaic(cells_dir: Path, tile: int = 48, pad: int = 2) -> str:
    """15x15 hücrelerin küçük mozaik önizlemesi (frontend’te debug görseli)."""
    W = 15 * tile + 16 * pad
    H = 15 * tile + 16 * pad
    sheet = Image.new("RGB", (W, H), (236, 240, 247))
    for r in range(15):
        for c in range(15):
            x = pad + c * (tile + pad)
            y = pad + r * (tile + pad)
            p = cells_dir / f"r{r:02d}_c{c:02d}.png"
            if p.exists():
                try:
                    im = Image.open(p).convert("RGB").resize((tile, tile), Image.Resampling.BICUBIC)
                except Exception:
                    im = Image.new("RGB", (tile, tile), (220, 220, 220))
            else:
                im = Image.new("RGB", (tile, tile), (220, 220, 220))
            sheet.paste(im, (x, y))
    buf = io.BytesIO()
    sheet.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _premium_score(cells_dir: Path) -> float:
    """Autobox çıktısının premium kareleri yakalama başarısını skorlar (0..1)."""
    ok = tot = 0
    for (r, c) in PREMIUM:
        p = cells_dir / f"r{r:02d}_c{c:02d}.png"
        if not p.exists():
            continue
        try:
            arr = Image.open(p).convert("HSV")
            s_mean = (sum(arr.getchannel("S").getdata()) / 255.0) / (arr.size[0] * arr.size[1])
            if s_mean >= 0.18:
                ok += 1
            tot += 1
        except Exception:
            pass
    return (ok / tot) if tot else 0.0


def _run_autobox_for_width(raw_path: Path, job_dir: Path, width: int) -> Dict:
    """Belirli genişlik için autobox çalıştır, yolları ve skorları döndür."""
    norm = job_dir / f"norm_{width}.png"
    out  = job_dir / f"out_w{width}"
    out.mkdir(parents=True, exist_ok=True)

    _normalize_image(raw_path, norm, target_w=width)
    subprocess.run(
        [sys.executable, str(AUTOBOX), "--image", str(norm), "--out", str(out), "--export-only"],
        check=True,
        cwd=str(THIRD.parent),
    )

    cells_all = sorted(glob.glob(str(out / "cells" / "*.png")))
    cells_occ = sorted(glob.glob(str(out / "cells_occupied" / "*.png")))
    rack_imgs = sorted(glob.glob(str(out / "rack" / "*.png")))
    score     = _premium_score(out / "cells")

    occ_names    = {Path(x).name for x in cells_occ}
    empty_paths  = [p for p in cells_all if Path(p).name not in occ_names]

    return {
        "width": width,
        "out_dir": out,
        "cells_all": cells_all,
        "cells_occ": cells_occ,
        "rack_imgs": rack_imgs,
        "empty_paths": empty_paths,
        "score": score,
        "debugGrid": _build_cells_mosaic(out / "cells"),
    }


# ---------------- 3Y güvenlik kontrolleri ----------------
def _is_premium_cell(r: int, c: int) -> bool:
    """Koordinat premium kare mi? (JSON’dan gelir)"""
    return any((r == pr and c == pc) for pr, pc in PREMIUM)


def _is_clearly_premium_like(img_path: str) -> bool:
    """EMP içinde premium yazı/zemin baskınsa üç yıldız olarak kabul etme."""
    bgr = cv2.imread(img_path)
    if bgr is None:
        return False
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    blue   = cv2.inRange(hsv, np.array([90, 40, 50]),  np.array([130, 255, 255]))
    green  = cv2.inRange(hsv, np.array([45, 35, 50]),  np.array([85, 255, 255]))
    purple = cv2.inRange(hsv, np.array([135, 35, 50]), np.array([165, 255, 255]))
    brown  = cv2.inRange(hsv, np.array([10, 60, 40]),  np.array([25, 220, 200]))
    prem_frac = (blue | green | purple | brown).mean() / 255.0
    return prem_frac > 0.06


# -----------------------------------------------------------------------------#
# API
# -----------------------------------------------------------------------------#
@app.post("/api/solve")
async def solve(file: UploadFile = File(...)):
    # 1) Dosyayı işle
    job = Path("/tmp") / f"kel_{uuid.uuid4().hex}"
    job.mkdir(parents=True, exist_ok=True)
    raw = job / "upload_raw"
    with raw.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # 2) Birkaç farklı genişlikte autobox çalıştır → en iyi eşleşmeyi seç
    tries = []
    for w in [900, 1080, 1260, 1440]:
        try:
            tries.append(_run_autobox_for_width(raw, job, w))
        except subprocess.CalledProcessError:
            pass
    if not tries:
        return JSONResponse({"error": "autobox başarısız"}, status_code=500)

    best         = max(tries, key=lambda t: t["score"])
    cells_occ    = best["cells_occ"]
    rack_imgs    = best["rack_imgs"]
    empty_paths  = best["empty_paths"]
    debug_grid   = best["debugGrid"]
    out_dir      = best["out_dir"]

    # 3) Gemini toplu sınıflandırma (tahta + rack + üç yıldız adayı)
    board_preds, rack_preds, star_emp_idx = classify_all_with_gemini(
        cells_occ, rack_imgs, empty_paths, api_key=GOOGLE_API_KEY
    )

    # 4) Tahtayı 15x15 matrise yaz (Gemini ne verdiyse onu kullan)
    board: List[List[str]] = [["." for _ in range(15)] for _ in range(15)]
    for p, (ch, _conf) in board_preds.items():
        stem = Path(p).stem  # r00_c00
        try:
            r = int(stem.split("_")[0][1:])
            c = int(stem.split("_")[1][1:])
            board[r][c] = ch if ch else "?"
        except Exception:
            pass

    # 5) Raf harfleri (sıra korunur; '?'→'*'; İ/I’ye burada dokunma)
    def r_index(path: str) -> int:
        try:
            return int(Path(path).stem[1:])
        except Exception:
            return 10_000

    rack_chars = ["?" for _ in rack_imgs]
    for p in rack_imgs:
        ch, _ = rack_preds.get(p, ("?", 0.0))
        if ch == "?":
            ch = "*"  # boş rack tile → joker
        idx = r_index(p)
        if 0 <= idx < len(rack_chars):
            rack_chars[idx] = ch
    rack = "".join(rack_chars) or ""

    # 6) Üç yıldız konumu (Gemini + güvenlik kontrolleri)
    three_star: Optional[Dict[str, int]] = None
    if star_emp_idx is not None and 0 <= star_emp_idx < len(empty_paths):
        emp_path = empty_paths[star_emp_idx]
        stem = Path(emp_path).stem  # rXX_cYY
        try:
            rr = int(stem.split("_")[0][1:])
            cc = int(stem.split("_")[1][1:])
            if (not _is_premium_cell(rr, cc)) and (not _is_clearly_premium_like(emp_path)):
                three_star = {"row": rr, "col": cc}
        except Exception:
            pass

    # 7) Yanıt
    return {
        "board": board,
        "rack": rack,
        "threeStar": three_star,
        "debug": {
            "occupied_count": len(cells_occ),
            "rack_tiles": len(rack_imgs),
            "empty_cells_sent": len(empty_paths),
            "chosen_width": best["width"],
            "out_dir": str(out_dir),
        },
        "debugGrid": debug_grid,
    }
