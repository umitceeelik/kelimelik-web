# processing/gemini_batch.py
# -----------------------------------------------------------------------------
# Kelimelik ekran görüntüsündeki kutucukları (OCC/RACK/EMP) tek bir “contact
# sheet” olarak Gemini'ye gönderir ve çıkan JSON’u parse eder.
# - OCC: Tahtadaki dolu harf kutusu (tek büyük harf varsa al).
# - RACK: Elde taşlar (tek harf varsa al; boşsa joker "*").
# - EMP: Boş kutu (3 yıldız ikonunu içerebilir). EMP için asla harf üretme.
#
# Özel durumlar:
# - Türkçe I/İ ayrımı görüntüden tespit edilir (nokta varsa I, yoksa İ).
# - Kalabalık ekranlarda stabilite için daha küçük chunk’lar gönderilir.
# - 3-yıldız Gemini’den gelmezse HSV tabanlı fallback ile bulunur.
# -----------------------------------------------------------------------------

import io, re, json, math, os
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps, Image
import numpy as np
import google.generativeai as genai

# Türkçe alfabe + RACK için joker
VALID = ["A","B","C","Ç","D","E","F","G","Ğ","H","I","İ","J","K","L",
         "M","N","O","Ö","P","R","S","Ş","T","U","Ü","V","Y","Z","*"]

# Model talimatı — EMP için harf üretmemesi ve RACK-joker kuralı net
PROMPT_ALL = f"""
You will receive ONE contact sheet with many tiles (image crops).
Legend maps TILE INDEX -> TAG -> FILENAME. TAG ∈ {{OCC, RACK, EMP}}.

Definitions:
- OCC: a board cell. If it shows exactly one uppercase Turkish letter, output it. If empty, ignore it.
- RACK: a rack tile. If it shows exactly one uppercase Turkish letter, output it. If it shows NO letter (blank center), OUTPUT "*" (joker).
- EMP: an empty board cell (may contain the orange three-star icon).

Return ONLY strict JSON:
{{
  "letters": [{{"idx": int, "char": one of [{", ".join(VALID)}], "conf": float}}],
  "threeStar_idx": int | null
}}

Rules:
- Emit letters only for OCC and RACK; never emit letters for EMP.
- For RACK: if the tile has no letter, emit "*" with conf 1.0.
- Do NOT emit "*" for OCC.

STRICT EMP RULES:
- Do NOT output any letters for EMP under ANY circumstance.
- If you see bonus texts like "H²", "K³" etc. on EMP, STILL OUTPUT NOTHING for EMP.
- When in doubt for EMP, prefer to output nothing.
"""

# ---------------- Görsel yardımcıları ----------------

def _pil_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO(); img.save(buf, fmt); return buf.getvalue()

def _to_square_pad(img: Image.Image, pad_ratio: float = 0.12, bg=(245,245,245)) -> Image.Image:
    """Kare kanvas + üstte ekstra pay (İ'nin noktası kesilmesin)."""
    w, h = img.size
    side = max(w, h)
    pad = int(side * pad_ratio)
    top = pad + pad//3
    canvas = Image.new("RGB", (side + 2*pad, side + top + pad), bg)
    ox = (canvas.width - w)//2
    oy = (canvas.height - h)//2 + (top - pad)//2
    canvas.paste(img, (ox, oy))
    return canvas

def _clahe_numpy(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)
    g -= g.min()
    g = np.clip(g / (g.max() + 1e-6), 0, 1)
    g = np.power(g, 0.85)
    return (g*255.0).astype(np.uint8)

def _prep_tile(im: Image.Image) -> Image.Image:
    """Tile'ı büyüt + keskinleştir + normalize + kare pad."""
    im = im.resize((int(im.width*2), int(im.height*2)), Image.Resampling.BICUBIC)
    g = np.array(ImageOps.exif_transpose(im).convert("L"))
    g = _clahe_numpy(g)
    g = Image.fromarray(g).filter(ImageFilter.UnsharpMask(radius=1.2, percent=160, threshold=2))
    arr = np.array(g, dtype=np.uint8)
    lo, hi = np.percentile(arr, (1, 99))
    if hi > lo:
        arr = np.clip((arr - lo) * (255.0/(hi - lo)), 0, 255).astype(np.uint8)
    return _to_square_pad(Image.fromarray(arr).convert("RGB"), pad_ratio=0.12, bg=(245,245,245))

def _grid_sheet(pairs: List[Tuple[str, str]], cols=10, tile=224, pad=12, cap_h=22):
    """Tile’ları tek sheet’e dizer; üstlerine (idx:TAG) etiketi basar."""
    rows = max(1, math.ceil(len(pairs)/cols))
    W = cols*tile + (cols+1)*pad
    H = rows*(tile+cap_h) + (rows+1)*pad
    sheet = Image.new("RGB", (W, H), (250, 250, 250))
    draw = ImageDraw.Draw(sheet)
    try: font = ImageFont.truetype("arial.ttf", 15)
    except: font = ImageFont.load_default()

    meta=[]
    for i,(tag,p) in enumerate(pairs):
        r=i//cols; c=i%cols
        x = pad + c*(tile+pad); y = pad + r*(tile+cap_h+pad)
        try: im = Image.open(p).convert("RGB")
        except: im = Image.new("RGB",(tile,tile),(235,235,235))
        im = _prep_tile(im)
        s = min(tile/im.width, tile/im.height)
        im = im.resize((max(1,int(im.width*s)), max(1,int(im.height*s))), Image.Resampling.BICUBIC)
        sheet.paste(im,(x+(tile-im.width)//2, y+(tile-im.height)//2))

        label = f"{i}:{tag}"
        wtxt = draw.textlength(label, font=font)
        draw.rectangle([x,y,x+max(38,int(wtxt)+10),y+18], fill=(0,0,0))
        draw.text((x+5,y+1), label, fill=(255,255,255), font=font)
        draw.text((x, y+tile+3), os.path.basename(p), fill=(0,0,0), font=font)
        meta.append({"idx": i, "tag": tag, "path": p})
    return sheet, meta

def _legend(meta: List[Dict]) -> str:
    return "Legend (idx -> tag -> filename):\n" + "\n".join(
        f"{m['idx']}: {m['tag']} {os.path.basename(m['path'])}" for m in meta
    )

def _ink_score(path: str) -> float:
    """Koyu piksel oranı ~ görünür mürekkep. Çok düşükse filtrelenir."""
    try: arr = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    except: return 0.0
    if arr.size == 0: return 0.0
    thr = max(40.0, float(arr.mean()) - 0.9*float(arr.std()))
    return float((arr < thr).mean())

# ---------------- Özel I/İ noktası tespiti ----------------
def detect_dotted_i(path: str) -> bool:
    """Üst-orta küçük ve kompakt koyu leke var mı? (İ'nin noktası)"""
    try: g = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    except Exception: return False
    if g.size == 0 or min(g.shape) < 12: return False

    h, w = g.shape
    y0, y1 = int(0.05*h), int(0.35*h)
    cx = w // 2; x0 = max(0, cx - int(0.12*w)); x1 = min(w, cx + int(0.12*w))
    roi = g[y0:y1, x0:x1]
    if roi.size == 0: return False

    mu, sigma = float(roi.mean()), float(roi.std())
    thr = max(40.0, mu - 0.5*sigma)
    ink = roi < thr
    ink_ratio = float(ink.mean())
    if ink_ratio < 0.012: return False

    ys, xs = np.where(ink)
    if ys.size == 0: return False
    bw, bh = xs.max()-xs.min()+1, ys.max()-ys.min()+1
    roi_area = roi.shape[0] * roi.shape[1]
    if (bw*bh) > 0.20*roi_area: return False
    if int(ink.sum()) < 6: return False
    if bh > int(0.50*(y1 - y0)) or bw > int(0.60*(x1 - x0)): return False
    return True

def tr_upper(ch: str) -> str:
    """Türkçe güvenli büyük harf (i→İ, ı→I)."""
    ch = (ch or "").strip()
    if ch == "i": return "İ"
    if ch == "ı": return "I"
    return ch.upper()

# ---------------- 3-yıldız fallback (HSV turuncu skoru) ----------------
def _orange_score(path: str) -> float:
    """Turuncu yıldız için gevşek HSV filtresi; farklı cihaz tonlarına toleranslı."""
    try: arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    except Exception: return 0.0
    if arr.size == 0: return 0.0

    r, g, b = [arr[...,i].astype(np.float32) for i in (0,1,2)]
    mx, mn = np.maximum(np.maximum(r,g), b), np.minimum(np.minimum(r,g), b)
    diff = mx - mn + 1e-6

    h = np.zeros_like(mx)
    m = (mx == r); h[m] = (60.0 * ((g[m]-b[m]) / diff[m]) + 360.0) % 360.0
    m = (mx == g); h[m] = (60.0 * ((b[m]-r[m]) / diff[m]) + 120.0) % 360.0
    m = (mx == b); h[m] = (60.0 * ((r[m]-g[m]) / diff[m]) + 240.0) % 360.0
    h = (h / 2.0)  # 0..180
    s = np.where(mx > 0, (diff / (mx + 1e-6)) * 255.0, 0.0)
    v = mx

    h_mask = (h >= 5) & (h <= 40)
    s_mask = s >= 100
    v_mask = v >= 110
    return float((h_mask & s_mask & v_mask).mean())

# ---------------- Ana işlev ----------------
def classify_all_with_gemini(
    occ_paths: List[str], rack_paths: List[str], emp_paths: List[str],
    api_key: str, model: str = "models/gemini-2.0-flash",
    cols: int = 10, tile: int = 224
) -> Tuple[Dict[str, Tuple[str,float]], Dict[str, Tuple[str,float]], Optional[int]]:
    """
    OCC/RACK/EMP tile'larını parça parça (chunk) Gemini'ye gönderir.
    Dönen harfleri OCR + iş kuralları ile normalize eder.
    """
    genai.configure(api_key=api_key)

    pairs = [("OCC", p) for p in occ_paths] + \
            [("RACK", p) for p in rack_paths] + \
            [("EMP", p) for p in emp_paths]

    board_out: Dict[str, Tuple[str,float]] = {}
    rack_out:  Dict[str, Tuple[str,float]] = {}
    star_emp_idx_global: Optional[int] = None

    # Kalabalıkta karışmayı azaltmak için küçük chunk'lar
    chunk_size = cols * 6

    for base in range(0, len(pairs), chunk_size):
        chunk = pairs[base:base+chunk_size]
        sheet, meta = _grid_sheet(pairs=chunk, cols=cols, tile=tile)
        legend = _legend(meta)
        image_part = {"mime_type":"image/png", "data": _pil_bytes(sheet)}

        resp = genai.GenerativeModel(model).generate_content(
            [PROMPT_ALL, legend, image_part],
            generation_config={"temperature": 0.0}
        )
        text = getattr(resp, "text", None)
        if not text:
            try: text = resp.candidates[0].content.parts[0].text
            except Exception: text = "{}"

        mobj = re.search(r"\{[\s\S]*\}", text)
        try:
            data = json.loads(mobj.group(0) if mobj else text)
        except Exception:
            data = {"letters": [], "threeStar_idx": None}

        letters = data.get("letters", []) or []
        ts_idx  = data.get("threeStar_idx", None)
        idx_map = {m["idx"]: m for m in meta}

        for item in letters:
            try:
                idx  = int(item.get("idx")); raw  = str(item.get("char",""))
                conf = float(item.get("conf",0))
            except Exception:
                continue

            meta_i = idx_map.get(idx)
            if not meta_i:
                continue

            tag, path = meta_i["tag"], meta_i["path"]

            # EMP'ten gelen her şeyi kesinlikle yok say
            if tag == "EMP":
                continue

            ch = tr_upper(raw)           # temel normalize (i/ı)
            if ch == "*" and tag != "RACK":
                continue                 # joker yalnız RACK

            ink = _ink_score(path)       # gürültü filtresi

            # I/İ özel: NOKTA VARSA → I, YOKSA → İ
            if ch in ("I", "İ"):
                try:
                    ch = "I" if detect_dotted_i(path) else "İ"
                except Exception:
                    pass

            if ch not in VALID:
                continue

            # Güven eşiği + mürekkep filtresi
            if (ch != "*" and (conf < 0.65 or ink < 0.02)) or (ch == "*" and conf < 0.50):
                continue

            if tag == "OCC":
                board_out[path] = (ch, conf)
            elif tag == "RACK":
                rack_out[path] = (ch, conf)

        # threeStar_idx chunk içindeki EMP indeksini, global EMP indeksine çevir
        if ts_idx is not None:
            try:
                ts_idx = int(ts_idx)
                if 0 <= ts_idx < len(meta) and idx_map[ts_idx]["tag"] == "EMP":
                    star_emp_idx_global = base + ts_idx - (len(occ_paths) + len(rack_paths))
            except Exception:
                pass

    # Fallback: Gemini threeStar döndürmediyse EMP'lerde turuncu ara
    if star_emp_idx_global is None and len(emp_paths) > 0:
        best_idx, best_score = None, 0.0
        for i, p in enumerate(emp_paths):
            sc = _orange_score(p)
            if sc > best_score:
                best_score, best_idx = sc, i
        if best_idx is not None and best_score >= 0.004:
            star_emp_idx_global = best_idx

    return board_out, rack_out, star_emp_idx_global
