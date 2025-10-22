# processing/gemini_batch.py
import io, re, json, math, os
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps, Image
import numpy as np
import google.generativeai as genai

# Türkçe alfabe + RACK için joker '*'
VALID = ["A","B","C","Ç","D","E","F","G","Ğ","H","I","İ","J","K","L",
         "M","N","O","Ö","P","R","S","Ş","T","U","Ü","V","Y","Z","*"]

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
"""

# ---------------- utils ----------------
def _pil_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO(); img.save(buf, fmt); return buf.getvalue()

def _to_square_pad(img: Image.Image, pad_ratio: float = 0.12, bg=(245,245,245)) -> Image.Image:
    # Kare kanvas + üstte biraz fazla pay (İ noktasını korumak için)
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
    rng = g.max() + 1e-6
    g = np.clip(g / rng, 0, 1)
    g = np.power(g, 0.85)
    return (g*255.0).astype(np.uint8)

def _prep_tile(im: Image.Image) -> Image.Image:
    # 2× büyüt + normalize + unsharp + kare pad
    scale = 2.0
    w, h = im.size
    im = im.resize((int(w*scale), int(h*scale)), Image.Resampling.BICUBIC)
    g = np.array(ImageOps.exif_transpose(im).convert("L"))
    g = _clahe_numpy(g)
    sharpen = Image.fromarray(g).filter(ImageFilter.UnsharpMask(radius=1.2, percent=160, threshold=2))
    arr = np.array(sharpen, dtype=np.uint8)
    lo, hi = np.percentile(arr, (1, 99))
    if hi > lo:
        arr = np.clip((arr - lo) * (255.0/(hi - lo)), 0, 255).astype(np.uint8)
    out = Image.fromarray(arr).convert("RGB")
    return _to_square_pad(out, pad_ratio=0.12, bg=(245,245,245))

def _grid_sheet(pairs: List[Tuple[str, str]], cols=12, tile=224, pad=12, cap_h=22):
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
        x = pad + c*(tile+pad)
        y = pad + r*(tile+cap_h+pad)
        try:
            im = Image.open(p).convert("RGB")
        except:
            im = Image.new("RGB",(tile,tile),(235,235,235))
        im = _prep_tile(im)
        iw, ih = im.size
        s = min(tile/iw, tile/ih)
        im = im.resize((max(1,int(iw*s)), max(1,int(ih*s))), Image.Resampling.BICUBIC)
        ox = x + (tile-im.size[0])//2
        oy = y + (tile-im.size[1])//2
        sheet.paste(im,(ox,oy))

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
    try: arr = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    except: return 0.0
    if arr.size == 0: return 0.0
    thr = max(40.0, float(arr.mean()) - 0.9*float(arr.std()))
    return float((arr < thr).mean())

# ---- yalnız model "I" dediğinde nokta arayan minik sezgi ----
def detect_dotted_i(path: str) -> bool:
    """
    Üst-orta bölgede küçük/kompakt koyu leke var mı? (İ'nin noktası)
    Sadece model "I" verdiyse çağrılır. Yanlış pozitifleri sınırlamak için
    epey dar bir pencere ve kompaktlık kontrolü kullanıyoruz.
    """
    try:
        g = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    except Exception:
        return False
    if g.size == 0:
        return False

    h, w = g.shape
    if h < 12 or w < 12:
        return False

    # Üst-orta pencere: yükseklik %5..35, genişlik merkez ±%12
    y0 = int(0.05 * h); y1 = int(0.35 * h)
    cx = w // 2
    x0 = max(0, cx - int(0.12 * w))
    x1 = min(w, cx + int(0.12 * w))
    roi = g[y0:y1, x0:x1]
    if roi.size == 0:
        return False

    mu, sigma = float(roi.mean()), float(roi.std())
    thr = max(40.0, mu - 0.5 * sigma)  # koyuyu yakala
    ink = roi < thr
    ink_ratio = float(ink.mean())

    if ink_ratio < 0.012:  # neredeyse hiç koyu yoksa: dot yok
        return False

    # Kompaktlık/ölçek kontrolü (dot çok büyük olmamalı)
    ys, xs = np.where(ink)
    if len(xs) == 0:
        return False
    bw = xs.max() - xs.min() + 1
    bh = ys.max() - ys.min() + 1
    box_area = bw * bh
    pix = int(ink.sum())

    # Dot için kaba sınırlar: kutunun küçük bir kısmını kaplasın
    roi_area = roi.shape[0] * roi.shape[1]
    if box_area > 0.20 * roi_area:   # çok geniş bir leke → büyük ihtimalle dot değil
        return False
    if pix < 6:                      # çok az piksel → gürültü
        return False

    # En-boy da çok uzun olmasın (sap gibi bir leke olmasın)
    if bh > int(0.50 * (y1 - y0)) or bw > int(0.60 * (x1 - x0)):
        return False

    return True

# -------- Türkçe büyük harfe çevirme (yalnızca i/ı için güvenli) --------
def tr_upper(ch: str) -> str:
    ch = ch.strip()
    if ch == "i": return "İ"
    if ch == "ı": return "I"
    return ch.upper()

# ---------------- main ----------------
def classify_all_with_gemini(
    occ_paths: List[str], rack_paths: List[str], emp_paths: List[str],
    api_key: str, model: str = "models/gemini-2.0-flash",
    cols: int = 12, tile: int = 224
) -> Tuple[Dict[str, Tuple[str,float]], Dict[str, Tuple[str,float]], Optional[int]]:
    genai.configure(api_key=api_key)

    pairs = [("OCC", p) for p in occ_paths] + \
            [("RACK", p) for p in rack_paths] + \
            [("EMP", p) for p in emp_paths]

    board_out: Dict[str, Tuple[str,float]] = {}
    rack_out:  Dict[str, Tuple[str,float]] = {}
    star_emp_idx_global: Optional[int] = None

    chunk_size = cols * 8
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

        letters = data.get("letters", [])
        ts_idx  = data.get("threeStar_idx", None)
        idx_map = {m["idx"]: m for m in meta}

        for item in letters:
            try:
                idx  = int(item.get("idx"))
                raw  = str(item.get("char",""))
                ch   = tr_upper(raw)       # Türkçe güvenli upper
                conf = float(item.get("conf",0))
            except Exception:
                continue
            if idx not in idx_map or ch not in VALID:
                continue

            tag, path = idx_map[idx]["tag"], idx_map[idx]["path"]

            # Joker yalnız RACK
            if ch == "*" and tag != "RACK":
                continue

            # hafif filtre
            if ch == "*":
                if conf < 0.50:
                    continue
            else:
                if conf < 0.65:
                    continue
                if _ink_score(path) < 0.02:
                    continue

            # ---- I → İ görsel sezgisi ----
            # Model "I" dediyse ve üst-ortada nokta benzeri leke bulunuyorsa, İ'ye çevir.
            if ch == "I":
                try:
                    if detect_dotted_i(path):
                        ch = "İ"
                except Exception:
                    pass

            if tag == "OCC":
                board_out[path] = (ch, conf)
            elif tag == "RACK":
                rack_out[path]  = (ch, conf)

        if ts_idx is not None:
            try:
                ts_idx = int(ts_idx)
                if 0 <= ts_idx < len(meta) and idx_map[ts_idx]["tag"] == "EMP":
                    star_emp_idx_global = base + ts_idx - (len(occ_paths) + len(rack_paths))
            except Exception:
                pass

    return board_out, rack_out, star_emp_idx_global
