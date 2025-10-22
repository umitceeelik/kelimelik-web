# processing/gemini_batch.py
import io, re, json, math, os, unicodedata
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps, Image
import numpy as np
import google.generativeai as genai

# --- DEBUG toggle (varsayılan AÇIK) ---
DEBUG = str(os.getenv("DEBUG_GEMINI", "1")).lower() in ("1", "true", "yes")

def _log(*args):
    if DEBUG:
        try:
            print(*args, flush=True)
        except Exception:
            pass

def _cp(s: str) -> str:
    """Karakter(ler) için kod noktalarını yazdır (örn: İ U+0130)."""
    if not s:
        return "∅"
    return " ".join(f"{ch} U+{ord(ch):04X}" for ch in str(s))

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

STRICT EMP RULES:
- Do NOT output any letters for EMP under ANY circumstance.
- If you see bonus texts like "H²", "K³" etc. on EMP, STILL OUTPUT NOTHING for EMP.
- When in doubt for EMP, prefer to output nothing.
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

# ---- Nokta tespiti (İ için) ----
def detect_dotted_i(path: str) -> bool:
    """
    Üst-orta bölgede küçük/kompakt koyu leke var mı? (İ'nin noktası)
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
    thr = max(40.0, mu - 0.5 * sigma)
    ink = roi < thr
    ink_ratio = float(ink.mean())

    if ink_ratio < 0.012:
        return False

    ys, xs = np.where(ink)
    if ys.size == 0:
        return False
    bw = xs.max() - xs.min() + 1
    bh = ys.max() - ys.min() + 1
    roi_area = roi.shape[0] * roi.shape[1]
    if (bw * bh) > 0.20 * roi_area:
        return False
    if int(ink.sum()) < 6:
        return False
    if bh > int(0.50 * (y1 - y0)) or bw > int(0.60 * (x1 - x0)):
        return False

    return True

# -------- Türkçe büyük harfe çevirme (yalnızca i/ı için güvenli) --------
def tr_upper(ch: str) -> str:
    ch = (ch or "").strip()
    if ch == "i": return "İ"
    if ch == "ı": return "I"
    return ch.upper()

# --------- Turuncu yıldız fallback skoru (HSV) ----------
def _orange_score(path: str) -> float:
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        return 0.0
    arr = np.asarray(im, dtype=np.uint8)
    if arr.size == 0:
        return 0.0

    # RGB -> HSV (0..255)
    r = arr[...,0].astype(np.float32)
    g = arr[...,1].astype(np.float32)
    b = arr[...,2].astype(np.float32)
    mx = np.maximum(np.maximum(r,g), b)
    mn = np.minimum(np.minimum(r,g), b)
    diff = mx - mn + 1e-6

    # Hue approx (0..180 OpenCV benzeri ölçek)
    h = np.zeros_like(mx)
    mask = (mx == r)
    h[mask] = (60.0 * ((g[mask]-b[mask]) / diff[mask]) + 360.0) % 360.0
    mask = (mx == g)
    h[mask] = (60.0 * ((b[mask]-r[mask]) / diff[mask]) + 120.0) % 360.0
    mask = (mx == b)
    h[mask] = (60.0 * ((r[mask]-g[mask]) / diff[mask]) + 240.0) % 360.0
    h = (h / 2.0)  # 0..180

    s = np.where(mx > 0, (diff / (mx + 1e-6)) * 255.0, 0.0)
    v = mx

    # Biraz gevşek eşikler (kalabalık/iOS toleranslı)
    h_mask = (h >= 5) & (h <= 40)   # eskiden 7..35
    s_mask = s >= 100               # eskiden 110
    v_mask = v >= 110               # eskiden 120
    mask = h_mask & s_mask & v_mask

    return float(mask.mean())

# ---------------- main ----------------
def classify_all_with_gemini(
    occ_paths: List[str], rack_paths: List[str], emp_paths: List[str],
    api_key: str, model: str = "models/gemini-2.0-flash",
    cols: int = 10, tile: int = 224
) -> Tuple[Dict[str, Tuple[str,float]], Dict[str, Tuple[str,float]], Optional[int]]:
    genai.configure(api_key=api_key)

    pairs = [("OCC", p) for p in occ_paths] + \
            [("RACK", p) for p in rack_paths] + \
            [("EMP", p) for p in emp_paths]

    board_out: Dict[str, Tuple[str,float]] = {}
    rack_out:  Dict[str, Tuple[str,float]] = {}
    star_emp_idx_global: Optional[int] = None

    # Kalabalıkta karışmayı azalt
    chunk_size = cols * 6

    for base in range(0, len(pairs), chunk_size):
        chunk = pairs[base:base+chunk_size]
        sheet, meta = _grid_sheet(pairs=chunk, cols=cols, tile=tile)
        legend = _legend(meta)
        image_part = {"mime_type":"image/png", "data": _pil_bytes(sheet)}

        _log("\n=== GEMINI REQUEST ===")
        _log("Legend:\n", legend)

        resp = genai.GenerativeModel(model).generate_content(
            [PROMPT_ALL, legend, image_part],
            generation_config={"temperature": 0.0}
        )
        text = getattr(resp, "text", None)
        if not text:
            try: text = resp.candidates[0].content.parts[0].text
            except Exception: text = "{}"

        _log("\n=== GEMINI RAW TEXT ===")
        _log(text if len(text) < 4000 else text[:4000] + "\n...[truncated]...")

        mobj = re.search(r"\{[\s\S]*\}", text)
        try:
            data = json.loads(mobj.group(0) if mobj else text)
        except Exception as e:
            _log("JSON parse ERROR:", e)
            data = {"letters": [], "threeStar_idx": None}

        letters = data.get("letters", []) or []
        ts_idx  = data.get("threeStar_idx", None)
        idx_map = {m["idx"]: m for m in meta}

        _log(f"\n=== PARSED letters={len(letters)} threeStar_idx={ts_idx} ===")

        for item in letters:
            try:
                idx  = int(item.get("idx"))
                raw  = str(item.get("char",""))
                conf = float(item.get("conf",0))
            except Exception:
                continue

            meta_i = idx_map.get(idx)
            if not meta_i:
                continue
            tag  = meta_i["tag"]
            path = meta_i["path"]

            # --- EMP'ten gelen harfleri tamamen ignore et ---
            if tag == "EMP":
                _log(f"[{idx:02d}] EMP {os.path.basename(path)} -> IGNORE any letters from model")
                continue

            # Normalize
            ch = tr_upper(raw)

            # Joker yalnız RACK
            if ch == "*" and tag != "RACK":
                _log(f"[{idx:02d}] {tag} {os.path.basename(path)} raw='{raw}'({_cp(raw)}) -> ch='{ch}'({_cp(ch)}) conf={conf:.2f}  REASON=reject:*_but_not_RACK")
                continue

            # ink skorunu hesapla (dot kararında da kullanalım)
            ink = _ink_score(path)

            # === SİMETRİK DOT KONTROLÜ (I/İ İÇİN) ===
            dotted = None
            if ch in ("I", "İ"):
                try:
                    dotted = detect_dotted_i(path)
                    # *** İSTENEN TERSLEME ***
                    # nokta varsa I, yoksa İ
                    ch = "I" if dotted else "İ"
                except Exception:
                    dotted = None

            # VALID check
            if ch not in VALID:
                _log(f"[{idx:02d}] {tag} {os.path.basename(path)} raw='{raw}'({_cp(raw)}) -> ch='{ch}'({_cp(ch)}) conf={conf:.2f} ink={ink:.3f} dotted={dotted}  REASON=reject:not_in_VALID")
                continue

            # conf / ink filtreleri
            if ch == "*":
                if conf < 0.50:
                    _log(f"[{idx:02d}] {tag} {os.path.basename(path)} raw='{raw}'({_cp(raw)}) -> ch='*' conf={conf:.2f} ink={ink:.3f} dotted={dotted}  REASON=reject:low_conf_joker")
                    continue
            else:
                if conf < 0.65 or ink < 0.02:
                    _log(f"[{idx:02d}] {tag} {os.path.basename(path)} raw='{raw}'({_cp(raw)}) -> ch='{ch}'({_cp(ch)}) conf={conf:.2f} ink={ink:.3f} dotted={dotted}  REASON=reject:low_conf_or_low_ink")
                    continue

            # Kabul
            if tag == "OCC":
                board_out[path] = (ch, conf)
            elif tag == "RACK":
                rack_out[path]  = (ch, conf)

            _log(f"[{idx:02d}] {tag} {os.path.basename(path)} raw='{raw}'({_cp(raw)}) -> FINAL='{ch}'({_cp(ch)}) conf={conf:.2f} ink={ink:.3f} dotted={dotted}")

        # threeStar köşe bilgisi geldiyse (chunk içi EMP index'ini global EMP index'ine çevir)
        if ts_idx is not None:
            try:
                ts_idx = int(ts_idx)
                if 0 <= ts_idx < len(meta) and idx_map[ts_idx]["tag"] == "EMP":
                    # global EMP index'i: tüm OCC+RACK'leri atla
                    star_emp_idx_global = base + ts_idx - (len(occ_paths) + len(rack_paths))
            except Exception:
                pass

    # ---- Fallback yıldız: Gemini vermezse EMP'leri tara (HSV) ----
    if star_emp_idx_global is None and len(emp_paths) > 0:
        best_idx = None
        best_score = 0.0
        for i, p in enumerate(emp_paths):
            sc = _orange_score(p)
            _log(f"[FALLBACK] EMP {os.path.basename(p)} orange_score={sc:.5f}")
            if sc > best_score:
                best_score = sc
                best_idx = i
        # Biraz daha toleranslı eşik:
        if best_idx is not None and best_score >= 0.004:
            star_emp_idx_global = best_idx

    _log("\n=== SUMMARY ===")
    _log("board_out:", {os.path.basename(k): v for k,v in board_out.items()})
    _log("rack_out:",  {os.path.basename(k): v for k,v in rack_out.items()})
    _log("star_emp_idx_global:", star_emp_idx_global)

    return board_out, rack_out, star_emp_idx_global
