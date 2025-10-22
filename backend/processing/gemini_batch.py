# processing/gemini_batch.py
import io, re, json, math, os
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps, Image
import numpy as np
import google.generativeai as genai

# ==============================
# DEBUG (her zaman açık)
# ==============================
DEBUG = True

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
    Sadece model "I" verdiyse çağrılır. Yanlış pozitifleri sınırlamak için
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

# -------- Turuncu ikon yakalama (fallback) --------
def _orange_score(path: str) -> float:
    """
    Turuncu üç-yıldız ikonunu HSV renk alanında kabaca skorlar.
    Dönüş: [0..1] arası turuncu yoğunluğu.
    """
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        return 0.0

    arr = np.asarray(im, dtype=np.uint8)
    if arr.size == 0:
        return 0.0

    hsv = Image.fromarray(arr).convert("HSV")
    h, s, v = np.dsplit(np.asarray(hsv, dtype=np.uint8), 3)
    h = h.squeeze(); s = s.squeeze(); v = v.squeeze()

    # Geniş turuncu aralığı (iOS farklılıkları için gevşek)
    # ~10..40 derece => 7..35 (0..255 skalasında)
    h_mask = (h >= 7) & (h <= 35)
    s_mask = s >= 110
    v_mask = v >= 120

    mask = h_mask & s_mask & v_mask
    ratio = float(mask.mean())

    if ratio < 0.003:  # %0.3'ten azsa yok say
        return 0.0

    # İkon çoğunlukla üst yarıda belirgin
    H = mask.shape[0]
    top_half = mask[: H // 2, :]
    top_ratio = float(top_half.mean())
    if top_ratio < ratio * 0.5:
        ratio *= 0.6

    return ratio

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

        _log("\n=== GEMINI REQUEST ===")
        _log("Legend:\n", legend)

        resp = genai.GenerativeModel(model).generate_content(
            [PROMPT_ALL, legend, image_part],
            generation_config={"temperature": 0.0}
        )
        text = getattr(resp, "text", None)
        if not text:
            try:
                text = resp.candidates[0].content.parts[0].text
            except Exception:
                text = "{}"

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
            tag = meta_i["tag"] if meta_i else "??"
            path = meta_i["path"] if meta_i else "??"

            # Normalize
            ch = tr_upper(raw)

            # Joker yalnız RACK
            if ch == "*" and tag != "RACK":
                _log(f"[{idx:02d}] {tag} {os.path.basename(path)} raw='{raw}'({_cp(raw)}) -> '*' REJECT (not RACK)")
                continue

            # ink skorunu hesapla (dot kararında da kullanalım)
            ink = _ink_score(path) if meta_i else 0.0

            # === I/İ TERS HARİTALAMA ===
            dotted = None
            if ch in ("I", "İ") and meta_i:
                try:
                    dotted = detect_dotted_i(path)
                    # BİLEREK TERS: nokta varsa "I", yoksa "İ"
                    ch = "I" if dotted else "İ"
                except Exception:
                    dotted = None

            # VALID check
            if idx not in idx_map or ch not in VALID:
                _log(f"[{idx:02d}] {tag} {os.path.basename(path)} -> '{ch}' INVALID  conf={conf:.2f} ink={ink:.3f} dotted={dotted}")
                continue

            # conf / ink filtreleri
            if ch == "*":
                if conf < 0.50:
                    _log(f"[{idx:02d}] {tag} {os.path.basename(path)} -> '*' REJECT low_conf={conf:.2f}")
                    continue
            else:
                if conf < 0.65 or ink < 0.02:
                    _log(f"[{idx:02d}] {tag} {os.path.basename(path)} -> '{ch}' REJECT low (conf={conf:.2f}, ink={ink:.3f}) dotted={dotted}")
                    continue

            # Kabul
            if tag == "OCC":
                board_out[path] = (ch, conf)
            elif tag == "RACK":
                rack_out[path]  = (ch, conf)

            _log(f"[{idx:02d}] {tag} {os.path.basename(path)} raw='{raw}'({_cp(raw)}) -> FINAL='{ch}'({_cp(ch)}) conf={conf:.2f} ink={ink:.3f} dotted={dotted}")

        # --- JSON threeStar_idx geldiyse kullan ---
        if ts_idx is not None:
            try:
                ts_idx = int(ts_idx)
                if 0 <= ts_idx < len(meta) and idx_map[ts_idx]["tag"] == "EMP":
                    star_emp_idx_global = base + ts_idx - (len(occ_paths) + len(rack_paths))
            except Exception:
                pass

        # --- GELMEZSE: turuncu-tarama fallback'i (iOS için) ---
        if star_emp_idx_global is None:
            best_local_idx = None
            best_score = 0.0
            for m in meta:
                if m["tag"] != "EMP":
                    continue
                score = _orange_score(m["path"])
                _log(f"[FALLBACK] EMP {os.path.basename(m['path'])} orange_score={score:.5f}")
                if score > best_score:
                    best_score = score
                    best_local_idx = m["idx"]

            if best_local_idx is not None and best_score >= 0.006:
                try:
                    star_emp_idx_global = base + best_local_idx - (len(occ_paths) + len(rack_paths))
                    _log(f"[FALLBACK] threeStar_idx -> local:{best_local_idx}  global_emp_idx:{star_emp_idx_global}  score={best_score:.5f}")
                except Exception as e:
                    _log("[FALLBACK] threeStar compute error:", e)

    _log("\n=== SUMMARY ===")
    _log("board_out:", {os.path.basename(k): v for k,v in board_out.items()})
    _log("rack_out:", {os.path.basename(k): v for k,v in rack_out.items()})
    _log("star_emp_idx_global:", star_emp_idx_global)

    return board_out, rack_out, star_emp_idx_global
