# backend/third_party/kelimelik_board_ocr.py
# Sadece kırpma/segmentasyon (OCR YOK)
# - Cihaz bazlı JSON'dan üst boşluk oranı (top_frac) seçimi
# - 15x15 hücreleri out/cells/ altına yaz
# - Taş içerenleri out/cells_occupied/ altına da kopyala
# - Rack bandını bulup out/rack/ altına tek tek yaz
# - --export-only uyumluluk için var (yoksayılır)

import os, cv2, json, argparse, numpy as np

# Varsayılan (JSON bulunamazsa/uygun eşleşme olmazsa)
DEFAULT_TOP_FRAC = 993.0 / 3088.0  # S23 Ultra örneği

def ensure_dir(p): os.makedirs(p, exist_ok=True)

# ---------- JSON: cihaz -> top_frac seçimi ----------
def _load_devices_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
        out = []
        for d in data:
            # Beklenen alanlar: screen_w, screen_h, (board_top_px | top_frac), optional name
            w = int(d.get("screen_w", 0) or 0)
            h = int(d.get("screen_h", 0) or 0)
            if w <= 0 or h <= 0:
                continue
            if "top_frac" in d and isinstance(d["top_frac"], (int, float)):
                tf = float(d["top_frac"])
            elif "board_top_px" in d and isinstance(d["board_top_px"], (int, float)):
                tf = float(d["board_top_px"]) / float(h)
            else:
                continue
            out.append({
                "name": d.get("name") or f"{w}x{h}",
                "w": w,
                "h": h,
                "top_frac": tf,
                "ar": float(h) / float(w)  # portre oranı (H/W)
            })
        return out
    except Exception:
        return []

def _pick_top_frac_for_image(H, W, devices, ar_tol=0.03):
    """
    Eşleşme stratejisi:
      1) En-boy oranına (H/W) en yakın cihazı bul (|Δar| en küçük).
      2) Eğer |Δar| <= ar_tol ise, eşleşmeyi kabul et.
      3) Aynı |Δar| seviyesinde birden fazla varsa, genişliği en yakın olanı seç.
      4) Şart sağlanmazsa None döndür (fallback devreye girer).
    """
    if not devices:
        return None, None

    ar_img = float(H) / float(W)
    scored = []
    for d in devices:
        dar = abs(d["ar"] - ar_img)
        scored.append((dar, abs(d["w"] - W), d))

    scored.sort(key=lambda x: (x[0], x[1]))
    best_dar, _, best = scored[0]
    if best_dar <= ar_tol:
        return best["top_frac"], best["name"]
    return None, None

# ---------- ROI: sabit oranla kırpma ----------
def board_box_fixed_ratio(full_bgr, top_frac, debug_dir=None, tag="fixed"):
    H, W = full_bgr.shape[:2]
    T = int(round(top_frac * H))
    side = W
    B = T + side
    if B > H:
        shift = B - H
        T = max(0, T - shift)
        B = H
    L, R = 0, W

    if debug_dir:
        ensure_dir(debug_dir)
        vis = full_bgr.copy()
        cv2.rectangle(vis, (L, T), (R, B), (0, 255, 0), 3)
        cv2.imwrite(os.path.join(debug_dir, f"board_box_{tag}.png"), vis)

    return [L, T, R, B], f"{tag}-ratio"

# ---------- taş tespiti (hücre) ----------
def yellow_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array([16,90,110]), np.array([46,255,255]))

def orange_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array([5,120,100]), np.array([22,255,255]))

def tile_bg_ratio(bgr):
    m = cv2.bitwise_or(yellow_mask(bgr), orange_mask(bgr))  # (h, w)
    h, w = m.shape
    b = int(min(h, w) * 0.12)
    if b > 0:
        m[:b, :] = 0
        m[-b:, :] = 0
        m[:, :b] = 0
        m[:, -b:] = 0
    return float(m.mean()) / 255.0

def brown_text_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array([8,60,40]), np.array([30,255,210]))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((2,2),np.uint8), 1)

def letter_stats(bgr):
    m = brown_text_mask(bgr)
    h, w = m.shape
    b = int(min(h, w) * 0.10)
    if b > 0:
        m[:b, :] = 0
        m[-b:, :] = 0
        m[:, :b] = 0
        m[:, -b:] = 0
    cc = cv2.connectedComponentsWithStats(m, 8)
    if cc[0] <= 1:
        return 0.0, 0.0, 0.0
    stats = cc[2]
    idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    x, y, bw, bh, area = stats[idx]
    return area/(h*w), area/(bw*bh + 1e-6), bh/(h + 1e-6)

def white_text_ratio(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S = hsv[..., 1]; V = hsv[..., 2]
    m = ((S < 40) & (V > 210)).astype(np.uint8) * 255
    h, w = m.shape
    b = int(min(h, w) * 0.12)
    if b > 0:
        m[:b, :] = 0
        m[-b:, :] = 0
        m[:, :b] = 0
        m[:, -b:] = 0
    return float(m.mean()) / 255.0

def bonus_color_ratio(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    blue  = cv2.inRange(hsv, np.array([90,60,60]),  np.array([130,255,255]))
    green = cv2.inRange(hsv, np.array([45,40,60]),  np.array([85,255,255]))
    purple= cv2.inRange(hsv, np.array([135,40,60]), np.array([165,255,255]))
    m = cv2.bitwise_or(cv2.bitwise_or(blue, green), purple)
    return float(m.mean())/255.0

def is_real_tile(bgr):
    if tile_bg_ratio(bgr) < 0.11: return False
    a, ext, hf = letter_stats(bgr)
    if not (a >= 0.0045 and ext >= 0.11 and hf >= 0.24): return False
    if bonus_color_ratio(bgr) > 0.06 and a < 0.012: return False
    if white_text_ratio(bgr) > 0.035 and a < 0.012: return False
    return True

# ---------- rack bandı ----------
def auto_find_rack_band(full, L, T, R, B):
    h, w = full.shape[:2]
    hsv = cv2.cvtColor(full, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([10,40,80]), np.array([45,255,255]))
    col = (mask/255).mean(axis=1); col[:B] = 0
    col = cv2.GaussianBlur(col.reshape(-1,1), (41,1), 0).ravel()
    thr = max(0.03, float(col.mean() + col.std()*0.8))
    idx = np.where(col > thr)[0]
    if len(idx) > 0:
        rt = int(idx[0]); rb = int(min(idx[-1] + 12, h-1))
        if rb - rt > 10: return rt, rb
    low = int(h * 0.65)
    col2 = (mask[low:, :] / 255).mean(axis=1)
    col2 = cv2.GaussianBlur(col2.reshape(-1,1), (41,1), 0).ravel()
    thr2 = max(0.03, float(col2.mean() + col2.std()*0.8))
    idx2 = np.where(col2 > thr2)[0]
    if len(idx2) > 0:
        rt2 = low + int(idx2[0]); rb2 = low + int(idx2[-1] + 12)
        if rb2 - rt2 > 10: return rt2, min(rb2, h-1)
    return None, None

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--devices", type=str, default="devices.json",
                    help="Cihaz ayarları JSON yolu")
    ap.add_argument("--force-top-frac", type=float, default=DEFAULT_TOP_FRAC,
                    help="JSON eşleşmesi olmazsa kullanılacak üst boşluk oranı")
    ap.add_argument("--export-only", action="store_true",
                    help="(ignored; compatibility)")
    args = ap.parse_args()

    ensure_dir(args.out)
    cells_dir = os.path.join(args.out, "cells");          ensure_dir(cells_dir)
    occ_dir   = os.path.join(args.out, "cells_occupied"); ensure_dir(occ_dir)
    rack_dir  = os.path.join(args.out, "rack");           ensure_dir(rack_dir)
    dbg_dir   = os.path.join(args.out, "debug") if args.debug else None
    if dbg_dir: ensure_dir(dbg_dir)

    full = cv2.imread(args.image)
    if full is None: raise SystemExit("Görsel açılamadı")
    H, W = full.shape[:2]

    # JSON'dan cihaz seçimi
    top_frac = None
    picked = None
    if args.devices and os.path.isfile(args.devices):
        devices = _load_devices_json(args.devices)
        top_frac, picked = _pick_top_frac_for_image(H, W, devices)

    # Eşleşme yoksa fallback
    if top_frac is None:
        top_frac = float(args.force_top_frac)
        (L, T, R, B), how = board_box_fixed_ratio(full, top_frac, dbg_dir, tag="fallback")
    else:
        (L, T, R, B), how = board_box_fixed_ratio(full, top_frac, dbg_dir, tag=str(picked))

    # ROI (tahta)
    roi = full[T:B, L:R].copy()
    hR, wR = roi.shape[:2]
    if dbg_dir:
        cv2.imwrite(os.path.join(dbg_dir, "board_roi.png"), roi)

    # 15×15 hücreleri yaz
    for r in range(15):
        for c in range(15):
            y0, y1 = int(r*hR/15), int((r+1)*hR/15)
            x0, x1 = int(c*wR/15), int((c+1)*wR/15)
            cell = roi[y0:y1, x0:x1]
            fn = f"r{r:02d}_c{c:02d}.png"
            cv2.imwrite(os.path.join(cells_dir, fn), cell)
            if is_real_tile(cell):
                cv2.imwrite(os.path.join(occ_dir, fn), cell)

    # rack
    rt, rb = auto_find_rack_band(full, L, T, R, B)
    if rt is not None and rb is not None and rb-rt > 10:
        strip = full[rt:rb, :]
        if dbg_dir: cv2.imwrite(os.path.join(dbg_dir, "rack_strip.png"), strip)
        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([10,50,80]), np.array([45,255,255]))
        s1 = mask.sum(axis=0).astype(np.float32)
        col = (s1 - s1.min()) / (np.ptp(s1) + 1e-6)
        on = col > 0.1; ranges = []; s = None
        for i, v in enumerate(on):
            if v and s is None: s = i
            elif not v and s is not None: ranges.append((s, i-1)); s = None
        if s is not None: ranges.append((s, len(on)-1))
        if len(ranges) > 7:
            ranges = sorted(ranges, key=lambda r:r[1]-r[0], reverse=True)[:7]
            ranges = sorted(ranges, key=lambda r:r[0])
        for idx, (x0, x1) in enumerate(ranges):
            tile = strip[:, x0:x1]
            cv2.imwrite(os.path.join(rack_dir, f"r{idx}.png"), tile)

    bridge = {"language": "tr", "board": ["."*15 for _ in range(15)], "rack": ""}
    with open(os.path.join(args.out, "bridge.json"), "w", encoding="utf-8") as f:
        json.dump(bridge, f, ensure_ascii=False, indent=2)

    tag = how
    if picked: tag += f" (device={picked})"
    print(f"Bitti ✅ (JSON cihaz seçimi, {tag}, top_frac={top_frac:.5f})")

if __name__ == "__main__":
    main()
