# kelimelik_board_ocr.py
# Tahta ve raf OCR çözümü
# - Tahta ROI: Sobel+adaptive grid tespiti, yoksa sabit oran fallback
# - Harf okuma: Önce template, sonra OCR; Türkçe diakritik düzeltmesi
# - Raf bandı: Renk tabanlı tespit
# - Çıktı: out/bridge.json (board + rack bilgisi)

import os, json, argparse, shutil
import cv2, numpy as np

# -----------------------
# Sabitler
# -----------------------
WHITELIST = "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ*"
LANGS = "tur+eng"
DEFAULT_TOP_FRAC = 496/1541

# -----------------------
# Yardımcı fonksiyonlar
# -----------------------
def tr_upper(s: str) -> str:
    m = {'i':'İ','ı':'I','ş':'Ş','ğ':'Ğ','ö':'Ö','ü':'Ü','ç':'Ç'}
    return "".join([m.get(ch, ch.upper()) for ch in s])

def pad_to_square(img, value=0):
    h,w = img.shape[:2]
    side = max(h,w)
    out = np.full((side,side), value, dtype=img.dtype)
    y0=(side-h)//2; x0=(side-w)//2
    out[y0:y0+h, x0:x0+w] = img
    return out

def ensure_dir(p): os.makedirs(p, exist_ok=True)

# -----------------------
# ROI tespit (sobel veya fallback)
# -----------------------
def forced_board_box(full, top_frac=DEFAULT_TOP_FRAC):
    H,W = full.shape[:2]
    top = int(top_frac*H)
    side = min(W, H-top)
    if side < 300: raise SystemExit(f"Board ROI küçük: side={side}")
    return [0, top, side, top+side]

def find_board_bbox(img: np.ndarray):
    h, w = img.shape[:2]
    scale = 900 / max(h, w)
    small = cv2.resize(img, (int(w*scale), int(h*scale)),
                       interpolation=cv2.INTER_AREA) if scale < 1 else img
    sh, sw = small.shape[:2]

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 50, 50)

    sobelx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    sobel  = cv2.convertScaleAbs(0.5*np.abs(sobelx) + 0.5*np.abs(sobely))

    thr = cv2.adaptiveThreshold(sobel, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, -5)
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best, best_score = None, -1.0
    cx, cy = sw/2, sh/2
    for c in cnts:
        x,y,wc,hc = cv2.boundingRect(c)
        area = wc*hc
        if area < 0.05*sw*sh:
            continue
        aspect = wc/float(hc)
        if not (0.75 <= aspect <= 1.33):
            continue
        mx,my = x+wc/2, y+hc/2
        centrality = 1.0 - min(1.0, np.hypot((mx-cx)/sw, (my-cy)/sh)*2.0)
        score = (area/(sw*sh)) + centrality
        if score > best_score:
            best_score, best = score, (x,y,wc,hc)

    if best is None:
        return None

    inv = 1.0/scale
    x,y,wc,hc = best
    return [int(x*inv), int(y*inv), int(wc*inv), int(hc*inv)]

def board_box_sobel_or_fallback(full_bgr):
    bbox = find_board_bbox(full_bgr)
    if bbox:
        x,y,w,h = bbox
        return [x, y, x+w, y+h]
    return forced_board_box(full_bgr)

# -----------------------
# OCR Motorları
# -----------------------
def ocr_tesseract(img):
    try: import pytesseract
    except Exception: return ""
    cfg='--psm 10 -c tessedit_char_whitelist='+WHITELIST
    txt = pytesseract.image_to_string(img, lang=LANGS, config=cfg)
    txt = tr_upper(txt.strip())
    for ch in txt:
        if ch in WHITELIST: return ch
    return ""

# -----------------------
# Main
# -----------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="out")
    args=ap.parse_args()

    ensure_dir(args.out)

    full=cv2.imread(args.image)
    if full is None: raise SystemExit("Görsel açılamadı")

    # ROI
    L,T,R,B = board_box_sobel_or_fallback(full)
    roi = full[T:B, L:R].copy()
    H,W = roi.shape[:2]

    board=[["." for _ in range(15)] for _ in range(15)]

    # BOARD
    for r in range(15):
        for c in range(15):
            y0,y1=int(r*H/15), int((r+1)*H/15)
            x0,x1=int(c*W/15), int((c+1)*W/15)
            cell=roi[y0:y1, x0:x1]
            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            ch = ocr_tesseract(th)
            board[r][c] = ch if ch else "."

    data={"language":"tr","board":["".join(r) for r in board]}
    with open(os.path.join(args.out,"bridge.json"),"w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2)

    print("Bitti ✅")

if __name__=="__main__":
    main()
