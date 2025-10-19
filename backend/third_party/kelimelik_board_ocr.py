# backend/third_party/kelimelik_board_ocr.py
# Yalnızca kırpma/segmentasyon (OCR YOK):
# - Tahta ROI (Sobel, yoksa sabit oran fallback)
# - 15x15 hücreleri out/cells/ altına yaz
# - Taş içerenleri out/cells_occupied/ altına da kopyala
# - Rack bandını bulup out/rack/ altına tek tek yaz
# - --export-only argümanı kabul edilir (uyumluluk için yok sayılır)

import os, cv2, json, math, argparse, numpy as np

DEFAULT_TOP_FRAC = 496/1541  # fallback ROI üst boşluk oranı

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def pad_to_square(img, value=0):
    h,w = img.shape[:2]; side = max(h,w)
    out = np.full((side,side), value, dtype=img.dtype)
    y0=(side-h)//2; x0=(side-w)//2
    out[y0:y0+h, x0:x0+w] = img
    return out

# ---------- ROI: fallback (üstten kare) ----------
def forced_board_box(full, top_px=None, top_frac=DEFAULT_TOP_FRAC, debug_dir=None):
    H,W = full.shape[:2]
    top = int(top_px if top_px is not None else max(0.0, min(1.0, top_frac))*H)
    side = min(W, H-top)
    if side < 300: raise SystemExit(f"Board ROI küçük: side={side}")
    L,T,R,B = 0, top, side, top+side
    if debug_dir:
        vis = full.copy(); cv2.rectangle(vis,(L,T),(R,B),(0,255,0),3)
        ensure_dir(debug_dir); cv2.imwrite(os.path.join(debug_dir,"board_box_forced.png"), vis)
    return [L,T,R,B]

# ---------- ROI: Sobel tabanlı ----------
def find_board_bbox(img):
    h,w = img.shape[:2]
    scale = 900 / max(h,w)
    small = cv2.resize(img,(int(w*scale),int(h*scale)),cv2.INTER_AREA) if scale<1 else img
    sh,sw = small.shape[:2]
    gray = cv2.bilateralFilter(cv2.cvtColor(small,cv2.COLOR_BGR2GRAY), 9, 50, 50)
    sx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
    sob = cv2.convertScaleAbs(0.5*np.abs(sx)+0.5*np.abs(sy))
    thr = cv2.adaptiveThreshold(sob,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,-5)
    k = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    thr = cv2.morphologyEx(cv2.morphologyEx(thr,cv2.MORPH_CLOSE,k,2), cv2.MORPH_OPEN,k,1)
    cnts,_ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best,score = None,-1.0; cx,cy = sw/2, sh/2
    for c in cnts:
        x,y,wc,hc = cv2.boundingRect(c)
        area = wc*hc
        if area < 0.05*sw*sh: continue
        aspect = wc/float(hc)
        squareish = 0.75 <= aspect <= 1.33
        mx,my = x+wc/2, y+hc/2
        centrality = 1.0 - min(1.0, math.hypot((mx-cx)/sw,(my-cy)/sh)*2.0)
        density = cv2.mean(thr[y:y+hc,x:x+wc])[0]/255.0
        sc = (area/(sw*sh))*1.5 + (1-abs(1-aspect))*0.8 + centrality*0.7 + density*0.6
        if squareish: sc += 0.5
        if sc > score: score, best = sc, (x,y,wc,hc)
    if best is None: return None, None

    inv = 1.0/scale
    x,y,wc,hc = best
    return (int(x*inv),int(y*inv),int(wc*inv),int(hc*inv)), {"thr":thr,"small":small,"rect":best}

def board_box_sobel_or_fallback(full_bgr, force_top_px, force_top_frac, debug_dir):
    bbox, dbg = find_board_bbox(full_bgr)
    if bbox is not None:
        x,y,w,h = bbox; pad=6
        L=max(0,x-pad); T=max(0,y-pad)
        R=min(full_bgr.shape[1], x+w+pad)
        B=min(full_bgr.shape[0], y+h+pad)
        side=min(R-L,B-T); cx=(L+R)//2; cy=(T+B)//2
        L=max(0,cx-side//2); R=L+side; T=max(0,cy-side//2); B=T+side
        if debug_dir:
            ensure_dir(debug_dir)
            vis_small = dbg["small"].copy(); sx,sy,sw,sh = dbg["rect"]
            cv2.rectangle(vis_small,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)
            cv2.imwrite(os.path.join(debug_dir,"sobel_small_rect.png"), vis_small)
            cv2.imwrite(os.path.join(debug_dir,"sobel_thr.png"), dbg["thr"])
            vis = full_bgr.copy(); cv2.rectangle(vis,(L,T),(R,B),(255,0,0),3)
            cv2.imwrite(os.path.join(debug_dir,"board_box_sobel.png"), vis)
        return [L,T,R,B], "sobel"
    return forced_board_box(full_bgr, force_top_px, force_top_frac, debug_dir), "forced"

# ---------- taş tespiti (hücre) ----------
def yellow_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array([16,90,110]), np.array([46,255,255]))

def orange_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array([5,120,100]), np.array([22,255,255]))

def tile_bg_ratio(bgr):
    m = cv2.bitwise_or(yellow_mask(bgr), orange_mask(bgr))
    h,w=m.shape; b=int(min(h,w)*0.12)
    if b>0: m[:b,:]=0; m[-b:,:]=0; m[:,:b]=0; m[:,-b:]=0
    return float(m.mean())/255.0

def brown_text_mask(bgr):
    hsv=cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m=cv2.inRange(hsv, np.array([8,60,40]), np.array([30,255,210]))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((2,2),np.uint8),1)

def letter_stats(bgr):
    m=brown_text_mask(bgr); h,w=m.shape; b=int(min(h,w)*0.10)
    if b>0: m[:b,:]=0; m[-b:,:]=0; m[:,:b]=0; m[:,-b:]=0
    cc=cv2.connectedComponentsWithStats(m,8)
    if cc[0]<=1: return 0.0,0.0,0.0
    stats=cc[2]; idx=1+np.argmax(stats[1:,cv2.CC_STAT_AREA])
    x,y,bw,bh,area=stats[idx]
    return area/(h*w), area/(bw*bh+1e-6), bh/(h+1e-6)

def white_text_ratio(bgr):
    hsv=cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV); S=hsv[...,1]; V=hsv[...,2]
    m=((S<40)&(V>210)).astype(np.uint8)*255
    h,w=m.shape; b=int(min(h,w)*0.12)
    if b>0: m[:b,:]=0; m[-b:,:]=0; m[:,:b]=0; m[:,-b:]=0
    return float(m.mean())/255.0

def bonus_color_ratio(bgr):
    hsv=cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    blue=cv2.inRange(hsv, np.array([90,60,60]), np.array([130,255,255]))
    green=cv2.inRange(hsv, np.array([45,40,60]), np.array([85,255,255]))
    purple=cv2.inRange(hsv, np.array([135,40,60]), np.array([165,255,255]))
    m=cv2.bitwise_or(cv2.bitwise_or(blue,green),purple)
    return float(m.mean())/255.0

def is_real_tile(bgr):
    if tile_bg_ratio(bgr) < 0.11: return False
    a,ext,hf = letter_stats(bgr)
    if not (a>=0.0045 and ext>=0.11 and hf>=0.24): return False
    if bonus_color_ratio(bgr)>0.06 and a<0.012: return False
    if white_text_ratio(bgr)>0.035 and a<0.012: return False
    return True

# ---------- rack bandı ----------
def auto_find_rack_band(full, L,T,R,B):
    h,w=full.shape[:2]
    hsv=cv2.cvtColor(full, cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv, np.array([10,40,80]), np.array([45,255,255]))
    col=(mask/255).mean(axis=1); col[:B]=0
    col=cv2.GaussianBlur(col.reshape(-1,1),(41,1),0).ravel()
    thr=max(0.03, float(col.mean()+col.std()*0.8))
    idx=np.where(col>thr)[0]
    if len(idx)>0:
        rt=int(idx[0]); rb=int(min(idx[-1]+12, h-1))
        if rb-rt>10: return rt, rb
    low=int(h*0.65)
    col2=(mask[low:,:]/255).mean(axis=1)
    col2=cv2.GaussianBlur(col2.reshape(-1,1),(41,1),0).ravel()
    thr2=max(0.03, float(col2.mean()+col2.std()*0.8))
    idx2=np.where(col2>thr2)[0]
    if len(idx2)>0:
        rt2=low+int(idx2[0]); rb2=low+int(idx2[-1]+12)
        if rb2-rt2>10: return rt2, min(rb2,h-1)
    return None, None

# ---------- main ----------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--force-top-px", type=int, default=None)
    ap.add_argument("--force-top-frac", type=float, default=DEFAULT_TOP_FRAC)
    # Uyumluluk için: app.py --export-only gönderiyor; burada kabul edip YOK SAYIYORUZ
    ap.add_argument("--export-only", action="store_true", help="(ignored; compatibility)")
    args=ap.parse_args()

    ensure_dir(args.out)
    cells_dir=os.path.join(args.out,"cells");          ensure_dir(cells_dir)
    occ_dir  =os.path.join(args.out,"cells_occupied"); ensure_dir(occ_dir)
    rack_dir =os.path.join(args.out,"rack");           ensure_dir(rack_dir)
    dbg_dir  =os.path.join(args.out,"debug") if args.debug else None
    if dbg_dir: ensure_dir(dbg_dir)

    full=cv2.imread(args.image)
    if full is None: raise SystemExit("Görsel açılamadı")

    (L,T,R,B), how = board_box_sobel_or_fallback(full, args.force_top_px, args.force_top_frac, dbg_dir)
    roi = full[T:B, L:R].copy()
    H,W = roi.shape[:2]
    if dbg_dir: cv2.imwrite(os.path.join(dbg_dir,"board_roi.png"), roi)

    # 15x15 hücreler
    for r in range(15):
        for c in range(15):
            y0,y1=int(r*H/15), int((r+1)*H/15)
            x0,x1=int(c*W/15), int((c+1)*W/15)
            cell=roi[y0:y1, x0:x1]
            fn=f"r{r:02d}_c{c:02d}.png"
            cv2.imwrite(os.path.join(cells_dir, fn), cell)
            if is_real_tile(cell):
                cv2.imwrite(os.path.join(occ_dir, fn), cell)

    # rack
    rt,rb = auto_find_rack_band(full, L,T,R,B)
    if rt is not None and rb is not None and rb-rt>10:
        strip=full[rt:rb,:]
        if dbg_dir: cv2.imwrite(os.path.join(dbg_dir,"rack_strip.png"), strip)
        hsv=cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(hsv, np.array([10,50,80]), np.array([45,255,255]))
        s1=mask.sum(axis=0).astype(np.float32)
        col=(s1 - s1.min())/(np.ptp(s1)+1e-6)
        on=col>0.1; ranges=[]; s=None
        for i,v in enumerate(on):
            if v and s is None: s=i
            elif not v and s is not None: ranges.append((s,i-1)); s=None
        if s is not None: ranges.append((s,len(on)-1))
        if len(ranges)>7:
            ranges=sorted(ranges,key=lambda r:r[1]-r[0], reverse=True)[:7]
            ranges=sorted(ranges,key=lambda r:r[0])
        for idx,(x0,x1) in enumerate(ranges):
            tile=strip[:, x0:x1]
            cv2.imwrite(os.path.join(rack_dir, f"r{idx}.png"), tile)

    # Bridge (opsiyonel; boş)
    bridge={"language":"tr","board":["."*15 for _ in range(15)],"rack":""}
    with open(os.path.join(args.out,"bridge.json"),"w",encoding="utf-8") as f:
        json.dump(bridge,f,ensure_ascii=False,indent=2)

    print("Bitti ✅ (yalnızca kırpma/segmentasyon)")

if __name__=="__main__":
    main()
