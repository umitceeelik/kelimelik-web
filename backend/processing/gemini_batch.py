# backend/processing/gemini_batch.py
import io, re, json, math
from typing import List, Dict, Tuple
from PIL import Image, ImageDraw, ImageFont
import google.generativeai as genai

VALID = ["A","B","C","Ç","D","E","F","G","Ğ","H","I","İ","J","K","L",
         "M","N","O","Ö","P","R","S","Ş","T","U","Ü","V","Y","Z"]

PROMPT = f"""You are an OCR expert for single UPPERCASE Turkish characters.
You will receive a contact sheet image with many tiles. Each tile contains EXACTLY ONE letter.
We also provide a legend mapping TILE INDEX -> ORIGINAL FILENAME.

Return ONLY a JSON array of objects: 
  - idx (int), char (one of: {", ".join(VALID)}), conf (0..1 float, 2 decimals)

Rules:
- Distinguish I vs İ, O vs Ö, U vs Ü, C vs Ç, S vs Ş, G vs Ğ.
- If uncertain, pick the most likely and lower confidence.
"""

def _pil_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO(); img.save(buf, fmt); return buf.getvalue()

def _grid_sheet(paths: List[str], cols=12, tile=160, pad=12, cap_h=26) -> Tuple[Image.Image, List[Dict]]:
    rows = math.ceil(len(paths)/cols)
    W = cols*tile + (cols+1)*pad
    H = rows*(tile+cap_h) + (rows+1)*pad
    sheet = Image.new("RGB", (W,H), (250,250,250))
    draw = ImageDraw.Draw(sheet)
    try: font = ImageFont.truetype("arial.ttf", 16)
    except: font = ImageFont.load_default()
    meta=[]
    for i,p in enumerate(paths):
        r=i//cols; c=i%cols
        x = pad + c*(tile+pad)
        y = pad + r*(tile+cap_h+pad)
        try:
            im = Image.open(p).convert("RGB")
        except:
            im = Image.new("RGB",(tile,tile),(240,240,240))
        iw,ih=im.size
        s=min(tile/iw, tile/ih)
        im=im.resize((max(1,int(iw*s)), max(1,int(ih*s))), Image.Resampling.BICUBIC)
        ox = x + (tile-im.size[0])//2
        oy = y + (tile-im.size[1])//2
        sheet.paste(im,(ox,oy))
        # index & caption
        draw.rectangle([x,y,x+38,y+20], fill=(0,0,0))
        draw.text((x+5,y+2), str(i), fill=(255,255,255), font=font)
        draw.text((x, y+tile+4), p.split("/")[-1], fill=(0,0,0), font=font)
        meta.append({"idx": i, "path": p})
    return sheet, meta

def _legend(meta: List[Dict]) -> str:
    return "Legend (idx -> filename):\n" + "\n".join(f"{m['idx']}: {m['path'].split('/')[-1]}" for m in meta)

def classify_with_gemini(paths: List[str], api_key: str, model="models/gemini-2.0-flash",
                         cols=12, tile=160) -> Dict[str, Tuple[str, float]]:
    """Returns {path: (char, conf)}"""
    if not paths: return {}
    genai.configure(api_key=api_key)
    out: Dict[str, Tuple[str,float]] = {}
    # path sayısı çoksa parçala
    chunk_size = cols * 6
    chunks = [paths[i:i+chunk_size] for i in range(0,len(paths),chunk_size)]
    for chunk in chunks:
        sheet, meta = _grid_sheet(chunk, cols=cols, tile=tile)
        legend = _legend(meta)
        image_part = {"mime_type":"image/png", "data": _pil_bytes(sheet)}
        resp = genai.GenerativeModel(model).generate_content(
            [PROMPT, legend, image_part], generation_config={"temperature":0.0}
        )
        text = getattr(resp, "text", None) or resp.candidates[0].content.parts[0].text
        m = re.search(r"\[.*\]", text, re.DOTALL)
        data = json.loads(m.group(0) if m else text)
        idx_map = {m["idx"]: m["path"] for m in meta}
        for item in data:
            idx = int(item.get("idx"))
            ch  = str(item.get("char","")).strip().upper()
            conf= float(item.get("conf",0))
            # güvenlik
            if ch not in VALID: continue
            path = idx_map.get(idx)
            if path: out[path] = (ch, max(0.0,min(1.0,conf)))
        # eksikler olursa boş bırakma
        for m in meta:
            out.setdefault(m["path"], ("?", 0.0))
    return out
