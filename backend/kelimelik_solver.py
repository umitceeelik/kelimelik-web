# backend/kelimelik_solver.py
# JS kelimelik-solver.js'in birebir Python portu
import json
from pathlib import Path
from collections import Counter

# Bu dosya config ve sözlüğü frontend klasöründen okur
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
CFG_PATH = FRONTEND_DIR / "kelimelik-config.json"
DICT_PATH = FRONTEND_DIR / "tr-dictionary.json"

# ---- Global config ----
SIZE = 15
EMPTY = '.'
BLANK = '*'   # <— JS ile aynı: joker = '*'
BINGO_SCORE = 50
LETTER_SCORES = {}
PREMIUM = None
CENTER_STAR = {"row": 7, "col": 7}

# ---- Load config/letters/premiums once ----
def _load_config_once():
    global SIZE, EMPTY, BLANK, BINGO_SCORE, LETTER_SCORES, PREMIUM, CENTER_STAR
    with CFG_PATH.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    SIZE = int(cfg.get("SIZE", 15))
    EMPTY = cfg.get("EMPTY", ".")
    BLANK = cfg.get("BLANK", "*")
    BINGO_SCORE = int(cfg.get("BINGO_SCORE", 50))
    LETTER_SCORES.clear()
    LETTER_SCORES.update(cfg.get("letterScores", {}))
    CENTER_STAR = cfg.get("centerStar", {"row": 7, "col": 7})
    PREMIUM = [[None for _ in range(SIZE)] for _ in range(SIZE)]
    for p in cfg.get("premiumSquares", []):
        PREMIUM[p["row"]][p["col"]] = p["type"]

CFG_LOADED = False
DICT_SET = None

def load_dictionary():
    global DICT_SET, CFG_LOADED
    if not CFG_LOADED:
        _load_config_once()
    if DICT_SET is not None:
        return DICT_SET
    with DICT_PATH.open("r", encoding="utf-8") as f:
        words = json.load(f)
    # Büyük harfe çevir, boş/geçersizi at
    DICT_SET = set(w.strip().upper() for w in words if w and isinstance(w, str))
    return DICT_SET

# ---- Helpers ----
def in_bounds(r, c):
    return 0 <= r < SIZE and 0 <= c < SIZE

def score_letter(ch):
    # joker harfi puanlandırma 0
    if ch == BLANK:
        return 0
    return int(LETTER_SCORES.get(ch, 0))

def board_empty(b):
    return all(all(ch == EMPTY for ch in row) for row in b)

def contiguous_ends_clear(b, r0, c0, dr, dc, length):
    br, bc = r0 - dr, c0 - dc
    ar, ac = r0 + dr * length, c0 + dc * length
    before_ok = (not in_bounds(br, bc)) or (b[br][bc] == EMPTY)
    after_ok  = (not in_bounds(ar, ac)) or (b[ar][ac] == EMPTY)
    return before_ok and after_ok

def extend(b, r, c, dr, dc):
    rs, cs, re, ce = r, c, r, c
    while in_bounds(rs - dr, cs - dc) and b[rs - dr][cs - dc] != EMPTY:
        rs -= dr; cs -= dc
    while in_bounds(re + dr, ce + dc) and b[re + dr][ce + dc] != EMPTY:
        re += dr; ce += dc
    chars = []
    rr, cc = rs, cs
    while True:
        chars.append(b[rr][cc])
        if rr == re and cc == ce:
            break
        rr += dr; cc += dc
    return {"rs": rs, "cs": cs, "re": re, "ce": ce, "word": "".join(chars)}

def has_perp_neighbor(b, r, c, dr, dc):
    pr, pc = dc, dr
    r1, c1 = r - pr, c - pc
    r2, c2 = r + pr, c + pc
    return (in_bounds(r1, c1) and b[r1][c1] != EMPTY) or (in_bounds(r2, c2) and b[r2][c2] != EMPTY)

def score_main_word(b, placements, r0, c0, dr, dc, length):
    total, mul = 0, 1
    for i in range(length):
        rr = r0 + dr * i
        cc = c0 + dc * i
        old = (b[rr][cc] != EMPTY)
        pl = next((p for p in placements if p["r"] == rr and p["c"] == cc), None)
        ch = b[rr][cc] if old else pl["ch"]
        base = score_letter(BLANK if (pl and pl.get("usedBlank")) else ch)
        if old:
            total += base
        else:
            prem = PREMIUM[rr][cc]
            if prem == "DL": total += base * 2
            elif prem == "TL": total += base * 3
            else: total += base
            if prem == "DW": mul *= 2
            if prem == "TW": mul *= 3
    return total * mul

def score_cross_word(b, r, c, dr, dc, placed_ch, used_blank):
    pr, pc = dc, dr
    ext = extend(b, r, c, pr, pc)
    # kelimeyi oluştur (merkezde placed_ch kullan)
    total, mul = 0, 1
    word_chars = []
    rr, cc = ext["rs"], ext["cs"]
    while True:
        is_new = (rr == r and cc == c)
        ch = placed_ch if is_new else b[rr][cc]
        word_chars.append(ch)
        base = score_letter(BLANK if (is_new and used_blank) else ch)
        if is_new:
            prem = PREMIUM[rr][cc]
            if prem == "DL": total += base * 2
            elif prem == "TL": total += base * 3
            else: total += base
            if prem == "DW": mul *= 2
            if prem == "TW": mul *= 3
        else:
            total += base
        if rr == ext["re"] and cc == ext["ce"]: break
        rr += pr; cc += pc
    word = "".join(word_chars)
    if len(word) <= 1:
        return {"ok": True, "score": 0, "word": None}
    return {"ok": True, "score": total * mul, "word": word}

def take_from_rack(rack_counter: Counter, need_ch: str):
    have = rack_counter.get(need_ch, 0)
    if have > 0:
        rack_counter[need_ch] -= 1
        return {"usedBlank": False, "placed": need_ch}
    blanks = rack_counter.get(BLANK, 0)
    if blanks > 0:
        rack_counter[BLANK] -= 1
        return {"usedBlank": True, "placed": need_ch}
    return None

# ---- Placement routines ----
def try_place(b, rack, word, rA, cA, direction, k, dict_has):
    dr = 1 if direction == "V" else 0
    dc = 1 if direction == "H" else 0
    L = len(word)
    r0, c0 = rA - dr * k, cA - dc * k
    if not (in_bounds(r0, c0) and in_bounds(r0 + dr * (L - 1), c0 + dc * (L - 1))):
        return None
    if not contiguous_ends_clear(b, r0, c0, dr, dc, L):
        return None

    rack_counter = Counter(rack)
    used_existing = False
    used_from_rack = 0
    placements = []

    for i in range(L):
        rr = r0 + dr * i
        cc = c0 + dc * i
        need = word[i]
        old = b[rr][cc]
        if old != EMPTY:
            if old != need:
                return None
            used_existing = True
        else:
            t = take_from_rack(rack_counter, need)
            if not t:
                return None
            used_from_rack += 1
            placements.append({"r": rr, "c": cc, "ch": need, "usedBlank": t["usedBlank"]})

    if used_from_rack == 0:
        return None

    if (not board_empty(b)) and (not used_existing):
        # en az bir yerleştirilen taşın dik komşusu dolu olmalı
        if not any(has_perp_neighbor(b, p["r"], p["c"], dr, dc) for p in placements):
            return None

    # ana kelime string'i
    main_word_chars = []
    for i in range(L):
        rr = r0 + dr * i
        cc = c0 + dc * i
        ch = b[rr][cc] if b[rr][cc] != EMPTY else word[i]
        main_word_chars.append(ch)
    main_word = "".join(main_word_chars)
    if not dict_has(main_word):
        return None

    total = score_main_word(b, placements, r0, c0, dr, dc, L)

    # çaprazlar
    for p in placements:
        cw = score_cross_word(b, p["r"], p["c"], dr, dc, p["ch"], p["usedBlank"])
        if cw["word"] and len(cw["word"]) > 1:
            if not dict_has(cw["word"]):
                return None
            total += cw["score"]

    if used_from_rack >= 7:
        total += BINGO_SCORE

    return {
        "word": main_word, "row": r0, "col": c0, "dir": direction,
        "score": total, "placed": placements
    }

def try_place_first(b, rack, word, dict_has):
    out = []
    centerR = int(CENTER_STAR["row"])
    centerC = int(CENTER_STAR["col"])
    for direction in ("H", "V"):
        dr = 1 if direction == "V" else 0
        dc = 1 if direction == "H" else 0
        L = len(word)
        for k in range(L):
            r0 = centerR - dr * k
            c0 = centerC - dc * k
            if not (in_bounds(r0, c0) and in_bounds(r0 + dr * (L - 1), c0 + dc * (L - 1))):
                continue
            if not contiguous_ends_clear(b, r0, c0, dr, dc, L):
                continue
            rack_counter = Counter(rack)
            placements = []
            fail = False
            for i in range(L):
                rr = r0 + dr * i
                cc = c0 + dc * i
                need = word[i]
                t = take_from_rack(rack_counter, need)
                if not t:
                    fail = True; break
                placements.append({"r": rr, "c": cc, "ch": need, "usedBlank": t["usedBlank"]})
            if fail:
                continue
            if not any(p["r"] == centerR and p["c"] == centerC for p in placements):
                continue
            if not dict_has(word):
                continue
            total = score_main_word(b, placements, r0, c0, dr, dc, L)
            if L >= 7: total += BINGO_SCORE
            out.append({"word": word, "row": r0, "col": c0, "dir": direction, "score": total, "placed": placements})
    return out

def generate_parallel_moves(board, rack, dict_by_len, dict_has):
    results = []
    MAX_USE = min(7, len(rack))

    def scan_direction(direction):
        dr = 0 if direction == "H" else 1
        dc = 1 if direction == "H" else 0
        for line in range(SIZE):
            def get(i):
                return board[line][i] if direction == "H" else board[i][line]
            s = -1
            for i in range(SIZE + 1):
                ch = get(i) if i < SIZE else "#"
                empty = (i < SIZE) and (ch == EMPTY)
                if empty:
                    if s == -1: s = i
                elif s != -1:
                    e = i - 1
                    seg_len = e - s + 1
                    # segmentte dikine komşuluk var mı?
                    touches = False
                    for t in range(s, e + 1):
                        r = line if direction == "H" else t
                        c = t if direction == "H" else line
                        if has_perp_neighbor(board, r, c, dr, dc):
                            touches = True; break
                    if touches and seg_len >= 2:
                        for o in range(s, e + 1):
                            maxL = min(MAX_USE, e - o + 1)
                            for L in range(2, maxL + 1):
                                # alt aralık içinde de en az bir dik komşu temas etsin
                                sub_touches = False
                                for t in range(o, o + L):
                                    r = line if direction == "H" else t
                                    c = t    if direction == "H" else line
                                    if has_perp_neighbor(board, r, c, dr, dc):
                                        sub_touches = True; break
                                if not sub_touches:
                                    continue
                                r0 = line if direction == "H" else o
                                c0 = o    if direction == "H" else line
                                words = dict_by_len.get(L, [])
                                for w in words:
                                    cand = try_place(board, rack, w, r0, c0, direction, 0, dict_has)
                                    if cand: results.append(cand)
                    s = -1

    scan_direction("H")
    scan_direction("V")
    return results

# ---- Public API ----
def solve_board(board_rows, rack_string):
    """
    board_rows: 15x15 matrix (list of lists) harfler ('.' boş)
    rack_string: eldeki taşlar (joker = '*')
    """
    dict_set = load_dictionary()
    # normalize board to uppercase
    board = [[(ch or EMPTY).upper() for ch in row] for row in board_rows]
    rack = (rack_string or "").upper()

    # indeksler
    index_by_char = {}   # harfe göre
    dict_by_len = {}     # uzunluğa göre
    for w in dict_set:
        L = len(w)
        if L < 2 or L > SIZE: 
            continue
        for ch in set(w):
            index_by_char.setdefault(ch, []).append(w)
        dict_by_len.setdefault(L, []).append(w)

    dict_has = dict_set.__contains__
    results = []

    if board_empty(board):
        # ilk hamle: merkez yıldız kapatmalı
        for w in dict_set:
            results.extend(try_place_first(board, rack, w, dict_has))
    else:
        # bindirmeli: tahtadaki mevcut harfleri anchor al
        anchors = []
        for r in range(SIZE):
            for c in range(SIZE):
                if board[r][c] != EMPTY:
                    anchors.append({"r": r, "c": c, "ch": board[r][c]})
        for a in anchors:
            ch = a["ch"]
            cand_words = index_by_char.get(ch, [])
            for w in cand_words:
                for k, ck in enumerate(w):
                    if ck != ch:
                        continue
                    h = try_place(board, rack, w, a["r"], a["c"], "H", k, dict_has)
                    if h: results.append(h)
                    v = try_place(board, rack, w, a["r"], a["c"], "V", k, dict_has)
                    if v: results.append(v)
        # paralel (boş segment içinde alt aralık)
        results.extend(generate_parallel_moves(board, rack, dict_by_len, dict_has))

    # dedup + sırala
    seen = set()
    dedup = []
    for r in results:
        key = f"{r['word']}|{r['row']}|{r['col']}|{r['dir']}"
        if key not in seen:
            seen.add(key)
            dedup.append(r)
    dedup.sort(key=lambda x: (x["score"], len(x["word"])), reverse=True)
    return dedup
