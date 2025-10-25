// kelimelik-solver.js — ES module
// configure(config) çağrısı sonrası solveBoard() kullanılmalı.

let CONFIG = null;
let SIZE, EMPTY, BLANK, BINGO_SCORE, letterScores, PREMIUM, centerStar;

export function configure(config) {
    CONFIG = config;
    SIZE = CONFIG.SIZE;
    EMPTY = CONFIG.EMPTY;
    BLANK = CONFIG.BLANK;
    BINGO_SCORE = CONFIG.BINGO_SCORE;
    letterScores = CONFIG.letterScores;
    centerStar = CONFIG.centerStar; // { row, col }

    PREMIUM = Array.from({ length: SIZE }, () => Array(SIZE).fill(null));
    for (const p of CONFIG.premiumSquares) PREMIUM[p.row][p.col] = p.type;
}

// ---------- yardımcılar ----------

const inBounds = (r, c) => r >= 0 && r < SIZE && c >= 0 && c < SIZE;
const scoreLetter = (ch) => (ch === BLANK ? 0 : (letterScores[ch] || 0));
const boardEmpty = (b) => b.every(row => row.every(ch => ch === EMPTY));

const multiset = (str) => {
    const m = new Map();
    for (const ch of str) m.set(ch, (m.get(ch) || 0) + 1);
    return m;
};

const takeFromRack = (rackMap, needCh) => {
    const have = rackMap.get(needCh) || 0;
    if (have > 0) {
        rackMap.set(needCh, have - 1);
        return { usedBlank: false, placed: needCh };
    }
    const blanks = rackMap.get(BLANK) || 0;
    if (blanks > 0) {
        rackMap.set(BLANK, blanks - 1);
        return { usedBlank: true, placed: needCh };
    }
    return null;
};

// board boyunca aynı doğrultuda (dr,dc) birleşik kelimeyi (mevcut + yerleşecek yeri "dolu" varsayarak sınırları bulmak için) çıkar.
// Not: Buradaki `extend` yalnızca sınırları bulmak için kullanılıyor; kelime string'i/puanı başka yerde derlenir.
function extend(b, r, c, dr, dc) {
    let rs = r, cs = c, re = r, ce = c;
    while (inBounds(rs - dr, cs - dc) && b[rs - dr][cs - dc] !== EMPTY) { rs -= dr; cs -= dc; }
    while (inBounds(re + dr, ce + dc) && b[re + dr][ce + dc] !== EMPTY) { re += dr; ce += dc; }
    const chars = [];
    for (let rr = rs, cc = cs; ; rr += dr, cc += dc) {
        chars.push(b[rr][cc]);
        if (rr === re && cc === ce) break;
    }
    return { rs, cs, re, ce, word: chars.join('') };
}

const contiguousEndsClear = (b, r0, c0, dr, dc, len) => {
    const br = r0 - dr, bc = c0 - dc, ar = r0 + dr * len, ac = c0 + dc * len;
    const beforeOK = !inBounds(br, bc) || b[br][bc] === EMPTY;
    const afterOK = !inBounds(ar, ac) || b[ar][ac] === EMPTY;
    return beforeOK && afterOK;
};

// En az bir dik komşu var mı? (bindirmesiz hamlelerde "temas" şartı)
const hasPerpNeighbor = (b, r, c, dr, dc) => {
    // Ana doğrultu (dr,dc) ise dik doğrultu (dc,dr)
    const pr = dc, pc = dr;
    const r1 = r - pr, c1 = c - pc;
    const r2 = r + pr, c2 = c + pc;
    return (inBounds(r1, c1) && b[r1][c1] !== EMPTY) || (inBounds(r2, c2) && b[r2][c2] !== EMPTY);
};

// ---------- puanlama ----------

// Ana kelime puanı (yerleştirilen harflerde primler aktif)
function scoreMainWord(b, placements, r0, c0, dr, dc, len) {
    let sum = 0, mul = 1;
    for (let i = 0; i < len; i++) {
        const rr = r0 + dr * i, cc = c0 + dc * i;
        const old = b[rr][cc] !== EMPTY;
        const pl = placements.find(p => p.r === rr && p.c === cc);
        const ch = old ? b[rr][cc] : pl.ch;

        const base = scoreLetter(pl && pl.usedBlank ? BLANK : ch);
        if (old) {
            sum += base;
        } else {
            const prem = PREMIUM[rr][cc];
            if (prem === 'DL') sum += base * 2;
            else if (prem === 'TL') sum += base * 3;
            else sum += base;

            if (prem === 'DW') mul *= 2;
            if (prem === 'TW') mul *= 3;
        }
    }
    return sum * mul;
}

// Çapraz kelimeyi doğru oluşturup puanlayan fonksiyon.
// Burada (r,c) konumuna yeni yerleştirilen karakter `placedCh`'dir.
function scoreCrossWord(b, r, c, dr, dc, placedCh, usedBlank) {
    // Ana doğrultu dr/dc; çapraz doğrultu:
    const pr = dc, pc = dr;

    // Sınırları mevcut tahtaya göre bul (merkezi boş olsa bile sınırlar doğru çıkar)
    const { rs, cs, re, ce } = extend(b, r, c, pr, pc);

    // Kelimeyi string olarak inşa et (merkezde yeni harfi kullan!)
    let sum = 0, mul = 1;
    let wordChars = [];

    for (let rr = rs, cc = cs; ; rr += pr, cc += pc) {
        const isNew = (rr === r && cc === c);
        const ch = isNew ? placedCh : b[rr][cc];
        wordChars.push(ch);

        const base = scoreLetter(isNew && usedBlank ? BLANK : ch);
        if (isNew) {
            const prem = PREMIUM[rr][cc];
            if (prem === 'DL') sum += base * 2;
            else if (prem === 'TL') sum += base * 3;
            else sum += base;

            if (prem === 'DW') mul *= 2;
            if (prem === 'TW') mul *= 3;
        } else {
            sum += base;
        }

        if (rr === re && cc === ce) break;
    }

    const word = wordChars.join('');
    if (word.length <= 1) return { ok: true, score: 0, word: null };
    return { ok: true, score: sum * mul, word };
}

// ---------- tek yerleştirme (H veya V) ----------

function tryPlace(b, rack, word, rA, cA, dir, k, dictHas) {
    const dr = (dir === 'V') ? 1 : 0, dc = (dir === 'H') ? 1 : 0;
    const r0 = rA - dr * k, c0 = cA - dc * k, L = word.length;

    if (!inBounds(r0, c0) || !inBounds(r0 + dr * (L - 1), c0 + dc * (L - 1))) return null;
    if (!contiguousEndsClear(b, r0, c0, dr, dc, L)) return null;

    const rackMap = multiset(rack);
    let usedExisting = false, usedFromRack = 0;
    const placements = [];

    for (let i = 0; i < L; i++) {
        const rr = r0 + dr * i, cc = c0 + dc * i, need = word[i];
        const old = b[rr][cc];
        if (old !== EMPTY) {
            if (old !== need) return null;
            usedExisting = true;
        } else {
            const t = takeFromRack(rackMap, need);
            if (!t) return null;
            usedFromRack++;
            placements.push({ r: rr, c: cc, ch: need, usedBlank: t.usedBlank });
        }
    }

    if (usedFromRack === 0) return null; // en az bir taş konmalı

    // İlk hamle değilse: ya bindirme, ya da dikine bir kelime üretimi şart
    if (!boardEmpty(b) && !usedExisting) {
        // En az bir yerleştirilen taşın dik komşusu dolu mu?
        let formsAnyCross = placements.some(p => hasPerpNeighbor(b, p.r, p.c, dr, dc));
        if (!formsAnyCross) return null;
    }

    // Ana kelime sözlükte mi? Şimdi string'i inşa edelim
    let mainWordChars = [];
    for (let i = 0; i < L; i++) {
        const rr = r0 + dr * i, cc = c0 + dc * i;
        const old = b[rr][cc] !== EMPTY;
        const ch = old ? b[rr][cc] : word[i];
        mainWordChars.push(ch);
    }
    const mainWord = mainWordChars.join('');
    if (!dictHas(mainWord)) return null;

    // Puanlama
    let total = scoreMainWord(b, placements, r0, c0, dr, dc, L);

    // Çapraz kelimeler sözlükte olmalı + puanı ekle
    for (const p of placements) {
        const cw = scoreCrossWord(b, p.r, p.c, dr, dc, p.ch, p.usedBlank);
        if (cw.word && cw.word.length > 1) {
            if (!dictHas(cw.word)) return null;
            total += cw.score;
        }
    }

    if (usedFromRack >= 7) total += BINGO_SCORE;

    return { word: mainWord, row: r0, col: c0, dir, score: total, placed: placements };
}

// ---------- ilk hamle (merkezi kapatmalı) ----------

function tryPlaceFirst(b, rack, word, dictHas) {
    const centerR = centerStar.row;
    const centerC = centerStar.col;
    const out = [];

    for (const dir of ['H', 'V']) {
        const dr = (dir === 'V') ? 1 : 0, dc = (dir === 'H') ? 1 : 0;

        for (let k = 0; k < word.length; k++) {
            const r0 = centerR - dr * k, c0 = centerC - dc * k, L = word.length;
            if (!inBounds(r0, c0) || !inBounds(r0 + dr * (L - 1), c0 + dc * (L - 1))) continue;
            if (!contiguousEndsClear(b, r0, c0, dr, dc, L)) continue;

            const rackMap = multiset(rack);
            const placements = [];
            let fail = false;

            for (let i = 0; i < L; i++) {
                const rr = r0 + dr * i, cc = c0 + dc * i, need = word[i];
                const t = takeFromRack(rackMap, need);
                if (!t) { fail = true; break; }
                placements.push({ r: rr, c: cc, ch: need, usedBlank: t.usedBlank });
            }
            if (fail) continue;

            const coversCenter = placements.some(p => p.r === centerR && p.c === centerC);
            if (!coversCenter) continue;

            // Ana kelime sözlükte mi?
            if (!dictHas(word)) continue;

            let total = scoreMainWord(b, placements, r0, c0, dr, dc, L);
            if (L >= 7) total += BINGO_SCORE;

            out.push({ word, row: r0, col: c0, dir, score: total, placed: placements });
        }
    }
    return out;
}

// ---------- PARALLEL: boş segment içinde alt-aralık yerleştirme ----------

function generateParallelMoves(board, rack, dictByLen, dictHas) {
    const results = [];
    const MAX_USE = Math.min(7, rack.length);

    function scanDirection(dir) {
        const dr = dir === 'H' ? 0 : 1;
        const dc = dir === 'H' ? 1 : 0;

        for (let line = 0; line < SIZE; line++) {
            const get = (i) => dir === 'H' ? board[line][i] : board[i][line];

            // boş segmentleri bul
            let s = -1;
            for (let i = 0; i <= SIZE; i++) {
                const ch = i < SIZE ? get(i) : '#'; // sınır bekçisi
                const empty = (i < SIZE) ? (ch === EMPTY) : false;

                if (empty) {
                    if (s === -1) s = i;
                } else if (s !== -1) {
                    const e = i - 1;
                    const segLen = e - s + 1;

                    // segmentte dikine komşulukla temas var mı?
                    let touches = false;
                    for (let t = s; t <= e; t++) {
                        const r = dir === 'H' ? line : t;
                        const c = dir === 'H' ? t : line;
                        if (hasPerpNeighbor(board, r, c, dr, dc)) { touches = true; break; }
                    }

                    if (touches && segLen >= 2) {
                        // ALT-ARALIK: başlangıç o, uzunluk L
                        for (let o = s; o <= e; o++) {
                            const maxL = Math.min(MAX_USE, e - o + 1);
                            for (let L = 2; L <= maxL; L++) {
                                // alt-aralık içinde de en az bir kare dikine temas etsin
                                let subTouches = false;
                                for (let t = o; t <= o + L - 1; t++) {
                                    const r = dir === 'H' ? line : t;
                                    const c = dir === 'H' ? t : line;
                                    if (hasPerpNeighbor(board, r, c, dr, dc)) { subTouches = true; break; }
                                }
                                if (!subTouches) continue;

                                const r0 = dir === 'H' ? line : o;
                                const c0 = dir === 'H' ? o : line;
                                const words = dictByLen.get(L) || [];

                                for (const w of words) {
                                    // paralelde bindirme hedefi yok -> k=0
                                    const cand = tryPlace(board, rack, w, r0, c0, dir, 0, dictHas);
                                    if (cand) results.push(cand);
                                }
                            }
                        }
                    }
                    s = -1; // segment kapandı
                }
            }
        }
    }

    scanDirection('H');
    scanDirection('V');
    return results;
}

// ---------- public api ----------

export async function loadDictionary() {
    const res = await fetch('/static/tr-dictionary.json');
    return res.json();
}

export function solveBoard(boardRows, rackString, dictionaryArray) {
    const board = boardRows.map(r => r.split('').map(ch => ch.toUpperCase()));
    const rack = (rackString || '').toUpperCase();

    const dictSet = new Set((dictionaryArray || []).map(w => w.toUpperCase()).filter(Boolean));
    const dictHas = (w) => dictSet.has(w);

    // index: harfe göre (bindirmeli aramalar için)
    const index = new Map();
    // index: uzunluğa göre (paralel aramalar için)
    const dictByLen = new Map();

    for (const w of dictSet) {
        const L = w.length;
        if (L < 2 || L > SIZE) continue;

        for (const ch of new Set([...w])) {
            const arr = index.get(ch) || [];
            arr.push(w);
            index.set(ch, arr);
        }

        const arrL = dictByLen.get(L) || [];
        arrL.push(w);
        dictByLen.set(L, arrL);
    }

    const results = [];

    if (boardEmpty(board)) {
        for (const w of dictSet) results.push(...tryPlaceFirst(board, rack, w, dictHas));
    } else {
        // 1) Bindirmeli standart aramalar (mevcut harf üzerine)
        const anchors = [];
        for (let r = 0; r < SIZE; r++)
            for (let c = 0; c < SIZE; c++)
                if (board[r][c] !== EMPTY) anchors.push({ r, c, ch: board[r][c] });

        for (const { r, c, ch } of anchors) {
            const candWords = index.get(ch) || [];
            for (const w of candWords) {
                for (let k = 0; k < w.length; k++) {
                    if (w[k] !== ch) continue;
                    const H = tryPlace(board, rack, w, r, c, 'H', k, dictHas);
                    if (H) results.push(H);
                    const V = tryPlace(board, rack, w, r, c, 'V', k, dictHas);
                    if (V) results.push(V);
                }
            }
        }

        // 2) Paralel aramalar (bindirmesiz ama dikine kelimeler oluşturan)
        results.push(...generateParallelMoves(board, rack, dictByLen, dictHas));
    }

    // dedup + sıralama
    const seen = new Set(), dedup = [];
    for (const r of results) {
        const key = `${r.word}|${r.row}|${r.col}|${r.dir}`;
        if (!seen.has(key)) {
            seen.add(key);
            dedup.push(r);
        }
    }
    dedup.sort((a, b) => (b.score - a.score) || (b.word.length - a.word.length));
    return dedup;
}

export function previewBoard(boardRows, move) {
    const b = boardRows.map(row => row.split(''));
    for (const p of (move?.placed || [])) b[p.r][p.c] = p.ch;
    return b.map(r => r.join(''));
}
