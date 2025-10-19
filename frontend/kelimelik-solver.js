// kelimelik-solver.js  —  ES module, JSON import YOK
// Bu modül, çalışmadan önce configure(config) ile kurulmalıdır.

let CONFIG = null;
let SIZE, EMPTY, BLANK, BINGO_SCORE, letterScores, PREMIUM, centerStar;

// dışarıdan konfigürasyonu ver
export function configure(config) {
    CONFIG = config;
    SIZE = CONFIG.SIZE;
    EMPTY = CONFIG.EMPTY;
    BLANK = CONFIG.BLANK;
    BINGO_SCORE = CONFIG.BINGO_SCORE;
    letterScores = CONFIG.letterScores;
    centerStar = CONFIG.centerStar;

    PREMIUM = Array.from({ length: SIZE }, () => Array(SIZE).fill(null));
    for (const p of CONFIG.premiumSquares) PREMIUM[p.row][p.col] = p.type; // 'DL'|'TL'|'DW'|'TW'
}

// --- yardımcılar ---
const inBounds = (r, c) => r >= 0 && r < SIZE && c >= 0 && c < SIZE;
const scoreLetter = (ch) => ch === BLANK ? 0 : (letterScores[ch] || 0);

const multiset = (str) => {
    const m = new Map(); for (const ch of str) m.set(ch, (m.get(ch) || 0) + 1); return m;
};
const takeFromRack = (rackMap, needCh) => {
    const have = rackMap.get(needCh) || 0;
    if (have > 0) { rackMap.set(needCh, have - 1); return { usedBlank: false, placed: needCh }; }
    const blanks = rackMap.get(BLANK) || 0;
    if (blanks > 0) { rackMap.set(BLANK, blanks - 1); return { usedBlank: true, placed: needCh }; }
    return null;
};

const boardEmpty = (b) => b.every(row => row.every(ch => ch === EMPTY));

function extend(b, r, c, dr, dc) {
    let rs = r, cs = c, re = r, ce = c;
    while (inBounds(rs - dr, cs - dc) && b[rs - dr][cs - dc] !== EMPTY) { rs -= dr; cs -= dc; }
    while (inBounds(re + dr, ce + dc) && b[re + dr][ce + dc] !== EMPTY) { re += dr; ce += dc; }
    const chars = []; for (let rr = rs, cc = cs; ; rr += dr, cc += dc) { chars.push(b[rr][cc]); if (rr === re && cc === ce) break; }
    return { rs, cs, re, ce, word: chars.join('') };
}

const contiguousEndsClear = (b, r0, c0, dr, dc, len) => {
    const br = r0 - dr, bc = c0 - dc, ar = r0 + dr * len, ac = c0 + dc * len;
    const beforeOK = !inBounds(br, bc) || b[br][bc] === EMPTY;
    const afterOK = !inBounds(ar, ac) || b[ar][ac] === EMPTY;
    return beforeOK && afterOK;
};

// --- scoring: premiums YALNIZCA yeni yerleştirilen karelerde ---
function scoreMainWord(b, placements, r0, c0, dr, dc, len) {
    let sum = 0, mul = 1;
    for (let i = 0; i < len; i++) {
        const rr = r0 + dr * i, cc = c0 + dc * i;
        const old = b[rr][cc] !== EMPTY;
        const pl = placements.find(p => p.r === rr && p.c === cc);
        const ch = old ? b[rr][cc] : pl.ch;
        const base = scoreLetter(pl && pl.usedBlank ? BLANK : ch);

        if (old) { sum += base; }
        else {
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

function scoreCrossWord(b, r, c, dr, dc, usedBlank) {
    const pr = dc, pc = dr;
    const { rs, cs, re, ce, word } = extend(b, r, c, pr, pc);
    if (word.length <= 1) return { ok: true, score: 0, word: null };

    let sum = 0, mul = 1;
    for (let rr = rs, cc = cs; ; rr += pr, cc += pc) {
        const isNew = (rr === r && cc === c);
        const ch = b[rr][cc];
        const base = scoreLetter(isNew && usedBlank ? BLANK : ch);
        if (isNew) {
            const prem = PREMIUM[rr][cc];
            if (prem === 'DL') sum += base * 2;
            else if (prem === 'TL') sum += base * 3;
            else sum += base;
            if (prem === 'DW') mul *= 2;
            if (prem === 'TW') mul *= 3;
        } else sum += base;
        if (rr === re && cc === ce) break;
    }
    return { ok: true, score: sum * mul, word };
}

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
            usedFromRack++; placements.push({ r: rr, c: cc, ch: need, usedBlank: t.usedBlank });
        }
    }
    if (usedFromRack === 0) return null;
    if (!usedExisting && !boardEmpty(b)) return null;

    let total = scoreMainWord(b, placements, r0, c0, dr, dc, L);
    for (const p of placements) {
        const cw = scoreCrossWord(b, p.r, p.c, dr, dc, p.usedBlank);
        if (cw.word && cw.word.length > 1) {
            if (!dictHas(cw.word)) return null;
            total += cw.score;
        }
    }
    if (usedFromRack >= 7) total += BINGO_SCORE;
    return { word, row: r0, col: c0, dir, score: total, placed: placements };
}

function tryPlaceFirst(b, rack, word) {
    const center = centerStar.row;
    const out = [];
    for (const dir of ['H', 'V']) {
        const dr = (dir === 'V') ? 1 : 0, dc = (dir === 'H') ? 1 : 0;
        for (let k = 0; k < word.length; k++) {
            const r0 = center - dr * k, c0 = center - dc * k, L = word.length;
            if (!inBounds(r0, c0) || !inBounds(r0 + dr * (L - 1), c0 + dc * (L - 1))) continue;
            const rackMap = multiset(rack), placements = [];
            for (let i = 0; i < L; i++) {
                const rr = r0 + dr * i, cc = c0 + dc * i, need = word[i];
                const t = takeFromRack(rackMap, need); if (!t) { placements.length = -1; break; }
                placements.push({ r: rr, c: cc, ch: need, usedBlank: t.usedBlank });
            }
            if (placements.length < 0) continue;
            const covers = placements.some(p => p.r === center && p.c === center);
            if (!covers) continue;
            let total = scoreMainWord(b, placements, r0, c0, dr, dc, L);
            if (L >= 7) total += BINGO_SCORE;
            out.push({ word, row: r0, col: c0, dir, score: total, placed: placements });
        }
    }
    return out;
}

// ---- public api ----
export async function loadDictionary() {
    const res = await fetch('/static/tr-dictionary.json');
    return res.json();
}

export function solveBoard(boardRows, rackString, dictionaryArray) {
    const board = boardRows.map(r => r.split('').map(ch => ch.toUpperCase()));
    const rack = (rackString || '').toUpperCase();

    const dictSet = new Set((dictionaryArray || []).map(w => w.toUpperCase()).filter(Boolean));
    const index = new Map();
    for (const w of dictSet) {
        if (w.length < 2 || w.length > SIZE) continue;
        for (const ch of new Set([...w])) {
            const arr = index.get(ch) || []; arr.push(w); index.set(ch, arr);
        }
    }
    const dictHas = (w) => dictSet.has(w);

    const results = [];
    if (boardEmpty(board)) {
        for (const w of dictSet) results.push(...tryPlaceFirst(board, rack, w));
    } else {
        const anchors = [];
        for (let r = 0; r < SIZE; r++) for (let c = 0; c < SIZE; c++) {
            if (board[r][c] !== EMPTY) anchors.push({ r, c, ch: board[r][c] });
        }
        for (const { r, c, ch } of anchors) {
            const cand = index.get(ch) || [];
            for (const w of cand) {
                for (let k = 0; k < w.length; k++) {
                    if (w[k] !== ch) continue;
                    const H = tryPlace(board, rack, w, r, c, 'H', k, dictHas); if (H) results.push(H);
                    const V = tryPlace(board, rack, w, r, c, 'V', k, dictHas); if (V) results.push(V);
                }
            }
        }
    }

    const seen = new Set(), dedup = [];
    for (const r of results) {
        const key = `${r.word}|${r.row}|${r.col}|${r.dir}`;
        if (!seen.has(key)) { seen.add(key); dedup.push(r); }
    }
    dedup.sort((a, b) => (b.score - a.score) || (b.word.length - a.word.length));
    return dedup;
}

export function previewBoard(boardRows, move) {
    const b = boardRows.map(row => row.split(''));
    for (const p of (move?.placed || [])) b[p.r][p.c] = p.ch;
    return b.map(r => r.join(''));
}
