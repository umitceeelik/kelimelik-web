// frontend/main.js
// ============================================================================
// Kelimelik OCR Frontend
// - Ekran görüntüsü seçimi, backend’e gönderimi
// - OCR sonuçlarını çizim ve kelime önerileri
// - 3 yıldız (bonus) kontrolü
// ============================================================================

import { configure, loadDictionary, solveBoard, previewBoard } from '/static/kelimelik-solver.js';

/* ---------- DOM Element Referansları ---------- */
const fileEl = document.getElementById('file');
const fileBtnTx = document.getElementById('fileBtnText');
const thumb = document.getElementById('thumb');

const solveBtn = document.getElementById('solveBtn');
const boardEl = document.getElementById('board');
const rackEl = document.getElementById('rack');
const preview = document.getElementById('preview');
const meta = document.getElementById('meta');

const movesEl = document.getElementById('moves');
const summary = document.getElementById('summary');
const debugGridEl = document.getElementById('debugGrid');

/* ---------- Global State ---------- */
let BOARD = Array.from({ length: 15 }, () => Array(15).fill('.'));
let RACK = '';
let MOVES = [];
let DICT = [];
let CFG = null;
let PREM = null;
let CURRENT_PREVIEW = null;

let THREE_STAR = null;           // Backend’den {row, col} veya null gelir
const THREE_STAR_BONUS = 25;     // 3 yıldızlı kareye ek puan

/* ========================================================================== */
/* BOARD PANELİNİ EKRANA TAM OTURTMA */
/* ========================================================================== */
function fitBoardToPanel() {
    const wrap = document.querySelector('.boardWrap');
    if (!wrap) return;

    const GAP = 2, BORDER = 2, FUDGE = 1;
    const inner = wrap.clientWidth;
    const cell = Math.floor((inner - GAP * 16 - BORDER - FUDGE) / 15);
    const clamped = Math.max(20, cell);
    document.documentElement.style.setProperty('--cell', clamped + 'px');
}

const ro = new ResizeObserver(fitBoardToPanel);
ro.observe(document.querySelector('.boardWrap') || document.body);
window.addEventListener('orientationchange', fitBoardToPanel);

/* ========================================================================== */
/* CONFIG YÜKLEME & PREMIUM HARİTA OLUŞTURMA */
/* ========================================================================== */
(async () => {
    CFG = await (await fetch('/static/kelimelik-config.json')).json();
    configure(CFG);
    PREM = buildPremium(CFG);
    drawBoard(BOARD);
    drawRack(RACK);
    fitBoardToPanel();
})();

function buildPremium(cfg) {
    const m = Array.from({ length: cfg.SIZE }, () => Array(cfg.SIZE).fill(null));
    for (const p of cfg.premiumSquares) m[p.row][p.col] = p.type;
    return m;
}

function premiumLabel(type) {
    return { DW: 'K²', TW: 'K³', DL: 'H²', TL: 'H³' }[type] || '';
}

/* ========================================================================== */
/* DOSYA SEÇİMİ */
/* ========================================================================== */
fileEl.addEventListener('change', () => {
    const f = fileEl.files?.[0];
    if (!f) return;
    const url = URL.createObjectURL(f);
    thumb.src = url;
    thumb.hidden = false;
    preview.src = url;
    preview.style.display = 'block';
    fileBtnTx.textContent = 'Değiştir';
});

/* ========================================================================== */
/* OCR + KELİME ÖNERİLERİ */
/* ========================================================================== */
solveBtn.addEventListener('click', async () => {
    const f = fileEl.files?.[0];
    if (!f) return alert('Lütfen ekran görüntüsü seçin.');

    solveBtn.disabled = true;
    solveBtn.textContent = 'Çözüyor...';

    try {
        const fd = new FormData();
        fd.append('file', f);

        const res = await fetch('/api/solve', { method: 'POST', body: fd });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'OCR hatası oluştu.');

        // Backend çıktıları
        BOARD = data.board;
        RACK = String(data.rack || '').toLocaleUpperCase('tr-TR').replace(/\?/g, '*');
        THREE_STAR = data.threeStar || null;

        CURRENT_PREVIEW = null;
        drawBoard(BOARD);
        drawRack(RACK);
        meta.textContent = `Dolu hücre: ${countFilled(BOARD)} • Eldeki taş: ${RACK.length}`;

        if (!DICT.length) DICT = await loadDictionary();
        await updateSuggestions();
    } catch (e) {
        alert(e.message);
    } finally {
        solveBtn.disabled = false;
        solveBtn.textContent = 'Çöz';
    }
});

/* ========================================================================== */
/* TAHTA & RAF ÇİZİMİ */
/* ========================================================================== */
function drawBoard(b) {
    boardEl.innerHTML = '';
    for (let r = 0; r < 15; r++) {
        for (let c = 0; c < 15; c++) {
            const d = document.createElement('div');
            d.className = 'cell';
            const ch = b[r][c];

            if (!ch || ch === '.' || ch === '?') {
                // 3 yıldızlı hücre (backend’den gelen koordinat)
                if (THREE_STAR && THREE_STAR.row === r && THREE_STAR.col === c) {
                    d.classList.add('threeStar', 'empty');
                    const star = document.createElement('span');
                    star.className = 'bigStar';
                    star.textContent = '★';
                    d.appendChild(star);
                    boardEl.appendChild(d);
                    continue;
                }

                // Premium kare etiketi (ör. H², K³)
                const p = PREM?.[r]?.[c];
                if (p) {
                    d.classList.add(p);
                    const mini = document.createElement('div');
                    mini.className = `mini ${p}`;
                    mini.textContent = premiumLabel(p);
                    d.appendChild(mini);
                }

                // Merkez yıldız
                if (r === 7 && c === 7) {
                    const center = document.createElement('div');
                    center.className = 'mini STAR';
                    center.textContent = '⭐️⭐️';
                    d.appendChild(center);
                }

                d.classList.add('empty');
            } else {
                d.classList.add('filled');
                d.textContent = ch;
            }

            d.dataset.r = r;
            d.dataset.c = c;
            boardEl.appendChild(d);
        }
    }
}

/* Raf (joker/unknown toggle) */
function drawRack(text) {
    rackEl.innerHTML = '';
    const arr = (text || '').split('');

    arr.forEach((ch, idx) => {
        const d = document.createElement('div');
        d.className = 'tile';

        if (ch === '*') {
            d.classList.add('tile--joker');
            d.textContent = '★';
            d.title = 'Joker (*): ? yap';
        } else if (ch === '?') {
            d.classList.add('tile--unknown');
            d.textContent = '?';
            d.title = 'Bilinmeyen: * yap';
        } else {
            d.textContent = ch || '';
        }

        d.addEventListener('click', () => {
            const cur = RACK.split('');
            if (cur[idx] === '*') cur[idx] = '?';
            else if (cur[idx] === '?') cur[idx] = '*';
            else return;
            RACK = cur.join('');
            drawRack(RACK);
            updateSuggestions();
        });

        rackEl.appendChild(d);
    });
}

function countFilled(b) {
    return b.flat().filter(ch => ch !== '.' && ch !== '?').length;
}

/* ========================================================================== */
/* KELİME ÖNERİLERİ & VURGULAMA */
/* ========================================================================== */
async function updateSuggestions() {
    const rows = BOARD.map(r => r.join(''));
    const rackForSolve = RACK.replace(/\?/g, '*');

    MOVES = solveBoard(rows, rackForSolve, DICT);

    // 3 yıldız bonusu ekle
    if (THREE_STAR) {
        for (const m of MOVES) {
            if ((m.placed || []).some(p => p.r === THREE_STAR.row && p.c === THREE_STAR.col)) {
                m.score += THREE_STAR_BONUS;
            }
        }
    }

    MOVES.sort((a, b) => (b.score - a.score) || (b.word.length - a.word.length));
    renderMoves();
}

function renderMoves() {
    summary.textContent = MOVES.length
        ? `${MOVES.length} aday bulundu (yüksek puandan düşüğe)`
        : 'Aday bulunamadı';

    movesEl.innerHTML = '';
    const show = MOVES.slice(0, 300);

    for (const m of show) {
        const li = document.createElement('li');
        li.innerHTML = `
          <span><b>${m.word}</b> <span class="meta">(${rcToHuman(m.row, m.col)} • ${m.dir})</span></span>
          <span class="badge">${m.score}</span>`;

        li.addEventListener('mouseenter', () => highlightMove(m, true));
        li.addEventListener('mouseleave', () => { if (CURRENT_PREVIEW !== m) highlightMove(m, false); });
        li.addEventListener('click', () => {
            CURRENT_PREVIEW = m;
            drawBoard(BOARD);
            const prev = previewBoard(BOARD.map(r => r.join('')), m).map(s => s.split(''));
            drawBoard(prev);
            highlightMove(m, true);
        });

        movesEl.appendChild(li);
    }
}

/* Yardımcılar */
function rcToHuman(r, c) { return `${String.fromCharCode(65 + c)}${r + 1}`; }
function highlightMove(m, on) {
    const sel = (r, c) => boardEl.querySelector(`.cell[data-r="${r}"][data-c="${c}"]`);
    (m.placed || []).forEach(p => {
        const el = sel(p.r, p.c);
        if (!el) return;
        el.classList.toggle('hl', on);
    });
}
