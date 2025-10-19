import { configure, loadDictionary, solveBoard, previewBoard } from '/static/kelimelik-solver.js';

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

let BOARD = Array.from({ length: 15 }, () => Array(15).fill('.'));
let RACK = '';
let MOVES = [];
let DICT = [];
let CFG = null;
let PREM = null;
let CURRENT_PREVIEW = null;

/* =========== Board’ı panele TAM oturt =========== */
function fitBoardToPanel() {
    const wrap = document.querySelector('.boardWrap');
    if (!wrap) return;
    const GAP = 2;     // CSS --gap
    const BORDER = 2;  // board 1px kenarlık * 2
    const FUDGE = 1;   // taşmayı kesin önlemek için 1px marj
    const inner = wrap.clientWidth;
    const cell = Math.floor((inner - GAP * 16 - BORDER - FUDGE) / 15);
    const clamped = Math.max(20, cell);
    document.documentElement.style.setProperty('--cell', clamped + 'px');
}
const ro = new ResizeObserver(fitBoardToPanel);
ro.observe(document.querySelector('.boardWrap') || document.body);
window.addEventListener('orientationchange', fitBoardToPanel);

/* =========== Config & premium map =========== */
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
    if (type === 'DW') return 'K²';
    if (type === 'TW') return 'K³';
    if (type === 'DL') return 'H²';
    if (type === 'TL') return 'H³';
    return '';
}

/* =========== Dosya seçimi =========== */
fileEl.addEventListener('change', () => {
    const f = fileEl.files?.[0];
    if (!f) return;

    const url = URL.createObjectURL(f);
    // üst bardaki küçük kare
    thumb.src = url;
    thumb.hidden = false;
    // alttaki büyük önizleme
    preview.src = url;
    preview.style.display = 'block';

    // buton metnini "Değiştir" yap
    fileBtnTx.textContent = 'Değiştir';
});

/* =========== OCR + otomatik öner =========== */
solveBtn.addEventListener('click', async () => {
    const f = fileEl.files?.[0];
    if (!f) { alert('Lütfen ekran görüntüsü seçin.'); return; }

    solveBtn.disabled = true;
    solveBtn.textContent = 'Çözüyor...';

    try {
        const fd = new FormData(); fd.append('file', f);
        const res = await fetch('/api/solve', { method: 'POST', body: fd });
        const data = await res.json(); if (!res.ok) throw new Error(data.error || 'Hata');

        BOARD = data.board;
        // OCR bazen '?' döndürebiliyor → varsayılan olarak joker kabul edelim (hemen öneri çıkması için).
        // Kullanıcı isterse rafta taşa dokunarak tekrar '?' yapabilir (toggle).
        RACK = String(data.rack || '').toLocaleUpperCase('tr-TR').replace(/\?/g, '*');

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

/* =========== Tahta & Eldeki Taşlar çizimi =========== */
function drawBoard(b) {
    boardEl.innerHTML = '';
    for (let r = 0; r < 15; r++) for (let c = 0; c < 15; c++) {
        const d = document.createElement('div'); d.className = 'cell';
        const ch = b[r][c];

        if (!ch || ch === '.' || ch === '?') {
            const p = PREM?.[r]?.[c];
            if (p) {
                d.classList.add(p);                        // premium zemin
                const mini = document.createElement('div'); // ortadaki beyaz etiket
                mini.className = `mini ${p}`;
                mini.textContent = premiumLabel(p);        // "H²", "K³" ...
                d.appendChild(mini);
            }
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
        d.dataset.r = r; d.dataset.c = c;
        boardEl.appendChild(d);
    }
}

// >>> Joker destekli rafta çizim (+ toggle * ↔ ?)
function drawRack(text) {
    rackEl.innerHTML = '';
    const arr = (text || '').split('');

    arr.forEach((ch, idx) => {
        const d = document.createElement('div');
        d.className = 'tile';

        if (ch === '*') {
            d.classList.add('tile--joker');
            d.textContent = '★';
            d.title = 'Joker (*): dokunarak ? yap';
        } else if (ch === '?') {
            d.classList.add('tile--unknown');
            d.textContent = '?';
            d.title = 'Bilinmeyen taş: dokunarak joker (*) yap';
        } else {
            d.textContent = ch || '';
        }

        // Yalnızca * ve ? için toggle aktif
        d.addEventListener('click', () => {
            const cur = RACK.split('');
            if (cur[idx] === '*') cur[idx] = '?';
            else if (cur[idx] === '?') cur[idx] = '*';
            else return;
            RACK = cur.join('');
            drawRack(RACK);
            updateSuggestions(); // rack değişti → önerileri yenile
        });

        rackEl.appendChild(d);
    });
}

function countFilled(b) {
    let n = 0; for (const r of b) for (const ch of r) if (ch !== '.' && ch !== '?') n++; return n;
}

/* =========== Öneriler – tek liste, yüksekten düşüğe =========== */
async function updateSuggestions() {
    const rows = BOARD.map(r => r.join(''));
    // Solver’a giderken '?' → '*' çeviriyoruz ki joker gibi davransın
    const rackForSolve = RACK.replace(/\?/g, '*');

    MOVES = solveBoard(rows, rackForSolve, DICT);
    MOVES.sort((a, b) => (b.score - a.score) || (b.word.length - a.word.length));
    renderMoves();
}

function renderMoves() {
    summary.textContent = MOVES.length ? `${MOVES.length} aday bulundu (yüksek puandan düşüğe)` : 'Aday bulunamadı';
    movesEl.innerHTML = '';

    const show = MOVES.slice(0, 300);
    for (const m of show) {
        const li = document.createElement('li');
        li.innerHTML = `
      <span><b>${m.word}</b> <span class="meta">(${rcToHuman(m.row, m.col)} • ${m.dir})</span></span>
      <span class="badge">${m.score}</span>
    `;
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

function rcToHuman(r, c) { return `${String.fromCharCode(65 + c)}${r + 1}`; }
function highlightMove(m, on) {
    const sel = (r, c) => boardEl.querySelector(`.cell[data-r="${r}"][data-c="${c}"]`);
    (m.placed || []).forEach(p => {
        const el = sel(p.r, p.c); if (!el) return;
        if (on) el.classList.add('hl'); else el.classList.remove('hl');
    });
}
