// =============================================================================
// OCR + Backend Solver + 2 Sekmeli Öneri (Puan / Uzunluğa göre)
// =============================================================================

/* DOM */
const fileEl = document.getElementById('file');
const fileBtnTx = document.getElementById('fileBtnText');
const thumb = document.getElementById('thumb');
const preview = document.getElementById('preview');
const solveBtn = document.getElementById('solveBtn');

const boardEl = document.getElementById('board');
const rackEl = document.getElementById('rack');
const meta = document.getElementById('meta');

const summary = document.getElementById('summary');

/* tabs */
const tabScore = document.getElementById('tabScore');
const tabLength = document.getElementById('tabLength');
const panelScore = document.getElementById('panelScore');
const panelLength = document.getElementById('panelLength');
const movesScore = document.getElementById('movesScore');  // <ul>
const movesLength = document.getElementById('movesLength'); // <div> groups

/* State */
let BOARD = Array.from({ length: 15 }, () => Array(15).fill('.'));
let RACK = '';
let MOVES = [];
let THREE_STAR = null;
let CURRENT_MOVE = null; // seçili öneri

// CFG & premium haritası
let CFG = null;
let PREM = null;

const THREE_STAR_BONUS = 25;
let ACTIVE_TAB = 'score'; // 'score' | 'length'

/* ==== Board’ı panel genişliğine “tam” oturt ==== */
function fitBoardToPanel() {
    const wrap = document.querySelector('.boardWrap');
    const board = document.getElementById('board');
    if (!wrap || !board) return;

    const gap = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--gap')) || 2;

    // board’un gerçek padding/border’ını ölçelim (taşmasın)
    const csb = getComputedStyle(board);
    const toPx = v => parseFloat(v) || 0;
    const gapsBetween = gap * 14;
    const paddingX = toPx(csb.paddingLeft) + toPx(csb.paddingRight);
    const borderX = toPx(csb.borderLeftWidth) + toPx(csb.borderRightWidth);
    const TOTAL_FIXED = gapsBetween + paddingX + borderX;

    const FUDGE = 1;
    const inner = Math.max(0, wrap.clientWidth);
    const available = Math.max(0, inner - TOTAL_FIXED - FUDGE);

    const cell = Math.floor(available / 15);

    // 🔽 küçük ekranlarda min’i düşürüyoruz
    let MIN_CELL = 20;
    if (inner < 380) MIN_CELL = 18;   // çoğu kompakt telefon
    if (inner < 340) MIN_CELL = 16;   // ekstra küçük ekranlar

    const MAX_CELL = 52;              // istersen 54 yap
    const clamped = Math.max(MIN_CELL, Math.min(cell, MAX_CELL));

    document.documentElement.style.setProperty('--cell', clamped + 'px');
}

/* Premium yardımcıları */
function buildPremium(cfg) {
    const m = Array.from({ length: cfg.SIZE }, () => Array(cfg.SIZE).fill(null));
    for (const p of cfg.premiumSquares) m[p.row][p.col] = p.type; // DL/TL/DW/TW
    return m;
}
function premiumLabel(type) {
    return { DW: 'K²', TW: 'K³', DL: 'H²', TL: 'H³' }[type] || '';
}

/* Sayfa açılışı: config yükle → boş tabloyu premium etiketleriyle çiz */
document.addEventListener('DOMContentLoaded', async () => {
    try {
        CFG = await (await fetch('/static/kelimelik-config.json')).json();
        PREM = buildPremium(CFG);
    } catch {
        PREM = Array.from({ length: 15 }, () => Array(15).fill(null));
    }

    drawBoard(BOARD);
    drawRack(RACK);
    meta.textContent = `Dolu hücre: 0 • Eldeki taş: 0`;
    fitBoardToPanel();

    // tab olayları
    tabScore.addEventListener('click', () => switchTab('score'));
    tabLength.addEventListener('click', () => switchTab('length'));
});
window.addEventListener('resize', fitBoardToPanel);
window.addEventListener('orientationchange', fitBoardToPanel);

/* Board çizer (premium/mini etiketlerle) */
function drawBoard(b) {
    boardEl.innerHTML = '';
    for (let r = 0; r < 15; r++) {
        for (let c = 0; c < 15; c++) {
            const d = document.createElement('div');
            d.className = 'cell';
            const ch = b[r][c];

            if (!ch || ch === '.' || ch === '?') {
                const p = PREM?.[r]?.[c];
                if (p) {
                    d.classList.add(p);
                    const mini = document.createElement('div');
                    mini.className = `mini ${p}`;
                    mini.textContent = premiumLabel(p);
                    d.appendChild(mini);
                }
                if (r === 7 && c === 7) {
                    const center = document.createElement('div');
                    center.className = 'mini STAR';
                    center.textContent = '⭐️⭐️';
                    d.appendChild(center);
                }
                if (THREE_STAR && THREE_STAR.row === r && THREE_STAR.col === c) {
                    d.classList.add('threeStar');
                    const star = document.createElement('span');
                    star.className = 'bigStar';
                    star.textContent = '★';
                    d.appendChild(star);
                } else {
                    d.classList.add('empty');
                }
            } else {
                d.classList.add('filled');
                d.textContent = ch;
            }

            d.dataset.r = r;
            d.dataset.c = c;
            boardEl.appendChild(d);
        }
    }
    fitBoardToPanel();
}

/* Rack çizer */
function drawRack(text) {
    rackEl.innerHTML = '';
    (text || '').split('').forEach((ch) => {
        const d = document.createElement('div');
        d.className = 'tile';
        d.textContent = ch === '*' ? '★' : ch;
        rackEl.appendChild(d);
    });
}

/* Dosya seçimi */
fileEl.addEventListener('change', () => {
    const f = fileEl.files?.[0];
    if (!f) return;
    const url = URL.createObjectURL(f);
    thumb.src = url; thumb.hidden = false; fileBtnTx.textContent = 'Değiştir';
    if (preview) { preview.src = url; preview.style.display = 'block'; }
});

/* Çöz */
solveBtn.addEventListener('click', async () => {
    const f = fileEl.files?.[0];
    if (!f) return alert('Lütfen ekran görüntüsü seçin.');

    solveBtn.disabled = true;
    solveBtn.textContent = 'Çözüyor...';

    try {
        // 1) OCR + board/rack
        const fd = new FormData();
        fd.append('file', f);
        const res = await fetch('/api/solve', { method: 'POST', body: fd });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'OCR hatası.');

        BOARD = data.board;
        RACK = String(data.rack || '').toUpperCase();
        THREE_STAR = data.threeStar || null;

        CURRENT_MOVE = null;
        drawBoard(BOARD);
        drawRack(RACK);
        meta.textContent = `Dolu hücre: ${countFilled(BOARD)} • Eldeki taş: ${RACK.length}`;

        // 2) Backend solver
        const wordsRes = await fetch('/api/words', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ board: BOARD, rack: RACK }),
        });
        const wordsData = await wordsRes.json();
        if (wordsData.error) throw new Error(wordsData.error);

        MOVES = wordsData.moves || [];

        // 3 yıldız bonusu
        if (THREE_STAR) {
            for (const m of MOVES) {
                if ((m.placed || []).some(p => p.r === THREE_STAR.row && p.c === THREE_STAR.col)) {
                    m.score += THREE_STAR_BONUS;
                }
            }
        }

        // Sıralı liste için puana göre sırala
        MOVES.sort((a, b) => (b.score - a.score) || (b.word.length - a.word.length));
        renderAll();
    } catch (e) {
        alert(e.message);
    } finally {
        solveBtn.disabled = false;
        solveBtn.textContent = 'Çöz';
    }
});

function countFilled(b) {
    return b.flat().filter(ch => ch !== '.' && ch !== '?').length;
}

/* === Sekmeler === */
function switchTab(which) {
    ACTIVE_TAB = which;
    tabScore.classList.toggle('active', which === 'score');
    tabLength.classList.toggle('active', which === 'length');
    panelScore.hidden = which !== 'score';
    panelLength.hidden = which !== 'length';
}

function renderAll() {
    summary.textContent = MOVES.length
        ? `${MOVES.length} aday bulundu`
        : 'Aday bulunamadı';

    renderScoreTab();
    renderLengthTab();

    // aktif sekmeye geç
    switchTab(ACTIVE_TAB);
}

/* === Puan sekmesi (sıralı tek liste) === */
function renderScoreTab() {
    movesScore.innerHTML = '';
    const show = MOVES.slice(0, 300);

    show.forEach((m) => {
        const li = document.createElement('li');
        li.dataset.key = keyForMove(m);
        li.innerHTML = `
      <span><b>${m.word}</b> <span class="meta">(${rcToHuman(m.row, m.col)} • ${m.dir})</span></span>
      <span class="badge">${m.score}</span>`;
        li.addEventListener('click', () => selectMove(m));

        if (isSameMove(CURRENT_MOVE, m)) li.classList.add('selected');
        movesScore.appendChild(li);
    });
}

/* === Uzunluğa göre sekmesi (gruplu) === */
function renderLengthTab() {
    movesLength.innerHTML = '';

    if (!MOVES.length) return;

    // gruplama: 7,6,5,4,3,2... (olanları sırayla göster)
    const groups = new Map(); // len -> array
    for (const m of MOVES) {
        const L = (m.word || '').length || 0;
        if (!groups.has(L)) groups.set(L, []);
        groups.get(L).push(m);
    }

    // Uzunlukları büyükten küçüğe
    const lengths = Array.from(groups.keys()).sort((a, b) => b - a);

    for (const L of lengths) {
        const arr = groups.get(L);
        if (!arr || !arr.length) continue;

        // Her grup kendi içinde skor sırasına göre (güzel dursun)
        arr.sort((a, b) => (b.score - a.score) || (b.word.length - a.word.length));

        const wrap = document.createElement('div');
        wrap.className = 'group';

        const h = document.createElement('h4');
        h.textContent = `${L} harfli (${arr.length})`;
        wrap.appendChild(h);

        const ul = document.createElement('ul');
        ul.className = 'items';

        arr.forEach((m) => {
            const li = document.createElement('li');
            li.dataset.key = keyForMove(m);
            li.innerHTML = `
        <span><b>${m.word}</b> <span class="meta">(${rcToHuman(m.row, m.col)} • ${m.dir})</span></span>
        <span class="badge">${m.score}</span>`;
            li.addEventListener('click', () => selectMove(m));
            if (isSameMove(CURRENT_MOVE, m)) li.classList.add('selected');
            ul.appendChild(li);
        });

        wrap.appendChild(ul);
        movesLength.appendChild(wrap);
    }
}

/* === Seçim / Önizleme === */
function selectMove(move) {
    CURRENT_MOVE = move;

    // Orijinal tahtayı premiumlarıyla tekrar çiz
    drawBoard(BOARD);

    // Önizleme uygula
    const prev = previewBoard(BOARD.map(r => r.join('')), move).map(s => s.split(''));
    drawBoard(prev);

    // Yeşil çerçeveyi uygula
    highlightMove(move, true);

    // Her iki listede de seçili satırı güncelle
    updateSelectedRows();
}

function updateSelectedRows() {
    const allLists = [
        ...movesScore.querySelectorAll('li'),
        ...movesLength.querySelectorAll('li')
    ];
    const key = keyForMove(CURRENT_MOVE);

    allLists.forEach(li => {
        li.classList.toggle('selected', li.dataset.key === key);
    });
}

/* Yardımcılar */
function rcToHuman(r, c) { return `${String.fromCharCode(65 + c)}${r + 1}`; }
function isSameMove(a, b) {
    if (!a || !b) return false;
    return a.word === b.word && a.row === b.row && a.col === b.col && a.dir === b.dir;
}
function keyForMove(m) {
    return `${m.word}|${m.row}|${m.col}|${m.dir}`;
}
function highlightMove(m, on) {
    const sel = (r, c) => boardEl.querySelector(`.cell[data-r="${r}"][data-c="${c}"]`);
    (m.placed || []).forEach(p => {
        const el = sel(p.r, p.c);
        if (!el) return;
        el.classList.toggle('hl', on);
    });
}

/* Basit preview: yerleştirilen harfleri tabloya uygula */
function previewBoard(boardRows, move) {
    const b = boardRows.map(row => row.split(''));
    for (const p of (move?.placed || [])) b[p.r][p.c] = p.ch;
    return b.map(r => r.join(''));
}
