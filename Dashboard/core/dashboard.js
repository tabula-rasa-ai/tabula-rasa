// Tabula Rasa Dashboard — Shared JS (V1.3)
// Include via: <script src="../core/dashboard.js"></script>

var API = window.location.protocol + '//' + window.location.hostname + ':' + window.location.port;
var API2 = API;

function $(id) { return document.getElementById(id); }

function esc(s) { var d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

function fmtNum(n) { if (n === null || n === undefined) return '-'; return n.toLocaleString(); }

function fmtTime(s) {
    if (!s || s < 0) return '-';
    var h = Math.floor(s / 3600);
    var m = Math.floor((s % 3600) / 60);
    var sec = Math.floor(s % 60);
    return (h > 0 ? h + 'h ' : '') + (m > 0 ? m + 'm ' : '') + sec + 's';
}

function fmtElapsed(ts) {
    var sec = Math.floor((Date.now() - ts) / 1000);
    if (sec < 60) return sec + 's ago';
    if (sec < 3600) return Math.floor(sec / 60) + 'm ago';
    return Math.floor(sec / 3600) + 'h ago';
}

function statusBadge(status) {
    var map = {
        'ready': 'badge badge-green',
        'trained': 'badge badge-green',
        'training': 'badge badge-yellow',
        'queued': 'badge badge-dim',
        'error': 'badge badge-red'
    };
    return '<span class="' + (map[status] || 'badge badge-dim') + '">' + esc(status) + '</span>';
}

// Simple polling helper
function poll(url, cb, interval) {
    async function run() {
        try {
            var r = await fetch(url);
            var d = await r.json();
            cb(null, d);
        } catch(e) { cb(e, null); }
    }
    run();
    return setInterval(run, interval);
}
