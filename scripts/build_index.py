"""Build the boards landing page: data/figures/index.html.

A static launcher that groups the generated boards by track, served at ``/`` when
you run an HTTP server from ``data/figures/``. It only links boards that actually
exist; it generates none of them (rebuild a board with its ``build_*`` script).

    uv run python scripts/build_index.py
    # then, to serve the group at http://localhost:8000/ :
    (cd data/figures && uv run python -m http.server 8000)

Modelled on the sibling space-world-models boards index, restyled to match the
AUTOPS boards (TUM blue, Computer Modern).
"""
from __future__ import annotations

import html
from pathlib import Path

FIG = Path("data/figures")
OUT = FIG / "index.html"

# Board catalog, grouped by track in declaration order.
BOARDS = [
    {
        "track": "EventSat — single-satellite benchmark",
        "file": "results_board.html",
        "title": "O-framework results board",
        "desc": "The 32-cell EventSat·SAS matrix (representation × operational "
                "paradigm) over the 14 metrics, with a metric-selector heatmap. "
                "Measured cells versus placeholders.",
    },
    {
        "track": "EventSat — single-satellite benchmark",
        "file": "results_inspector.html",
        "title": "Episode inspector",
        "desc": "Per-episode telemetry traces (modes, battery, data pools, "
                "decision rationales) for the measured EventSat runs.",
    },
    {
        "track": "Flamingo — multi-satellite organisation axis",
        "file": "flamingo_board.html",
        "title": "Organisation board",
        "desc": "The five literature organisations (SAS / CMAS / IMAS / DMAS / "
                "HMAS) under a contended SSA task: mission metrics, coordination "
                "cost, and the M-10 scale-efficiency curves across N = 1…12.",
    },
]


def _card(board: dict, exists: bool) -> str:
    title = html.escape(board["title"])
    file = html.escape(board["file"])
    desc = html.escape(board["desc"])
    if exists:
        return (
            f'<a class="card" href="{file}">'
            f'<b>{title}</b> &middot; <span class="file">{file}</span>'
            f'<span class="desc">{desc}</span></a>'
        )
    return (
        f'<div class="card missing">'
        f'<b>{title}</b> &middot; <span class="file">{file}</span>'
        f'<span class="desc">{desc}</span>'
        f'<span class="tag">not built yet</span></div>'
    )


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    tracks: list[str] = []
    for board in BOARDS:
        if board["track"] not in tracks:
            tracks.append(board["track"])

    sections = []
    linked = 0
    for track in tracks:
        cards = []
        for board in BOARDS:
            if board["track"] != track:
                continue
            exists = (FIG / board["file"]).exists()
            linked += exists
            cards.append(_card(board, exists))
        sections.append(
            f'<div class="track"><h2>{html.escape(track)}</h2>'
            f'<div class="cards">{"".join(cards)}</div></div>'
        )

    OUT.write_text(TEMPLATE.replace("__SECTIONS__", "\n".join(sections)), encoding="utf-8")
    total = len(BOARDS)
    print(f"wrote {OUT}: {linked}/{total} boards linked")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AUTOPS — Boards</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/aaaakshat/cm-web-fonts@latest/fonts.css">
<style>
 :root { --blue:#0065BD; --ink:#172026; --muted:#5d6872; --line:#d8e0e6; }
 * { box-sizing:border-box; }
 body { margin:0; background:#eef2f5; color:var(--ink);
   font-family:'Computer Modern Sans',Arial,sans-serif; }
 header { padding:30px 40px 20px; background:#fff; border-bottom:2px solid var(--blue); }
 h1 { margin:0 0 6px; font-size:24px; color:var(--blue); font-weight:600;
   font-family:'Computer Modern Serif',Georgia,serif; }
 .sub { color:var(--muted); font-size:13.5px; max-width:880px; line-height:1.5; }
 main { padding:26px 40px 50px; max-width:1140px; }
 .track { margin-bottom:30px; }
 .track h2 { font-size:12.5px; text-transform:uppercase; letter-spacing:.07em;
   color:var(--muted); margin:0 0 12px; font-weight:700; }
 .cards { display:grid; grid-template-columns:repeat(auto-fill,minmax(330px,1fr)); gap:14px; }
 .card { display:block; text-decoration:none; color:inherit; background:#fff;
   border:1px solid var(--line); border-radius:12px; padding:16px 18px;
   transition:border-color .15s, transform .15s; }
 a.card:hover { border-color:var(--blue); transform:translateY(-2px); }
 .card b { font-size:15px; }
 .card .file { color:var(--blue); font-size:12px; font-family:ui-monospace,monospace; }
 .card span.desc { display:block; color:var(--muted); font-size:12.5px; margin-top:6px; line-height:1.45; }
 .card.missing { opacity:.55; }
 .card .tag { display:inline-block; margin-top:8px; font-size:11px; color:#9a6200;
   border:1px solid #9a6200; border-radius:3px; padding:1px 7px; }
 footer { color:var(--muted); font-size:12px; padding:0 40px 30px; }
 code { font-family:ui-monospace,monospace; font-size:12px; }
</style></head><body>
<header>
 <h1>AUTOPS — Boards</h1>
 <div class="sub">Local instruments for the cognitive-architecture benchmark. Grouped by track and served from
 <code>data/figures/</code>. Rebuild the numbers with <code>scripts/refresh_board.py</code>.</div>
</header>
<main>
__SECTIONS__
</main>
<footer>Static launcher — links existing boards, generates none. Serve the group with
<code>(cd data/figures &amp;&amp; python -m http.server 8000)</code> → <code>http://localhost:8000/</code>.</footer>
</body></html>
"""


if __name__ == "__main__":
    main()
