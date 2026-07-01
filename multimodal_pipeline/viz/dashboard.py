"""Build one HTML dashboard embedding every gallery video + image.

Open the generated ``artifacts/gallery/index.html`` in a browser to browse
everything in one panel — play the MP4s inline, view the charts — instead of
opening files one by one in VLC.

Run:  python3 -m multimodal_pipeline.viz.dashboard [gallery_dir]
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

# (section title, filename substrings that belong to it) — first match wins.
_GROUPS = [
    ("IL training (curves + rollout)", ["il_train_curve", "il_rollout"]),
    ("Retarget — Wuji hand overlaid on RGB", ["robothand"]),
    ("Retarget — fit point clouds", ["retarget_"]),
    ("Wuji URDF / robot model", ["urdf_", "synth_arm"]),
    ("World verification (MANO + 3D)", ["_world", "_mano"]),
]

_CSS = """
body{background:#15151a;color:#ddd;font-family:system-ui,sans-serif;margin:24px}
h1{font-size:22px} h2{border-bottom:1px solid #3a3a44;padding-bottom:6px;margin-top:32px}
.grid{display:flex;flex-wrap:wrap;gap:16px}
.card{background:#1e1e26;padding:8px;border-radius:8px}
.card video,.card img{max-width:540px;max-height:360px;display:block;border-radius:4px}
.cap{font-size:12px;color:#9a9aa6;margin-top:6px;max-width:540px;word-break:break-all}
.count{color:#888;font-weight:normal;font-size:14px}
"""


def build(gallery_dir: str | Path = "artifacts/gallery") -> tuple[Path, int]:
    gd = Path(gallery_dir)
    files = sorted([p.name for p in gd.glob("*.mp4")] + [p.name for p in gd.glob("*.png")])
    used: set[str] = set()
    sections: list[tuple[str, list[str]]] = []
    for title, subs in _GROUPS:
        items = [f for f in files if f not in used and any(s in f for s in subs)]
        used |= set(items)
        if items:
            sections.append((title, items))
    other = [f for f in files if f not in used and f != "index.html"]
    if other:
        sections.append(("Other", other))

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>minipipline — visualization</title>",
        f"<style>{_CSS}</style></head><body>",
        "<h1>minipipline — visualization dashboard</h1>",
    ]
    for title, items in sections:
        parts.append(f"<h2>{html.escape(title)} <span class='count'>({len(items)})</span></h2>")
        parts.append("<div class='grid'>")
        for f in items:
            media = (f"<video src='{html.escape(f)}' controls loop muted preload='metadata'></video>"
                     if f.endswith(".mp4") else f"<img loading='lazy' src='{html.escape(f)}'>")
            parts.append(f"<div class='card'>{media}<div class='cap'>{html.escape(f)}</div></div>")
        parts.append("</div>")
    parts.append("</body></html>")

    out = gd / "index.html"
    out.write_text("\n".join(parts), encoding="utf-8")
    return out, sum(len(i) for _, i in sections)


def main():
    gd = sys.argv[1] if len(sys.argv) > 1 else "artifacts/gallery"
    out, n = build(gd)
    print(f"wrote {out}  ({n} items)")
    print(f"open it with:  xdg-open {out}")


if __name__ == "__main__":
    main()
