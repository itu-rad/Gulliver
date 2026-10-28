#!/usr/bin/env python3
"""Render a pipeline config's stage graph to Graphviz DOT (and an image).

Edges to a lower stage id are back-edges (cycles) and are drawn as retries.

    python scripts/render_pipeline_graph.py CONFIG.yml [-o OUT.dot] [--format pdf]
    python scripts/render_pipeline_graph.py CONFIG.yml --format png -o graph.png
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import yaml

# Node styles by component substring; first match wins.
ROLE_STYLES = [
    ("llm_", dict(shape="box", style="filled,bold", fillcolor="#dde8f5")),
    ("Router", dict(shape="diamond", style="filled", fillcolor="#fbe6c2")),
    ("DataLoader", dict(shape="oval", style="filled", fillcolor="#e6e6e6")),
    ("Loader", dict(shape="oval", style="filled", fillcolor="#e6e6e6")),
    ("TerminalCapture", dict(shape="oval", style="filled", fillcolor="#e6e6e6")),
    ("Formatter", dict(shape="box", style="rounded,filled", fillcolor="#ffffff")),
]
DEFAULT_STYLE = dict(shape="box", style="rounded,filled", fillcolor="#ffffff")


def style_for(component: str) -> dict:
    for key, style in ROLE_STYLES:
        if key.lower() in component.lower():
            return style
    return DEFAULT_STYLE


def esc(s: str) -> str:
    return str(s).replace('"', '\\"')


def render(cfg_path: Path) -> tuple[str, dict]:
    doc = yaml.safe_load(cfg_path.read_text())
    lines = ["digraph pipeline {", "  rankdir=LR;", "  node [fontname=Helvetica, fontsize=10];",
             "  edge [fontname=Helvetica, fontsize=9];", "  compound=true;"]
    stats = {"pipelines": 0, "stages": 0, "back_edges": 0}

    for pi, pipe in enumerate(doc.get("pipelines", [])):
        stats["pipelines"] += 1
        lines.append(f"  subgraph cluster_{pi} {{")
        lines.append(f'    label="{esc(pipe.get("name", f"pipeline {pi}"))}";')
        lines.append("    style=dashed; color=gray50; fontname=Helvetica; fontsize=11;")

        stages = pipe.get("stages") or []
        by_id = {s.get("id"): s for s in stages}
        for s in stages:
            stats["stages"] += 1
            st = style_for(str(s.get("component", "")))
            attrs = ", ".join(f'{k}="{v}"' for k, v in st.items())
            lines.append(f'    n{pi}_{s.get("id")} [label="{esc(s.get("name", s.get("id")))}", {attrs}];')

        for s in stages:
            src = s.get("id")
            # A stage may name the same target twice (distinct routing branches);
            # collapse those into one edge carrying the multiplicity.
            for dst, n in Counter(s.get("outputs") or []).items():
                if dst not in by_id:
                    continue
                back = dst < src
                if back:
                    stats["back_edges"] += 1
                attrs = ['color="#b03030"', "style=dashed", "constraint=false",
                         'label="retry"'] if back else []
                if n > 1:
                    attrs.append(f'taillabel="x{n}"')
                a = (" [" + ", ".join(attrs) + "]") if attrs else ""
                lines.append(f"    n{pi}_{src} -> n{pi}_{dst}{a};")
        lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n", stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output path; extension picks the format unless --format is given")
    ap.add_argument("--format", default=None,
                    help="image format to render with Graphviz (pdf, png, svg). "
                         "Omit to write DOT only.")
    args = ap.parse_args()

    if not args.config.exists():
        print(f"render_pipeline_graph: no such config: {args.config}", file=sys.stderr)
        return 1

    dot, stats = render(args.config)
    fmt = args.format or (args.out.suffix.lstrip(".") if args.out and args.out.suffix != ".dot" else None)
    out = args.out or args.config.with_suffix(".dot")
    dot_path = out.with_suffix(".dot")
    dot_path.write_text(dot)
    print(f"{dot_path}  ({stats['pipelines']} pipeline(s), {stats['stages']} stages, "
          f"{stats['back_edges']} back-edge(s))")

    if fmt:
        if not shutil.which("dot"):
            print("render_pipeline_graph: Graphviz `dot` not on PATH -- DOT written, "
                  "image skipped", file=sys.stderr)
            return 0
        img = out if out.suffix.lstrip(".") == fmt else out.with_suffix("." + fmt)
        subprocess.run(["dot", f"-T{fmt}", str(dot_path), "-o", str(img)], check=True)
        print(f"{img}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
