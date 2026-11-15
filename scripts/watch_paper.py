"""Rebuild the paper PDF whenever the paper or its results change. Leave it running in a terminal.

* Any .tex / .bib / .sty / figure PDF under CVPR/ changes      -> latexmk, then CVPR/main.pdf is replaced.
* A result file in outputs/paper/ or the exhibits script changes -> scripts/make_paper_exhibits.py first
  (tables T1/T3/T6 and figures F3/F5/F6), then the PDF.

Checks every 2 s (no inotify on this machine). One line per build with the time, the page count, or the LaTeX error.

    /mnt/external/backup/dhia/conda_envs/opencood-blackwell/bin/python scripts/watch_paper.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "CVPR"
GENERATED = PAPER / "tables" / "gen"


def paper_files() -> dict:
    files = {}
    for pattern in ("**/*.tex", "**/*.bib", "**/*.sty", "figures/*.pdf"):
        for path in PAPER.glob(pattern):
            if "build" in path.parts or "old_draft" in path.parts:
                continue
            files[path] = path.stat().st_mtime
    return files


def result_files() -> dict:
    files = {ROOT / "scripts" / "make_paper_exhibits.py": 0.0}
    for pattern in ("*.json", "hosts/*.json", "hosts_v2xreal/*.json", "hosts_ego/*.json"):
        files.update({path: 0.0 for path in (ROOT / "outputs" / "paper").glob(pattern)})
    return {path: path.stat().st_mtime for path in files if path.exists()}


def stamp() -> str:
    return time.strftime("%H:%M:%S")


def make_exhibits() -> bool:
    run = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_paper_exhibits.py")], cwd=ROOT,
                         capture_output=True, text=True)
    if run.returncode != 0:
        print(f"[{stamp()}] exhibits FAILED:\n{run.stderr.strip()[-1500:]}", flush=True)
        return False
    print(f"[{stamp()}] exhibits regenerated (tables and figures)", flush=True)
    return True


def _latexmk() -> subprocess.CompletedProcess:
    return subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-outdir=build", "main.tex"],
                          cwd=PAPER, capture_output=True, text=True)


def _clean_auxiliary() -> None:
    """Remove LaTeX's auxiliary files (not the PDF): a half-written .aux breaks every later build."""
    for pattern in ("*.aux", "*.bbl", "*.blg", "*.fdb_latexmk", "*.fls", "*.out", "*.toc"):
        for path in (PAPER / "build").glob(pattern):
            path.unlink(missing_ok=True)


def build() -> None:
    import fcntl

    start = time.time()
    # one build at a time, even with several watchers: two latexmk runs on one build/ corrupt its .aux
    with open(PAPER / "build" / ".watch.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run = _latexmk()
        if run.returncode != 0:                    # self-heal once: a stale or corrupted .aux is the usual cause
            _clean_auxiliary()
            run = _latexmk()
    log = (PAPER / "build" / "main.log")
    if run.returncode != 0:
        text = log.read_text(errors="replace") if log.exists() else run.stdout
        errors = [line for line in text.splitlines() if line.startswith("!") or line.startswith("l.")][:6]
        print(f"[{stamp()}] BUILD FAILED, main.pdf not replaced:\n  " + "\n  ".join(errors or [run.stdout[-800:]]),
              flush=True)
        return
    shutil.copy2(PAPER / "build" / "main.pdf", PAPER / "main.pdf")
    pages = re.search(r"Output written on .*?\((\d+) pages", log.read_text(errors="replace"))
    print(f"[{stamp()}] built CVPR/main.pdf, {pages.group(1) if pages else '?'} pages, {time.time() - start:.0f} s",
          flush=True)


def main() -> None:
    print(f"[{stamp()}] watching CVPR/ and outputs/paper/ (Ctrl+C to stop)", flush=True)
    results = result_files()
    make_exhibits()
    paper = paper_files()
    build()
    paper = paper_files()          # the build itself may touch nothing we watch, but the exhibits did
    while True:
        time.sleep(2)
        new_results = result_files()
        if new_results != results:
            results = new_results
            make_exhibits()
        new_paper = paper_files()
        if new_paper != paper:
            time.sleep(0.5)        # let an editor finish writing
            build()
            paper = paper_files()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n[{stamp()}] stopped", flush=True)
