# AlignFormer as a plug-in -- CVPR 2027 paper

- `main.tex`: the new paper first (currently an outline), then the old draft at the bottom for reference.
- `sec/00_glance.tex`: thesis, rules, the six claims. **Read this first.**
- `sec/0_abstract.tex` .. `sec/7_conclusion.tex`: one `\para{topic}{highlights}` per future paragraph.
- `sec/8_supplement_plan.tex`: what the supplement holds.
- `sec/9_ledger.tex`: exhibit -> run, code gaps to close, runs we will not make. Keep it in sync with the code.
- `tables/t*.tex`, `figures/f*.tex`: empty shells. Their rows and columns are what the experiments must produce.
- `preamble.tex`: packages and macros. The outline macros are in the last block.
- `old_draft/`: the Oct 22 draft (`sec`, `tables`, `figures`, `main_oct22.tex`, `main_oct22.pdf`). Reference only.

Build: `latexmk -pdf -outdir=build main.tex && cp build/main.pdf main.pdf`
