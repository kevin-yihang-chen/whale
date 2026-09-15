# Manuscript workspace

`main.tex` and `supplement.tex` contain the prospective protocol and explicitly
marked missing results. They do not claim measured VETO gains. From the repository
root, use the existing system Python (with matplotlib):

```
python -m ours.fast_chart_paper --output data/fast-chart-20260915-v1/paper-build-next
```

Use a fresh output directory. The build uses the downloaded official author kit
and local Tectonic cache. Both PDFs in `paper-build-v8` compiled successfully,
including the method figure and the real initial-prompt calibration table in
the supplement. The pinned author-kit README currently says2026;
this is provisional drafting layout, not a verified2027 submission package.

Planned figures: method flow, stage results, performance/cost. Planned tables:
main comparison, same-data/augmentation controls, external subset and costs.
Only traced real results may replace the current placeholders. Store generated
builds and evaluation-derived artifacts under the compact experiment output root.
