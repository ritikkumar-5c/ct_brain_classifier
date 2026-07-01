# v3 CT-Brain Test-Set Analysis — web app

Interactive webpage to browse every test study, its actual (radiologist)
classification, and the **2-stage cascade** prediction from the **v3** model,
with **live Grad-CAM++** explainability on click.

Two datasets are kept **separate** and switchable with a tab in the UI:
- **Held-out enriched split** — 1,583 studies (`runs/.../series_probs_{val,test}.csv`)
- **June 21–27 production week** — 6,682 studies (`runs/.../eval_test_21_27/`)

## Run

```bash
PY=/root/ritikkumar/ct_brain/bin/python
cd /root/ritikkumar/ct_brain_classifier

# 1. (re)build the case table + cascade thresholds  -> webapp/cases.json
$PY webapp/prepare_cases.py

# 2. serve (stdlib only; loads v3 best.pt lazily on first Grad-CAM request)
$PY webapp/serve.py --port 8080          # --device cuda|cpu|auto  --topk 6
# open http://localhost:8080
```

## What the page shows
- **Dataset tabs** (held-out / June) — chosen in the UI, never merged.
- **Operating point** selector — Stage-1 abnormal-sensitivity 0.95 / 0.98 / 0.99.
  Thresholds T1 (abn-vs-rest) and T2 (near-vs-normal) are fit on **val** and
  applied to **test** (no leakage); switching recomputes predictions instantly
  in the browser. Mirrors `eval_cascade.py`.
- **Table**: study_path / id, StudyInstanceUID, actual class (+ radiologist
  label), cascade prediction, triage bucket (escalate / light review /
  auto-clear), per-class prob bar, correctness. Filters: class, prediction,
  bucket, "mispredictions only", "false-clears only" (a not-normal auto-cleared
  as normal — the safety-critical miss, highlighted red), and free-text search.
- **Click a study** → drawer with iuid, findings, cascade logic, and buttons to
  render **Grad-CAM++** overlays for the top-attended slices, targeting any
  class (normal / near_normal / abnormal). Click an overlay to toggle raw CT.

## Data joins
- Predictions: per-study **mean** of v3 per-series softmax (`series_probs_*`).
- Held-out iuid/findings: `disk_vdc/ct_brain_orig_csv/ct_brain_jan-may_2026_final.csv`
  joined on the folder-ID hex in its `study_path` column (1,582/1,583).
- June iuid: `_download.log` (`<iuid> -> <date>/<HEX>`); findings:
  `ct_brain_test_set_june_21_27.csv` on iuid (6,680/6,682).

Grad-CAM++ layer + method: `xai/gradcampp.py` (`stages.3.blocks.1.conv.conv2_kxk`),
run through the full MIL model per slice (same as `infer.py`).
