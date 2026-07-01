"""
serve.py
Local web app for the v3 CT-brain test-set analysis.

Serves an instant, filterable table of every test study (study_path / iuid,
radiologist classification if available, and the 2-stage cascade prediction from
the v3 model) for BOTH datasets (held-out enriched split + June 21-27 production
week). Clicking a study runs Grad-CAM++ LIVE on the GPU and returns explainable-AI
overlays for the top-attended slices.

Stdlib only (no Flask/FastAPI). Torch is imported lazily on the first Grad-CAM
request so the table loads even while the model is cold.

Run with the ct_brain venv:
  /root/ritikkumar/ct_brain/bin/python webapp/serve.py --port 8080
  # then open http://localhost:8080

Prereq: webapp/cases.json (build with webapp/prepare_cases.py).
"""
import os, io, sys, json, base64, argparse, threading, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

def soft_overlay(img_uint8, cam, max_alpha=0.6):
    """Jet heatmap blended over the CT with per-pixel alpha = CAM intensity.
    Unlike a flat-alpha blend (which tints the whole brain jet-BLUE where CAM~0,
    making every map look 'blue'), cool regions here stay as the raw CT and only
    salient regions get colour -> the hot spot pops and differs by target class."""
    import numpy as np
    import matplotlib.cm as cm
    heat = (cm.jet(cam)[..., :3] * 255)                      # HxWx3
    a = (cam[..., None] * max_alpha)                         # HxWx1 in [0,max_alpha]
    return (a * heat + (1 - a) * img_uint8).astype(np.uint8)


HERE = os.path.dirname(os.path.abspath(__file__))
# Code repo (for importing config/models/infer/xai and the default checkpoint).
# Explicit so this app can live anywhere (e.g. copied to disk_vdc); override with
# --repo or CT_BRAIN_REPO. Falls back to the parent dir if webapp/ is still inside the repo.
REPO = os.environ.get("CT_BRAIN_REPO") or (
    "/root/ritikkumar/ct_brain_classifier"
    if os.path.isfile("/root/ritikkumar/ct_brain_classifier/config.py")
    else os.path.dirname(HERE))
sys.path.insert(0, REPO)

# ---------------------------------------------------------------- model (lazy)
class Explainer:
    """Lazily loads the v3 checkpoint and produces Grad-CAM++ overlays.
    Inference is serialized with a lock (single GPU, shared with training)."""
    def __init__(self, run_dir, ckpt, device, topk):
        self.run_dir, self.ckpt, self.device_arg, self.topk = run_dir, ckpt, device, topk
        self.model = self.cfg = None
        self.lock = threading.Lock()

    def _ensure(self):
        if self.model is not None:
            return
        import torch
        from config import get_config, Config
        from models.maxvit_mil import build_model
        dev = self.device_arg
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = dev
        path = os.path.join(self.run_dir, self.ckpt)
        print(f"[model] loading {path} onto {dev} ...", flush=True)
        ck = torch.load(path, map_location=dev, weights_only=False)
        cfg = get_config(**{k: v for k, v in ck.get("cfg", {}).items() if hasattr(Config, k)})
        model = build_model(cfg)
        model.load_state_dict(ck["model"])
        model.to(dev).eval()
        self.model, self.cfg = model, cfg
        print(f"[model] ready (epoch {ck.get('epoch')}, image_size {cfg.image_size})", flush=True)

    @staticmethod
    def _series_dirs(study_path):
        """Immediate sub-folders holding .dcm = one series each. Matches the
        evaluation pipeline, which scores each series separately (the study
        probability is then the mean over series). Falls back to the study dir."""
        import glob
        subs = [d for d in sorted(glob.glob(os.path.join(study_path, "*")))
                if os.path.isdir(d) and glob.glob(os.path.join(d, "**", "*.dcm"), recursive=True)]
        return subs if subs else [study_path]

    @staticmethod
    def _eval_bag(series_dir, cfg):
        """Build a series bag EXACTLY like StudyMILDataset in eval mode: sort by
        InstanceNumber, evenly-spaced subsample to max_slices_per_study, multiwindow
        + eval transforms. This is what makes the live probs match series_probs_*.csv
        (infer.load_study instead takes the FIRST 96 slices -> different bag)."""
        import glob, numpy as np, torch, pydicom
        from data.dicom_dataset import _instance_number
        from data.transforms import build_transforms, dicom_to_multiwindow
        paths = sorted(glob.glob(os.path.join(series_dir, "**", "*.dcm"), recursive=True))
        if not paths:
            paths = [p for p in glob.glob(os.path.join(series_dir, "**", "*"), recursive=True)
                     if os.path.isfile(p)]
        paths = sorted(paths, key=_instance_number)
        n = len(paths)
        if n == 0:
            return None, []
        k = min(cfg.max_slices_per_study, n)
        chosen = np.linspace(0, n - 1, k).round().astype(int).tolist()
        tf = build_transforms(cfg, train=False)
        tiles, kept = [], []
        for j in chosen:
            try:
                ds = pydicom.dcmread(paths[j])
                tiles.append(tf(dicom_to_multiwindow(ds, cfg.windows)))
                kept.append(paths[j])
            except Exception:
                pass
        if not tiles:
            return None, []
        return torch.stack(tiles), kept

    def explain(self, study_path, target):
        import torch
        import numpy as np
        from PIL import Image
        from xai.gradcampp import GradCAMpp, denormalize
        with self.lock:
            self._ensure()
            cfg = self.cfg
            if target is None:
                target = cfg.num_classes - 1
            target = max(0, min(int(target), cfg.num_classes - 1))

            def png_b64(arr):
                buf = io.BytesIO()
                Image.fromarray(arr).save(buf, format="PNG")
                return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

            # Score EACH series separately (as in eval); study prob = mean over series.
            series = []
            for sd in self._series_dirs(study_path):
                bag, paths = self._eval_bag(sd, cfg)
                if bag is None:
                    continue
                bag = bag.to(self.device)
                mask = torch.ones(1, bag.size(0), dtype=torch.bool, device=self.device)
                with torch.no_grad():
                    logits, attn = self.model(bag.unsqueeze(0), mask, return_attn=True)
                    prob = torch.softmax(logits, 1)[0].cpu().numpy()
                    attn = attn[0].cpu().numpy()
                series.append({"name": os.path.basename(sd.rstrip("/")), "bag": bag,
                               "paths": paths, "prob": prob, "attn": attn})
            if not series:
                raise RuntimeError("no readable series/slices for this study")

            mean_prob = np.mean([s["prob"] for s in series], axis=0)
            # Grad-CAM++ on the PRIMARY (largest) series' top-attended slices.
            primary = max(series, key=lambda s: s["bag"].size(0))
            bag, attn, paths = primary["bag"], primary["attn"], primary["paths"]

            cam = GradCAMpp(self.model, cfg.gradcam_layer)
            slices = []
            try:
                top_idx = np.argsort(attn)[::-1][: self.topk]
                for rank, i in enumerate(top_idx):
                    i = int(i)
                    slice_t = bag[i].clone().requires_grad_(True)
                    heat = cam(slice_t, target_class=target)      # HxW in [0,1]
                    img = denormalize(bag[i], cfg.norm_mean, cfg.norm_std)
                    blended = soft_overlay(img, heat)
                    slices.append({
                        "rank": rank, "idx": i, "attn": round(float(attn[i]), 4),
                        "cam_peak": round(float(heat.max()), 3),   # 0 => flat (all-blue) map
                        "file": os.path.basename(paths[i]) if i < len(paths) else "",
                        "raw": png_b64(img), "overlay": png_b64(blended),
                    })
            finally:
                cam.remove()
            return {
                "study_path": study_path,
                "n_slices": int(sum(s["bag"].size(0) for s in series)),
                "n_series": len(series),
                "primary_series": primary["name"],
                "target": target,
                "target_name": cfg.class_names[target],
                "probs": {cfg.class_names[j]: round(float(mean_prob[j]), 4) for j in range(cfg.num_classes)},
                "series": [{"name": s["name"], "n_slices": int(s["bag"].size(0)),
                            "probs": {cfg.class_names[j]: round(float(s["prob"][j]), 4) for j in range(cfg.num_classes)}}
                           for s in series],
                "slices": slices,
            }


# ---------------------------------------------------------------- http
class App(BaseHTTPRequestHandler):
    payload = None          # cases.json (raw bytes)
    index_html = None       # bytes
    pid_index = None        # {dataset: {pid: study_path}}
    explainer = None

    def log_message(self, *a):  # quieter
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            return self._send(200, self.index_html, "text/html; charset=utf-8")
        if u.path in ("/cases.json", "/api/cases"):
            return self._send(200, self.payload, "application/json")
        if u.path == "/gradcam":
            q = parse_qs(u.query)
            ds = (q.get("dataset") or [""])[0]
            pid = (q.get("pid") or [""])[0]
            target = q.get("target", [None])[0]
            target = int(target) if target not in (None, "") else None
            study_path = self.pid_index.get(ds, {}).get(pid)
            if not study_path:
                return self._send(404, json.dumps({"error": f"unknown study {ds}/{pid}"}))
            if not os.path.isdir(study_path):
                return self._send(404, json.dumps({"error": f"study folder missing on disk: {study_path}"}))
            try:
                res = self.explainer.explain(study_path, target)
                return self._send(200, json.dumps(res))
            except Exception as e:
                traceback.print_exc()
                return self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}))
        return self._send(404, json.dumps({"error": "not found"}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--cases", default=os.path.join(HERE, "cases.json"))
    ap.add_argument("--repo", default=REPO, help="ct_brain_classifier code repo (for imports + ckpt)")
    ap.add_argument("--run", default=os.path.join(REPO, "runs/maxvit384_3class_clinical_v3"))
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--topk", type=int, default=6, help="top-attended slices to explain")
    args = ap.parse_args()
    if args.repo not in sys.path:
        sys.path.insert(0, args.repo)

    with open(args.cases, "rb") as f:
        App.payload = f.read()
    data = json.loads(App.payload)
    App.pid_index = {
        ds: {c["pid"]: c["study_path"] for c in d["cases"]}
        for ds, d in data["datasets"].items()
    }
    with open(os.path.join(HERE, "index.html"), "rb") as f:
        App.index_html = f.read()
    App.explainer = Explainer(args.run, args.ckpt, args.device, args.topk)

    srv = ThreadingHTTPServer((args.host, args.port), App)
    print(f"serving on http://localhost:{args.port}  (device={args.device}, topk={args.topk})", flush=True)
    print(f"datasets: " + ", ".join(f"{k}={len(v['cases'])}" for k, v in data["datasets"].items()), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
