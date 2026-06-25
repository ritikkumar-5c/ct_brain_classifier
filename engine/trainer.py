"""
engine/trainer.py
Training engine with comprehensive TensorBoard logging.

Logs per epoch:
  scalars : train/val loss, accuracy, precision, recall, f1, auc, lr
  images  : confusion matrix, ROC curve, Grad-CAM++ overlays on top-attended slices
  figures : per-study MIL attention weights
  hist    : weight & gradient histograms (optional)
  hparams : final best metrics vs config (for the TB HPARAMS tab)
Also: best-by-val-F1 checkpointing and early stopping.
"""
import os
import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .losses import build_loss
from .metrics import (compute_metrics, confusion_figure, roc_figure,
                      pathology_operating_point, apply_operating_point)
from xai.gradcampp import GradCAMpp, denormalize, overlay


def build_optimizer(cfg, model):
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer.lower() in ("adam", "madam"):
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer.lower() == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    raise ValueError(cfg.optimizer)


def cosine_warmup(optimizer, cfg, steps_per_epoch):
    total = cfg.epochs * steps_per_epoch
    warm = cfg.warmup_epochs * steps_per_epoch

    def fn(step):
        if step < warm:
            return (step + 1) / max(warm, 1)
        prog = (step - warm) / max(total - warm, 1)
        return 0.5 * (1 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def _attn_figure(attn, study_id):
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.bar(range(len(attn)), attn)
    ax.set_xlabel("slice index"); ax.set_ylabel("attention")
    ax.set_title(f"MIL attention — study {study_id}")
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = buf.reshape(h, w, 4)[..., :3].copy()
    plt.close(fig)
    return img


class Trainer:
    def __init__(self, cfg, model, train_loader, val_loader, class_weight, device):
        self.cfg = cfg
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = build_loss(cfg, class_weight.to(device) if class_weight is not None else None)
        self.optimizer = build_optimizer(cfg, model)
        # gradient accumulation -> the scheduler counts OPTIMIZER updates, not batches
        self.accum = max(1, int(getattr(cfg, "grad_accum_steps", 1)))
        steps_per_epoch = max(1, math.ceil(len(train_loader) / self.accum))
        self.scheduler = cosine_warmup(self.optimizer, cfg, steps_per_epoch)
        # mixed precision (CUDA only; auto-disabled on CPU)
        self.use_amp = bool(getattr(cfg, "use_amp", False)) and str(device) == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        os.makedirs(cfg.out_dir, exist_ok=True)
        self.writer = SummaryWriter(cfg.out_dir)
        self.best_f1 = -1.0
        self.epochs_no_improve = 0
        self.global_step = 0
        self.start_epoch = 0
        # Optionally resume full training state (model+optim+sched+scaler+counters).
        resume = getattr(cfg, "resume", "")
        if resume:
            self._load_ckpt(resume)

    # ---------- checkpoint I/O (full training state) ----------
    def _save_ckpt(self, path, epoch, val_metrics):
        """Save COMPLETE state so training can resume bit-for-bit later."""
        import random
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": epoch,
            "best_score": self.best_f1,
            "epochs_no_improve": self.epochs_no_improve,
            "global_step": self.global_step,
            "monitor": getattr(self.cfg, "monitor", "f1"),
            "cfg": self.cfg.to_dict(),
            "val_metrics": val_metrics,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        tmp = path + ".tmp"
        torch.save(ckpt, tmp)
        os.replace(tmp, path)

    def _load_ckpt(self, path):
        """Restore full training state from a checkpoint and continue after its epoch."""
        if not os.path.exists(path):
            print(f"[resume] checkpoint not found: {path} — starting fresh")
            return
        import random
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model"])
        if "optimizer" in ck:
            self.optimizer.load_state_dict(ck["optimizer"])
        if "scheduler" in ck:
            self.scheduler.load_state_dict(ck["scheduler"])
        if ck.get("scaler") is not None:
            self.scaler.load_state_dict(ck["scaler"])
        self.best_f1 = ck.get("best_score", ck.get("val_metrics", {}).get("f1", -1.0))
        self.epochs_no_improve = ck.get("epochs_no_improve", 0)
        self.global_step = ck.get("global_step", 0)
        self.start_epoch = int(ck.get("epoch", -1)) + 1
        rng = ck.get("rng")
        if rng:
            try:
                random.setstate(rng["python"]); np.random.set_state(rng["numpy"])
                torch.set_rng_state(rng["torch"])
                if rng.get("cuda") is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(rng["cuda"])
            except Exception as e:
                print(f"[resume] RNG restore skipped: {e}")
        print(f"[resume] loaded {path}: continuing from epoch {self.start_epoch} "
              f"(best {ck.get('monitor','?')}={self.best_f1:.4f})")

    # ---------- one epoch ----------
    def _run_epoch(self, loader, train: bool, epoch: int):
        self.model.train(train)
        phase = "train" if train else "val"
        losses, all_true, all_prob = [], [], []
        pbar = tqdm(loader, desc=f"[{phase}] epoch {epoch}", leave=False)
        n_batches = len(loader)
        amp_dev = "cuda" if self.use_amp else "cpu"
        if train:
            self.optimizer.zero_grad(set_to_none=True)
        for it, batch in enumerate(pbar):
            bag = batch["bag"].to(self.device)
            mask = batch["mask"].to(self.device)
            y = batch["label"].to(self.device)
            with torch.set_grad_enabled(train), \
                    torch.autocast(device_type=amp_dev, dtype=torch.float16, enabled=self.use_amp):
                logits = self.model(bag, mask)
                loss = self.criterion(logits, y)
            if train:
                # scale by 1/accum so accumulated grads average over the effective batch
                self.scaler.scale(loss / self.accum).backward()
                step_now = ((it + 1) % self.accum == 0) or (it + 1 == n_batches)
                if step_now:
                    if self.cfg.grad_clip:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    scale_before = self.scaler.get_scale()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    # only advance LR if the scaler actually stepped (it skips on inf/nan grads)
                    if self.scaler.get_scale() >= scale_before:
                        self.scheduler.step()
                    self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], self.global_step)
                    self.global_step += 1
            losses.append(loss.item())
            all_true.extend(y.cpu().tolist())
            all_prob.extend(torch.softmax(logits.float(), 1).detach().cpu().tolist())
            pbar.set_postfix(loss=f"{np.mean(losses):.4f}")

        metrics = compute_metrics(all_true, all_prob, self.cfg.num_classes,
                                  class_names=self.cfg.class_names)
        metrics["loss"] = float(np.mean(losses))
        for k, v in metrics.items():
            self.writer.add_scalar(f"{phase}/{k}", v, epoch)
        return metrics, np.array(all_true), np.array(all_prob)

    # ---------- XAI logging ----------
    @torch.no_grad()
    def _collect_attn(self, n_studies):
        self.model.eval()
        out = []
        for batch in self.val_loader:
            bag = batch["bag"].to(self.device); mask = batch["mask"].to(self.device)
            logits, attn = self.model(bag, mask, return_attn=True)
            for b in range(bag.size(0)):
                k = int(mask[b].sum().item())
                out.append((batch["study_id"][b], bag[b, :k].cpu(),
                            attn[b, :k].cpu().numpy(),
                            int(logits[b].argmax().item())))
                if len(out) >= n_studies:
                    return out
        return out

    def _log_xai(self, epoch):
        cam = GradCAMpp(self.model, self.cfg.gradcam_layer)
        try:
            for sid, bag, attn, pred in self._collect_attn(self.cfg.xai_num_studies):
                top = int(np.argmax(attn))                      # most-attended slice
                slice_t = bag[top].to(self.device).requires_grad_(True)
                heat = cam(slice_t, target_class=self.cfg.num_classes - 1)  # abnormal map
                img = denormalize(bag[top], self.cfg.norm_mean, self.cfg.norm_std)
                blended = overlay(img, heat)
                self.writer.add_image(f"xai/{sid}_slice{top}_pred{pred}",
                                      np.transpose(blended, (2, 0, 1)), epoch)
                self.writer.add_image(f"xai/{sid}_attention",
                                      np.transpose(_attn_figure(attn, sid), (2, 0, 1)), epoch)
        finally:
            cam.remove()

    def _log_hist(self, epoch):
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.writer.add_histogram(f"weights/{name}", p.detach().cpu(), epoch)
                if p.grad is not None:
                    self.writer.add_histogram(f"grads/{name}", p.grad.detach().cpu(), epoch)

    # ---------- main loop ----------
    def fit(self):
        for epoch in range(self.start_epoch, self.cfg.epochs):
            # reshuffle length-bucketed batches each epoch (no-op for a plain sampler)
            bs = getattr(self.train_loader, "batch_sampler", None)
            if hasattr(bs, "set_epoch"):
                bs.set_epoch(epoch)
            self._run_epoch(self.train_loader, train=True, epoch=epoch)
            val_metrics, y_true, y_prob = self._run_epoch(self.val_loader, train=False, epoch=epoch)

            # figures
            self.writer.add_image("val/confusion_matrix",
                                  np.transpose(confusion_figure(y_true, y_prob.argmax(1),
                                                                self.cfg.class_names), (2, 0, 1)), epoch)
            if self.cfg.num_classes == 2:
                self.writer.add_image("val/roc",
                                      np.transpose(roc_figure(y_true, y_prob), (2, 0, 1)), epoch)
            if self.cfg.log_histograms:
                self._log_hist(epoch)
            if self.cfg.xai_enabled and (
                    epoch % self.cfg.xai_every_n_epochs == 0 or epoch == self.cfg.epochs - 1):
                self._log_xai(epoch)

            monitor = getattr(self.cfg, "monitor", "f1")
            score = val_metrics.get(monitor, val_metrics["f1"])
            print(f"epoch {epoch}: val[{monitor}]={score:.4f}  f1={val_metrics['f1']:.4f} "
                  f"bal_acc={val_metrics.get('balanced_acc', float('nan')):.4f} "
                  f"not_normal_sens={val_metrics.get('not_normal_sensitivity', float('nan')):.4f} "
                  f"auc={val_metrics['auc']:.4f}")

            # checkpoint + early stop on the configured monitor metric (higher = better)
            improved = score > self.best_f1
            if improved:
                self.best_f1 = score
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
            # always save full state as last.pt (exact resume); best.pt on improvement
            self._save_ckpt(os.path.join(self.cfg.out_dir, "last.pt"), epoch, val_metrics)
            if improved:
                self._save_ckpt(os.path.join(self.cfg.out_dir, "best.pt"), epoch, val_metrics)
            if not improved and self.epochs_no_improve >= self.cfg.early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        self.writer.add_hparams(
            {k: v for k, v in self.cfg.to_dict().items() if isinstance(v, (int, float, str, bool))},
            {f"hparam/best_val_{getattr(self.cfg, 'monitor', 'f1')}": self.best_f1},
        )
        self.writer.close()
        return self.best_f1

    @torch.no_grad()
    def _collect_probs(self, loader):
        """Run a loader through the model -> (y_true, y_prob) arrays."""
        self.model.eval()
        ys, ps = [], []
        for batch in loader:
            bag = batch["bag"].to(self.device)
            mask = batch["mask"].to(self.device)
            with torch.autocast(device_type="cuda" if self.use_amp else "cpu",
                                dtype=torch.float16, enabled=self.use_amp):
                logits = self.model(bag, mask)
            ys.extend(batch["label"].tolist())
            ps.extend(torch.softmax(logits.float(), 1).cpu().tolist())
        return np.array(ys), np.array(ps)

    # ---------- held-out test ----------
    @torch.no_grad()
    def test(self, loader, ckpt="best.pt"):
        """Evaluate the best checkpoint on a held-out loader; log test/* and return metrics."""
        path = os.path.join(self.cfg.out_dir, ckpt)
        if os.path.exists(path):
            state = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state["model"])
        all_true, all_prob = self._collect_probs(loader)
        metrics = compute_metrics(all_true, all_prob, self.cfg.num_classes,
                                  class_names=self.cfg.class_names)

        # Clinical operating point: choose the pathology-flag threshold on VAL to hit
        # the target not-normal sensitivity, then report sens/spec on TEST at that
        # threshold (no test leakage). Only meaningful for the 3-class screening view.
        if self.cfg.num_classes >= 3 and self.val_loader is not None:
            v_true, v_prob = self._collect_probs(self.val_loader)
            op = pathology_operating_point(v_true, v_prob,
                                           target_sensitivity=getattr(self.cfg, "target_sensitivity", 0.95))
            test_op = apply_operating_point(all_true, all_prob, op["threshold"])
            metrics["op_threshold"] = op["threshold"]
            metrics["op_val_sensitivity"] = op["sensitivity"]
            metrics["op_val_specificity"] = op["specificity"]
            metrics["op_test_sensitivity"] = test_op["op_sensitivity"]
            metrics["op_test_specificity"] = test_op["op_specificity"]
            print(f"[operating point @target_sens={op['target']:.2f}] "
                  f"thr={op['threshold']:.3f} | val sens/spec={op['sensitivity']:.3f}/{op['specificity']:.3f} "
                  f"| TEST sens/spec={test_op['op_sensitivity']:.3f}/{test_op['op_specificity']:.3f}")

        w = SummaryWriter(self.cfg.out_dir)                       # writer was closed after fit()
        for k, v in metrics.items():
            w.add_scalar(f"test/{k}", v, 0)
        w.add_image("test/confusion_matrix",
                    np.transpose(confusion_figure(np.array(all_true), np.array(all_prob).argmax(1),
                                                  self.cfg.class_names), (2, 0, 1)), 0)
        w.close()
        return metrics
