"""
engine/metrics.py
Evaluation metrics matching the paper (Sec 5.1, 6.1): accuracy, precision,
recall, F1, plus ROC-AUC. Confusion-matrix and ROC figures are rendered for
TensorBoard image logging.
"""
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, roc_curve,
)


def compute_metrics(y_true, y_prob, num_classes=2, class_names=None, normal_index=0):
    """y_true: (N,), y_prob: (N, C) softmax probs.

    Reports balanced (macro) metrics plus, for a clinical "don't miss pathology"
    view, per-class recall (= per-class sensitivity), balanced accuracy, and the
    normal-vs-not-normal framing: not_normal_sensitivity (caught pathology) and
    normal_specificity. `normal_index` is the class treated as 'normal' (0).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = y_prob.argmax(1)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    out = {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

    # per-class recall (= sensitivity per class) + balanced accuracy
    labels = list(range(num_classes))
    _, rec_pc, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    out["balanced_acc"] = float(np.mean(rec_pc))
    names = (list(class_names) if class_names and len(class_names) == num_classes
             else [str(i) for i in labels])
    for i, nm in enumerate(names):
        out[f"recall_{nm}"] = float(rec_pc[i])

    if num_classes == 2:
        out["specificity"] = float(rec_pc[0])   # normal recall
        out["sensitivity"] = float(rec_pc[1])   # abnormal recall
    elif num_classes >= 3:
        # normal vs not-normal (the clinical screening gate, under argmax)
        is_normal = y_true == normal_index
        n_norm = max(int(is_normal.sum()), 1)
        n_path = max(int((~is_normal).sum()), 1)
        out["normal_specificity"] = float(((y_pred == normal_index) & is_normal).sum() / n_norm)
        out["not_normal_sensitivity"] = float(((y_pred != normal_index) & ~is_normal).sum() / n_path)

    try:
        if num_classes == 2:
            out["auc"] = roc_auc_score(y_true, y_prob[:, 1])
        else:
            out["auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        out["auc"] = float("nan")           # only one class present in this split
    return out


def ruleout_metrics(y_true_bin, score, target_sensitivity=0.95):
    """Metrics for the binary rule-out head. `score` = P(not-normal) (higher = flag).

    Reports full AUC plus two TAIL-focused numbers used to track/select v5:
      * ruleout_spec_at_sens — max specificity at sensitivity >= target (the single
        auto-report operating point), and
      * ruleout_pauc — MEAN of that specificity over a grid of sensitivities in
        [target, ~1], i.e. a smooth 'how good is the whole high-sensitivity region'
        score. Smoother than a single point, so it's the preferred checkpoint monitor.
    """
    y_true_bin = np.asarray(y_true_bin).astype(int)
    score = np.asarray(score, dtype=float)
    out = {}
    try:
        out["ruleout_auc"] = float(roc_auc_score(y_true_bin, score))
    except ValueError:
        out["ruleout_auc"] = float("nan")
        out["ruleout_spec_at_sens"] = float("nan")
        out["ruleout_pauc"] = float("nan")
        return out
    fpr, tpr, _ = roc_curve(y_true_bin, score)

    def spec_at(t):
        ok = tpr >= t
        return float(1.0 - fpr[ok].min()) if ok.any() else 0.0

    out["ruleout_spec_at_sens"] = spec_at(target_sensitivity)
    grid = np.linspace(target_sensitivity, 0.999, 20)
    out["ruleout_pauc"] = float(np.mean([spec_at(t) for t in grid]))
    return out


def score_operating_point(y_true_bin, score, target_sensitivity=0.95):
    """Threshold on a 1-D score maximizing specificity s.t. sensitivity >= target."""
    y_true_bin = np.asarray(y_true_bin).astype(int)
    score = np.asarray(score, dtype=float)
    if y_true_bin.min() == y_true_bin.max():
        return {"threshold": float("nan"), "sensitivity": float("nan"), "specificity": float("nan")}
    fpr, tpr, thr = roc_curve(y_true_bin, score)
    ok = tpr >= target_sensitivity
    if ok.any():
        cand = np.where(ok)[0]
        idx = int(cand[np.argmin(fpr[cand])])
    else:
        idx = int(np.argmax(tpr))
    return {"threshold": float(thr[idx]), "sensitivity": float(tpr[idx]),
            "specificity": float(1.0 - fpr[idx])}


def apply_score_threshold(y_true_bin, score, threshold):
    """Sensitivity/specificity of a 1-D score at a fixed threshold (>= flags positive)."""
    y_true_bin = np.asarray(y_true_bin).astype(bool)
    flag = np.asarray(score, dtype=float) >= threshold
    npos = max(int(y_true_bin.sum()), 1)
    nneg = max(int((~y_true_bin).sum()), 1)
    return {"op_sensitivity": float((flag & y_true_bin).sum() / npos),
            "op_specificity": float((~flag & ~y_true_bin).sum() / nneg)}


def pathology_operating_point(y_true, y_prob, target_sensitivity=0.95, normal_index=0):
    """Pick a threshold on the pathology score s = 1 - P(normal) that MAXIMIZES
    specificity subject to (normal-vs-not-normal) sensitivity >= target.

    Choosing the threshold on a held-out split (val) and applying it to another
    (test) avoids argmax's symmetric bias when the priority is not missing
    near_normal / abnormal. Returns {threshold, sensitivity, specificity, target}.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    s = 1.0 - y_prob[:, normal_index]
    true_path = (y_true != normal_index).astype(int)
    if true_path.sum() in (0, len(true_path)):
        return {"threshold": float("nan"), "sensitivity": float("nan"),
                "specificity": float("nan"), "target": float(target_sensitivity)}
    fpr, tpr, thr = roc_curve(true_path, s)
    ok = tpr >= target_sensitivity
    if ok.any():                       # among points meeting the floor, minimize FPR (max specificity)
        cand = np.where(ok)[0]
        idx = int(cand[np.argmin(fpr[cand])])
    else:                              # floor unreachable -> take highest achievable sensitivity
        idx = int(np.argmax(tpr))
    return {"threshold": float(thr[idx]), "sensitivity": float(tpr[idx]),
            "specificity": float(1.0 - fpr[idx]), "target": float(target_sensitivity)}


def apply_operating_point(y_true, y_prob, threshold, normal_index=0):
    """Sensitivity/specificity (normal vs not-normal) on a split at a fixed threshold."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    s = 1.0 - y_prob[:, normal_index]
    flag = s >= threshold
    is_normal = y_true == normal_index
    n_norm = max(int(is_normal.sum()), 1)
    n_path = max(int((~is_normal).sum()), 1)
    return {
        "op_sensitivity": float((flag & ~is_normal).sum() / n_path),
        "op_specificity": float((~flag & is_normal).sum() / n_norm),
    }


def _fig_to_array(fig):
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = buf.reshape(h, w, 4)[..., :3].copy()
    plt.close(fig)
    return img            # HxWx3 uint8


def confusion_figure(y_true, y_pred, class_names=("normal", "abnormal")):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names)
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return _fig_to_array(fig)


def roc_figure(y_true, y_prob):
    fpr, tpr, _ = roc_curve(np.asarray(y_true), np.asarray(y_prob)[:, 1])
    auc = roc_auc_score(y_true, np.asarray(y_prob)[:, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve"); ax.legend(loc="lower right")
    fig.tight_layout()
    return _fig_to_array(fig)
