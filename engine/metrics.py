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


def compute_metrics(y_true, y_prob, num_classes=2):
    """y_true: (N,), y_prob: (N, C) softmax probs."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = y_prob.argmax(1)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    out = {"accuracy": acc, "precision": p, "recall": r, "f1": f1}
    if num_classes == 2:
        # per-class recall (positive = abnormal = class 1)
        _, rec_pc, _, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1], average=None, zero_division=0
        )
        out["specificity"] = rec_pc[0]   # normal recall   = TN / (TN + FP)
        out["sensitivity"] = rec_pc[1]   # abnormal recall = TP / (TP + FN)
    try:
        if num_classes == 2:
            out["auc"] = roc_auc_score(y_true, y_prob[:, 1])
        else:
            out["auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        out["auc"] = float("nan")           # only one class present in this split
    return out


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
