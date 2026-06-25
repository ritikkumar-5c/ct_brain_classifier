"""
train_main.py
Entrypoint. Example:

  python train_main.py \
      --data_root /data/ct_studies --labels_csv /data/labels.csv \
      --backbone maxvit --epochs 50 --lr 3e-4 --batch_size 8 \
      --out_dir runs/maxvit_mil

Then:  tensorboard --logdir runs
"""
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import get_config
from data.dicom_dataset import (
    index_studies, index_studies_from_csv, grouped_split, class_weights,
    StudyMILDataset, mil_collate, _study_of,
    effective_lengths, LengthBucketedBatchSampler,
)
from models.maxvit_mil import build_model
from engine.trainer import Trainer


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    # accept any Config field as --field
    for k, v in get_config().to_dict().items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
        elif isinstance(v, (int, float, str)):
            p.add_argument(f"--{k}", type=type(v), default=None)
    return p.parse_args()


def main():
    args = parse_args()
    overrides = {k: v for k, v in vars(args).items() if v is not None}
    cfg = get_config(**overrides)
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if cfg.train_csv:
        # (A) preferred: pre-split path CSVs (already patient-grouped by split_classifier_csv.py)
        train_items = index_studies_from_csv(cfg.train_csv, cfg.class_names)
        val_items = index_studies_from_csv(cfg.val_csv, cfg.class_names) if cfg.val_csv else []
        test_items = index_studies_from_csv(cfg.test_csv, cfg.class_names) if cfg.test_csv else []
    else:
        # (B) legacy: data_root + labels_csv, split in-memory (study-grouped, stratified)
        items = index_studies(cfg.data_root, cfg.labels_csv, cfg.class_names)
        train_items, val_items, test_items = grouped_split(
            items, cfg.val_fraction, cfg.test_fraction, cfg.seed)

    def npat(its):                                    # distinct studies (patients) in a split
        return len({_study_of(it[0]) for it in its})
    print(f"series : train={len(train_items)} val={len(val_items)} test={len(test_items)}")
    print(f"patients: train={npat(train_items)} val={npat(val_items)} test={npat(test_items)}")
    # sanity: splits must be patient-disjoint
    g = [{_study_of(it[0]) for it in s} for s in (train_items, val_items, test_items)]
    assert not (g[0] & g[1]) and not (g[0] & g[2]) and not (g[1] & g[2]), "patient leakage across splits!"

    cw = class_weights(train_items, cfg.num_classes) if cfg.use_class_weights else None
    print(f"class_weights={cw}")

    def loader(its, train):
        ds = StudyMILDataset(its, cfg, train=train)
        if cfg.length_bucketing:
            bsampler = LengthBucketedBatchSampler(
                effective_lengths(its, cfg, train), cfg.batch_size, shuffle=train,
                seed=cfg.seed, pool_factor=cfg.bucket_pool_factor)
            return DataLoader(ds, batch_sampler=bsampler, num_workers=cfg.num_workers,
                              collate_fn=mil_collate, pin_memory=True)
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=train,
                          num_workers=cfg.num_workers, collate_fn=mil_collate, pin_memory=True)
    train_loader = loader(train_items, True)
    val_loader = loader(val_items, False)

    model = build_model(cfg)
    trainer = Trainer(cfg, model, train_loader, val_loader, cw, device)
    best_f1 = trainer.fit()
    print(f"Best val F1: {best_f1:.4f}  (checkpoint: {cfg.out_dir}/best.pt)")

    # final evaluation on the held-out test set using the best checkpoint
    if test_items:
        test_metrics = trainer.test(loader(test_items, False))
        print("TEST (best ckpt): " + "  ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()))


if __name__ == "__main__":
    main()
