"""Evaluation metrics."""
from __future__ import annotations

import numpy as np
import torch
from pytorch_msssim import ssim
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


@torch.no_grad()
def psnr_per_image(x_hat: torch.Tensor, x: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    mse = ((x_hat.float() - x.float()) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return 10 * torch.log10(data_range ** 2 / mse)


@torch.no_grad()
def ssim_per_image(x_hat: torch.Tensor, x: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    return ssim(x_hat.float(), x.float(), data_range=data_range, size_average=False)


def restoration_score(psnr: float, ssim_val: float) -> float:
    """Optuna objective for restoration: 0.5 * min(PSNR, 40)/40 + 0.5 * SSIM  (both terms in [0, 1])."""
    return 0.5 * min(psnr, 40.0) / 40.0 + 0.5 * ssim_val


def classification_report(y_true, y_pred, labels=(0, 1, 2, 3), names=None) -> dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=list(labels), zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(labels), average="macro",
                                                    zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    cm_norm = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    names = names or [str(l) for l in labels]
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(mp), "macro_recall": float(mr), "macro_f1": float(mf),
            "per_class": {n: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                              "support": int(s[i])} for i, n in enumerate(names)},
            "confusion_matrix": cm.tolist(), "confusion_matrix_normalized": cm_norm.tolist()}
