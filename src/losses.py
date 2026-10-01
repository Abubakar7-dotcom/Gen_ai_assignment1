"""Loss functions. All image losses expect float32 tensors in [0, 1] (restoration) unless stated."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from pytorch_msssim import ssim


def ssim_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return 1.0 - ssim(x_hat.float(), x.float(), data_range=1.0, size_average=True)


def recon_loss(x_hat: torch.Tensor, x: torch.Tensor, alpha: float = 0.8) -> tuple[torch.Tensor, dict]:
    """L_UDAE = alpha * L1 + (1 - alpha) * (1 - SSIM)."""
    l1 = F.l1_loss(x_hat.float(), x.float())
    ls = ssim_loss(x_hat, x)
    return alpha * l1 + (1 - alpha) * ls, {"l1": l1.item(), "ssim_loss": ls.item()}


def balance_loss(w: torch.Tensor) -> torch.Tensor:
    """sum_k (mean_batch(w_k) - 1/K)^2 — penalises routing collapse onto one branch (assignment eq.)."""
    k = w.shape[1]
    return ((w.mean(0) - 1.0 / k) ** 2).sum()


def moe_loss(x_hat, x, w, logits, y, l1_w=0.8, ssim_w=0.2, ce_w=0.1, bal_w=0.01) -> tuple[torch.Tensor, dict]:
    """L_MoE = l1_w*L1 + ssim_w*(1-SSIM) + ce_w*CE(gate, label) + bal_w*L_balance."""
    l1 = F.l1_loss(x_hat.float(), x.float())
    ls = ssim_loss(x_hat, x)
    ce = F.cross_entropy(logits.float(), y)
    bal = balance_loss(w.float())
    total = l1_w * l1 + ssim_w * ls + ce_w * ce + bal_w * bal
    return total, {"l1": l1.item(), "ssim_loss": ls.item(), "ce": ce.item(), "balance": bal.item()}


# ----------------------------------------------------------------- GAN (BCE with logits)
def d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> tuple[torch.Tensor, dict]:
    lr = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
    lf = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    return 0.5 * (lr + lf), {"d_real": lr.item(), "d_fake": lf.item()}


def g_loss(fake_logits: torch.Tensor, fake: torch.Tensor, target: torch.Tensor, lambda_l1: float = 100.0):
    """L_G = L_adv + lambda_L1 * L1(y, G(x,s))."""
    adv = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
    l1 = F.l1_loss(fake, target)
    return adv + lambda_l1 * l1, {"g_adv": adv.item(), "g_l1": l1.item()}
