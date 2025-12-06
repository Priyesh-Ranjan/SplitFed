import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List
import random
import math

# -------------------------
# Helper: unbiased (linear-time) MMD estimator (RBF kernel)
# We'll implement a stable, vectorized version.
# -------------------------
def class_mmd_table(normalized_dict):
    # collect unique clients for indexing
    clients = sorted(list({cid for pair in normalized_dict.keys() for cid in pair}))

    # header
    header = "       " + "  ".join([f"{c:>10}" for c in clients]) + "\n"
    rows = ""

    for ci in clients:
        row = [f"{ci:>5}"]
        for cj in clients:
            key = tuple(sorted((ci, cj)))
            val = normalized_dict.get(key, None)
            if val is None:
                row.append(f"{'--':>10}")
            else:
                row.append(f"{val:10.5f}")
        rows += "  ".join(row) + "\n"

    table = "\n" + header + rows + "\n"
    return table


def rbf_kernel_matrix(X, Y, gamma):
    # X: (n, d), Y: (m, d)
    # returns K_xy shape (n, m)
    # use ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x@y
    X_norm = (X**2).sum(dim=1).unsqueeze(1)  # (n,1)
    Y_norm = (Y**2).sum(dim=1).unsqueeze(0)  # (1,m)
    d2 = X_norm + Y_norm - 2.0 * (X @ Y.t())
    return torch.exp(-gamma * d2)

def unbiased_mmd2_rbf(X, Y, gamma):
    """
    Unbiased squared MMD estimator with RBF kernel.
    X: torch.Tensor (n, d)
    Y: torch.Tensor (m, d)
    gamma: kernel bandwidth parameter (1/(2*sigma^2)) or a list for multi-bandwidth averaging
    Returns scalar float (MMD^2)
    Note: uses the 'U-statistic' unbiased estimator.
    """
    if X.numel() == 0 or Y.numel() == 0:
        return float('nan')

    # If gamma is a list / array: average across gammas
    if isinstance(gamma, (list, tuple, np.ndarray)):
        vals = []
        for g in gamma:
            vals.append(unbiased_mmd2_rbf(X, Y, g))
        return float(np.mean(vals))

    n = X.shape[0]
    m = Y.shape[0]

    # For tiny sets fall back to biased but stable estimator (kernel mean squared)
    if n < 2 or m < 2:
        Kxx = rbf_kernel_matrix(X, X, gamma)
        Kyy = rbf_kernel_matrix(Y, Y, gamma)
        Kxy = rbf_kernel_matrix(X, Y, gamma)
        return float(Kxx.mean().item() + Kyy.mean().item() - 2.0 * Kxy.mean().item())

    # U-statistic for Kxx (exclude diagonal)
    Kxx = rbf_kernel_matrix(X, X, gamma)
    Kxx_sum = (Kxx.sum() - Kxx.diag().sum()) / (n * (n - 1))

    Kyy = rbf_kernel_matrix(Y, Y, gamma)
    Kyy_sum = (Kyy.sum() - Kyy.diag().sum()) / (m * (m - 1))

    Kxy = rbf_kernel_matrix(X, Y, gamma)
    Kxy_sum = Kxy.mean()

    mmd2 = Kxx_sum + Kyy_sum - 2.0 * Kxy_sum
    return float(mmd2)
import gc

# (Keep your rbf_kernel_matrix and unbiased_mmd2_rbf as-is above)

class ServerLatentMMDAnalyzer:
    def __init__(
        self,
        num_classes: int,
        device='cpu',
        kernel_gamma=None,
        max_samples_per_class=256,
        subsample_per_mmd=128,
        metric='mmd',
        clip_nan_to=1e6,
        smoothing_alpha=0.8,
        tau=None,
        min_pairs_per_client=3
    ):
        self.num_classes = num_classes
        self.device = 'cpu'  # keep buffers on CPU for memory stability
        self.kernel_gamma = kernel_gamma
        self.max_samples = int(max_samples_per_class)
        self.subsample = int(subsample_per_mmd)
        self.metric = metric
        self.clip_nan_to = clip_nan_to

        # smoothing
        self.smoothing_alpha = float(smoothing_alpha)
        self.tau = tau
        self.min_pairs_per_client = int(min_pairs_per_client)

        # storage buffers
        self.buffers: Dict[str, Dict[int, torch.Tensor]] = {}
        self.counts: Dict[str, Dict[int, int]] = {}

        # persistent EMA state
        self.prev_trust_scores: Dict[str, float] = {}

    def update_client_latent(self, client_id: str, latent: torch.Tensor, labels: torch.Tensor):
        latent = latent.detach().to(self.device)
        labels = labels.detach().to(self.device)

        if client_id not in self.buffers:
            self.buffers[client_id] = {}
            self.counts[client_id] = {}

        for c in range(self.num_classes):
            idx = (labels == c).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            Zc = latent[idx]

            if c not in self.buffers[client_id]:
                # first time
                to_store = Zc.clone().detach()
                if to_store.shape[0] > self.max_samples:
                    to_store = to_store[torch.randperm(to_store.size(0))[: self.max_samples]]
                self.buffers[client_id][c] = to_store
                self.counts[client_id][c] = to_store.size(0)
            else:
                # append with clipping
                cur = self.buffers[client_id][c]
                combined = torch.cat([cur, Zc], dim=0)
                if combined.size(0) > self.max_samples:
                    combined = combined[torch.randperm(combined.size(0))[: self.max_samples]]
                self.buffers[client_id][c] = combined
                self.counts[client_id][c] = combined.size(0)

    def compute_trust_scores(self, round_id: int = 0, eps=1e-8):
        client_ids = list(self.buffers.keys())
        n_clients = len(client_ids)
        if n_clients == 0:
            return {}, {}

        # per-class clients
        class_clients = {
            c: [cid for cid in client_ids if c in self.buffers[cid]]
            for c in range(self.num_classes)
        }

        client_mmd_sum = {cid: 0.0 for cid in client_ids}
        client_mmd_count = {cid: 0 for cid in client_ids}

        per_pair_info = []  # used for tau selection

        # NEW: class-wise pairwise MMD storage
        class_pair_mmds = {c: {} for c in range(self.num_classes)}

        # ------------------------
        # MAIN LOOP: per-class pairwise MMD
        # ------------------------
        for c in range(self.num_classes):
            ids = class_clients[c]
            if len(ids) < 2:
                continue

            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    ci = ids[i]
                    cj = ids[j]

                    Xi = self.buffers[ci][c]
                    Xj = self.buffers[cj][c]

                    # subsample
                    si = min(Xi.size(0), self.subsample)
                    sj = min(Xj.size(0), self.subsample)

                    Xi_s = Xi[torch.randperm(Xi.size(0))[:si]] if si < Xi.size(0) else Xi
                    Xj_s = Xj[torch.randperm(Xj.size(0))[:sj]] if sj < Xj.size(0) else Xj

                    # gamma selection
                    gamma = self.kernel_gamma
                    if gamma is None:
                        XY = torch.cat([Xi_s, Xj_s], dim=0)
                        M = min(512, XY.size(0))
                        XYs = XY[torch.randperm(XY.size(0))[:M]] if XY.size(0) > M else XY
                        mat = torch.cdist(XYs, XYs)
                        med = float(np.median(mat.cpu().numpy().reshape(-1)))
                        if med <= 0:
                            med = 1.0
                        gamma = 1.0 / (2.0 * med)

                    # compute MMD
                    try:
                        mmd2 = unbiased_mmd2_rbf(Xi_s, Xj_s, gamma)
                        if math.isnan(mmd2):
                            mmd2 = self.clip_nan_to
                    except Exception:
                        Kxx = rbf_kernel_matrix(Xi_s, Xi_s, gamma)
                        Kyy = rbf_kernel_matrix(Xj_s, Xj_s, gamma)
                        Kxy = rbf_kernel_matrix(Xi_s, Xj_s, gamma)
                        mmd2 = float(Kxx.mean() + Kyy.mean() - 2 * Kxy.mean())

                    # ---------------------------
                    # NEW: store per-class, per-pair MMD
                    # ---------------------------
                    #class_pair_mmds[c][(ci, cj)] = float(mmd2)
                    # NEW: accumulate per class, per client-pair
                    pair_key = tuple(sorted((ci, cj)))

                    if pair_key not in class_pair_mmds[c]:
                        class_pair_mmds[c][pair_key] = {"mmd_sum": float(mmd2), "count": 1}
                    else:
                        class_pair_mmds[c][pair_key]["mmd_sum"] += float(mmd2)
                        class_pair_mmds[c][pair_key]["count"] += 1
                    # bookkeeping
                    client_mmd_sum[ci] += mmd2
                    client_mmd_sum[cj] += mmd2
                    client_mmd_count[ci] += 1
                    client_mmd_count[cj] += 1
                    per_pair_info.append(mmd2)

        # -------------------------
        # Average per client
        # -------------------------
        client_mean_mmd = {
            cid: (client_mmd_sum[cid] / (client_mmd_count[cid] + 1e-12)
                  if client_mmd_count[cid] > 0 else float('nan'))
            for cid in client_ids
        }

        # convert to numpy
        vals = np.array([v for v in client_mean_mmd.values()], dtype=np.float64)
        valid = ~np.isnan(vals)
        valid_vals = vals[valid]

        if valid_vals.size == 0:
            neutral = {cid: 0.5 for cid in client_ids}
            return neutral, {"client_mean_mmd": client_mean_mmd,
                             "class_pair_mmds": class_pair_mmds}

        # tau selection
        tau = self.tau
        if tau is None:
            tau = max(1e-6, float(np.median(per_pair_info))) if len(per_pair_info) else max(1e-6, float(np.median(valid_vals)))

        # score mapping
        raw_scores = np.zeros(len(client_ids))
        median_valid = np.median(valid_vals)

        for i, cid in enumerate(client_ids):
            v = client_mean_mmd[cid]
            if math.isnan(v) or client_mmd_count[cid] < self.min_pairs_per_client:
                raw_scores[i] = math.exp(-median_valid / tau)
            else:
                raw_scores[i] = math.exp(-v / tau)

        # normalize to [0,1]
        max_r = float(np.max(raw_scores))
        norm_arr = raw_scores / (max_r + 1e-12) if max_r > 0 else np.ones_like(raw_scores) * 0.5

        # EMA smoothing
        norm_scores = {}
        for i, cid in enumerate(client_ids):
            new = float(norm_arr[i])
            prev = self.prev_trust_scores.get(cid)
            smoothed = new if prev is None else self.smoothing_alpha * prev + (1 - self.smoothing_alpha) * new
            self.prev_trust_scores[cid] = smoothed
            norm_scores[cid] = smoothed

        
        # --------------------------------------------
# NEW: normalize class-pair MMDs (mmd_sum / count)
# --------------------------------------------
        normalized_class_pair_mmds = {}

        for c in range(self.num_classes):
            normalized_class_pair_mmds[c] = {}
            for pair, vals in class_pair_mmds[c].items():
                count = vals["count"]
                if count > 0:
                    normalized = vals["mmd_sum"] / count
                else:
                    normalized = float("nan")
                normalized_class_pair_mmds[c][pair] = normalized

        class_tables = {
    c: class_mmd_table(normalized_class_pair_mmds[c])
    for c in range(self.num_classes)}        

        diagnostics = {
    "client_mean_mmd": client_mean_mmd,
    "raw_scores": raw_scores.tolist(),
    "tau_used": tau,
    "pair_counts": {cid: client_mmd_count[cid] for cid in client_ids},

    # NEW:
    "class_pair_mmds": class_pair_mmds,
    "normalized_class_pair_mmds": normalized_class_pair_mmds,
    "class_pair_tables": class_tables}

        print(diagnostics)
        tables = diagnostics["class_pair_tables"]
        for c, table in tables.items():
            print(f"\n===== Class {c} Pairwise MMD Table =====")
            print(table)
        return norm_scores, diagnostics

    def reset(self, keep_ema=False):
        self.buffers = {}
        self.counts = {}
        if not keep_ema:
            self.prev_trust_scores = {}
        gc.collect()


