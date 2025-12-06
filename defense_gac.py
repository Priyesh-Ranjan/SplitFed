import torch
import torch.nn.functional as F
from collections import defaultdict
import numpy as np

class GradientActivationCorrelationAnalyzer:
    """
    Computes gradient–activation correlation (GAC) per client.
    Call add_batch() each time the server receives activations and gradients.
    Call finalize_scores() after all clients finish for the round.
    """

    def __init__(self, tau=0.5, eps=1e-8):
        self.tau = tau
        self.eps = eps

        # Store sums per client
        self.client_corr_sum = defaultdict(float)
        self.client_corr_count = defaultdict(int)

        # Optional: class-wise correlation
        self.class_corr_sum = defaultdict(lambda: defaultdict(float))
        self.class_corr_count = defaultdict(lambda: defaultdict(int))

        self.client_ids_seen = set()

    # ---------------------------------------------------------------
    def _compute_batch_corr(self, activations, gradients):
        """
        activations: [B, D]
        gradients:   [B, D]
        Computes per-sample cosine similarity and returns mean over batch.
        """

        # Normalize
        a_norm = F.normalize(activations, p=2, dim=1)
        g_norm = F.normalize(gradients, p=2, dim=1)

        # Cosine similarity per sample
        corr = torch.sum(a_norm * g_norm, dim=1)  # [B]
        return corr.mean().item()

    # ---------------------------------------------------------------
    def add_batch(self, client_id, activations, gradients, labels):
        """
        Called each time the server receives one batch from a client.
        activations: output of client model (cut layer)  [B, D]
        gradients:   backward signal from server         [B, D]
        labels:      class labels                       [B]
        """

        self.client_ids_seen.add(client_id)

        if activations.ndim > 2:
            activations = activations.view(activations.size(0), -1)
        if gradients.ndim > 2:
            gradients = gradients.view(gradients.size(0), -1)

        # -------- Per-batch correlation --------
        batch_corr = self._compute_batch_corr(activations, gradients)

        # Aggregate per client
        self.client_corr_sum[client_id] += batch_corr
        self.client_corr_count[client_id] += 1

        # -------- Class-wise correlation --------
        labels_np = labels.detach().cpu().numpy()

        for cls in np.unique(labels_np):
            mask = (labels_np == cls)
            if mask.sum() == 0:
                continue

            a_cls = activations[mask]
            g_cls = gradients[mask]

            corr_cls = self._compute_batch_corr(a_cls, g_cls)

            self.class_corr_sum[client_id][int(cls)] += corr_cls
            self.class_corr_count[client_id][int(cls)] += 1

    # ---------------------------------------------------------------
    def finalize_scores(self):
        """
        Computes normalized trust scores for all clients.
        """
        client_ids = sorted(self.client_ids_seen)

        # Compute per-client avg GAC
        client_avg = {}
        for cid in client_ids:
            if self.client_corr_count[cid] == 0:
                client_avg[cid] = 0.0
            else:
                client_avg[cid] = self.client_corr_sum[cid] / self.client_corr_count[cid]

        # Convert to numpy for thresholding
        raw_scores = np.array([client_avg[cid] for cid in client_ids])

        # Normalize into [0,1]
        if len(raw_scores) > 1:
            min_val, max_val = raw_scores.min(), raw_scores.max()
            if max_val - min_val < 1e-9:
                norm_scores = np.ones_like(raw_scores) * 0.5
            else:
                norm_scores = (raw_scores - min_val) / (max_val - min_val)
        else:
            norm_scores = np.array([1.0])

        # Apply threshold
        trust_scores = {cid: float(norm_scores[i] >= self.tau)
                        for i, cid in enumerate(client_ids)}

        # Build class-wise tables
        class_tables = {}
        for cid in client_ids:
            table = {}
            for cls in sorted(self.class_corr_sum[cid].keys()):
                cnt = self.class_corr_count[cid][cls]
                table[cls] = (
                    self.class_corr_sum[cid][cls] / cnt if cnt > 0 else 0.0
                )
            class_tables[cid] = table

        diagnostics = {
            "client_avg_corr": {cid: float(client_avg[cid]) for cid in client_ids},
            "normalized_scores": {cid: float(norm_scores[i]) for i, cid in enumerate(client_ids)},
            "trust_scores": trust_scores,
            "tau_used": self.tau,
            "class_tables": class_tables,
        }
        
        print(diagnostics)     
        return trust_scores, diagnostics
    
    def reset(self):
        """
        Clears all correlation statistics for the next round.
        """
        self.client_corr_sum = defaultdict(float)
        self.client_corr_count = defaultdict(int)

        self.class_corr_sum = defaultdict(lambda: defaultdict(float))
        self.class_corr_count = defaultdict(lambda: defaultdict(int))

        self.client_ids_seen = set()
