import torch
import numpy as np
from scipy.stats import linregress
from collections import defaultdict
import csv
import os

class LatentTrustAnalyzer:
    def __init__(self, device, trust_threshold=0.5, log_dir="logs"):
        self.device = device
        self.V_ref = None
        self.trust_threshold = trust_threshold
        self.log_dir = log_dir

        # runtime buffers
        self.batch_metrics = defaultdict(list)
        self.trust_scores = {}
        self.metrics_summary = {}

        # initialize log file
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "trust_metrics.csv")

        # Write header once
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Round", "Client_ID",
                    "Avg_Effective_Rank", "Avg_Spectral_Slope",# "Avg_Similarity",
                    "Avg_Trust_Score"
                ])

    # -----------------------------------------------------
    def compute_svd_metrics(self, fx_client):
        """Compute SVD, effective rank, and spectral slope of activations."""
        B, C, H, W = fx_client.shape
        X = fx_client.view(B, -1)

        try:
            U, S, Vt = torch.linalg.svd(X, full_matrices=False)
        except RuntimeError:
            print("[Warning] SVD failed; using zeros instead.")
            S = torch.zeros(1, device=self.device)
            Vt = torch.zeros(1, 1, device=self.device)

        p = S / (S.sum() + 1e-8)
        effective_rank = (-p * torch.log(p + 1e-8)).sum().exp().item()

        x = np.arange(len(S.cpu()))
        ylog = np.log(S.cpu().numpy() + 1e-8)
        slope, _, _, _, _ = linregress(x, ylog)

        return Vt, effective_rank, slope

    # -----------------------------------------------------
    #def compare_subspace(self, Vt, k=10):
    #    """Compare client's subspace with reference."""
    #    if self.V_ref is None:
    #        self.V_ref = Vt.detach()
    #        return 1.0
    #    with torch.no_grad():
    #        V_ref_k = self.V_ref[:, :k]
    #        V_client_k = Vt[:, :k]
    #        sim = torch.norm(V_ref_k.T @ V_client_k, p='fro').item() / k
    #    return sim

    # -----------------------------------------------------
    def update(self, fx_client, client_id):
        """Batch-level update for one client."""
        fx_client = fx_client.to(self.device)
        Vt, effective_rank, slope = self.compute_svd_metrics(fx_client)
        #sim = self.compare_subspace(Vt)
        trust_score = 1 - 0.1 * abs(slope) - 0.05 * effective_rank

        self.batch_metrics[client_id].append({
            "rank": effective_rank,
            "slope": slope,
            #"sim": sim,
            "trust": trust_score
        })

        if trust_score < self.trust_threshold:
            print(f"[ALERT] Client {client_id}: latent anomaly detected "
                  f"(trust={trust_score:.3f}, rank={effective_rank:.2f},")# sim={sim:.2f})")

        #return trust_score

    # -----------------------------------------------------
    def finalize_client(self, client_id, round_id=0):
        """Aggregate all batch metrics per client and log to CSV."""
        if len(self.batch_metrics[client_id]) == 0:
            print(f"[WARN] No batch data for client {client_id}")
            return 0.0

        ranks = [m["rank"] for m in self.batch_metrics[client_id]]
        slopes = [m["slope"] for m in self.batch_metrics[client_id]]
        #sims = [m["sim"] for m in self.batch_metrics[client_id]]
        trusts = [m["trust"] for m in self.batch_metrics[client_id]]

        avg_rank = np.mean(ranks)
        avg_slope = np.mean(slopes)
        #avg_sim = np.mean(sims)
        avg_trust = np.mean(trusts)

        self.metrics_summary[client_id] = {
            "avg_rank": avg_rank,
            "avg_slope": avg_slope,
            #"avg_sim": avg_sim
        }
        self.trust_scores[client_id] = avg_trust

        # log to CSV
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                round_id, client_id,
                avg_rank, avg_slope, #avg_sim,
                avg_trust
            ])

        # optional clear after epoch
        self.batch_metrics[client_id].clear()

        print(f"[INFO] Client {client_id} Summary — "
              f"Trust={avg_trust:.3f}, Rank={avg_rank:.2f},")# Sim={avg_sim:.2f}")
        return avg_trust

    # -----------------------------------------------------
    #def set_reference(self, fx_client):
    #    """Set a clean reference subspace from known benign client."""
    #    _, _, Vt = torch.linalg.svd(fx_client.view(fx_client.size(0), -1), full_matrices=False)
    #    self.V_ref = Vt.detach()
    #    print("[INFO] Reference latent subspace set.")
