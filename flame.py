import copy
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from codecarbon import EmissionsTracker
import logging

# -------------------------
# Helpers
# -------------------------
def _flatten_state_dict(state_dict):
    parts = []
    for v in state_dict.values():
        parts.append(v.detach().cpu().float().reshape(-1))
    return torch.cat(parts, dim=0)

def _unflatten_to_state_dict(flat, template_state_dict):
    out = {}
    idx = 0
    flat = flat.detach().cpu()
    for k, v in template_state_dict.items():
        numel = v.numel()
        chunk = flat[idx: idx + numel].reshape(v.shape).to(v.device).type(v.dtype)
        out[k] = chunk.clone()
        idx += numel
    return out

# -------------------------
# FLAME-style Aggregator (Class-based)
# -------------------------
class FLAME:
    def __init__(self,
                 n_clusters: int = 3,
                 clip_percentile: float = 95.0,
                 noise_scale_factor: float = 1.0,
                 pca_components: int = 5,
                 random_state: int = 0):
        """
        Args:
            n_clusters: maximum number of clusters
            clip_percentile: percentile inside each cluster used for clipping threshold
            noise_scale_factor: multiplier for Gaussian noise scale
            pca_components: number of PCA dimensions to keep before clustering
            random_state: RNG seed
        """
        self.n_clusters = n_clusters
        self.clip_percentile = clip_percentile
        self.noise_scale_factor = noise_scale_factor
        self.pca_components = pca_components
        self.random_state = random_state

    def _normalize_inputs(self, entity, nature):
        """
        Converts input into a list of state_dicts,
        depending on whether entity is clients or server.
        """
        if nature == "clients":
            return [c.model.state_dict() for c in entity]

        elif nature == "server":
            return [entity.net_model_server[i].state_dict()
                    for i in range(entity.get_num_clients())]

        else:
            raise ValueError(f"Unknown nature {nature}")


    def aggregator(self,
                   global_state,
                   entity,
                   nature="clients",
                   client_weights=None):
        """
        Args:
            global_state: server's current state_dict
            entity: either clients (list) or server (object)
            nature: "clients" | "server"
            client_weights: optional list of floats
        Returns:
            new global state_dict, agg (emissions)
        """
        tracker = EmissionsTracker(log_level=logging.CRITICAL)
        tracker.start()

        # normalize input → list of state_dicts
        #client_states = self._normalize_inputs(entity, nature)
        client_states = entity
        num_clients = len(client_states)
        if num_clients == 0:
            return copy.deepcopy(global_state), tracker.stop()

        # flatten states
        global_vec = _flatten_state_dict(global_state).numpy()
        client_vecs = np.stack(
            [_flatten_state_dict(sd).numpy() for sd in client_states], axis=0
        )
        deltas = client_vecs - global_vec[None, :]

        # clustering with PCA
        K = min(self.n_clusters, num_clients)
        if K <= 1:
            labels = np.zeros(num_clients, dtype=int)
        else:
            comps = min(self.pca_components, deltas.shape[1], num_clients)
            reduced = PCA(n_components=comps, random_state=self.random_state).fit_transform(deltas)
            km = KMeans(n_clusters=K, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(reduced)

        norms = np.linalg.norm(deltas, axis=1)
        processed = np.zeros_like(deltas, dtype=np.float32)

        for c in range(K):
            idxs = np.where(labels == c)[0]
            if len(idxs) == 0:
                continue

            clust_norms = norms[idxs]
            clip_thresh = max(1e-12, float(np.percentile(clust_norms, self.clip_percentile)))

            if len(idxs) == 1:
                sigma_est = 0.0
            else:
                sigma_est = float(np.std(deltas[idxs, :], axis=0).mean())

            sigma = max(1e-12, sigma_est * self.noise_scale_factor)

            for i in idxs:
                d = deltas[i].astype(np.float32)
                norm = norms[i]
                if norm > clip_thresh:
                    d = d * (clip_thresh / (norm + 1e-12))
                if sigma > 0:
                    noise = np.random.normal(0.0, sigma, size=d.shape).astype(np.float32)
                    d = d + noise
                processed[i] = d

        # weighted average
        if client_weights is None:
            weights = np.ones(num_clients, dtype=np.float32) / num_clients
        else:
            weights = np.array(client_weights, dtype=np.float32)
            weights = weights / (weights.sum() + 1e-12)

        agg_delta = np.sum(processed * weights[:, None], axis=0).astype(np.float32)
        new_global_vec = (global_vec + agg_delta).astype(np.float32)
        agg: float = tracker.stop()

        return _unflatten_to_state_dict(torch.from_numpy(new_global_vec), global_state), agg
