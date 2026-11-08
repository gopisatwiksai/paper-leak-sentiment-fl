"""
Step 7 – Federated Learning with FedAvg (implemented from scratch in NumPy).

Model
  Multinomial logistic regression (softmax) on hashing features.
  Parameters: W (262,144 x 3) and b (3). Small, fast and easy to explain.

FedAvg (McMahan et al., 2017)
  for each round r = 1..R:
      server sends the global model (W, b) to every client
      each client trains it on ITS OWN comments for E local epochs (mini-batch SGD)
      each client sends back ONLY its updated (W, b) – never any text
      server averages them, weighted by each client's number of comments:
              W_global = sum_k (n_k / n) * W_k
  The global test set is held by an evaluator and used only to score the model.

Optional extensions (used in Step 8)
  mu > 0           -> FedProx: keeps local models close to the global one (helps non-IID data)
  dp_noise > 0     -> Differential-privacy style protection: each client update is clipped and
                      Gaussian noise is added on the server before averaging.

Run:  python src/federated.py          (main experiment: 5 platform clients, non-IID)
"""
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from preprocess import make_hashing

ROOT = Path(__file__).resolve().parents[1]
DATA, MODELS = ROOT / "data", ROOT / "models"
TABLES, REPORTS = ROOT / "results" / "tables", ROOT / "reports"
LABELS = np.array(["negative", "neutral", "positive"])
PLATFORMS = ["YouTube", "X", "Instagram", "Reddit", "NewsPortal"]


@dataclass
class FLConfig:
    rounds: int = 40            # communication rounds
    local_epochs: int = 2       # passes over local data per round
    batch_size: int = 64
    lr: float = 1.2             # learning rate at round 1
    lr_decay: float = 0.05      # lr_r = lr / (1 + lr_decay * r)
    l2: float = 1e-6            # weight decay
    mu: float = 0.0             # FedProx strength (0 = plain FedAvg)
    dp_clip: float = 0.0        # max norm of one client's update (0 = no clipping)
    dp_noise: float = 0.0       # noise multiplier (0 = no noise)
    seed: int = 42


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
def softmax_logits(X, W, b):
    Z = X @ W + b
    Z -= Z.max(axis=1, keepdims=True)
    P = np.exp(Z)
    return P / P.sum(axis=1, keepdims=True)


def predict(X, W, b):
    return LABELS[(X @ W + b).argmax(axis=1)]


def scores(X, y, W, b):
    p = predict(X, W, b)
    return accuracy_score(y, p), f1_score(y, p, average="macro")


def local_train(X, y_idx, W, b, cfg, lr, rng, W_global=None, b_global=None):
    """One client's training: mini-batch SGD on its own data, starting from the global model.

    Only the weight rows of words that appear in a batch change, so the update is applied
    sparsely (much faster). Weight decay and the FedProx pull towards the global model are
    applied once per local epoch.
    """
    W, b = W.copy(), b.copy()
    n = X.shape[0]
    n_batches = int(np.ceil(n / cfg.batch_size))
    for _ in range(cfg.local_epochs):
        order = rng.permutation(n)
        for s in range(0, n, cfg.batch_size):
            idx = order[s:s + cfg.batch_size]
            Xb = X[idx].tocoo()
            G = softmax_logits(X[idx], W, b)
            G[np.arange(len(idx)), y_idx[idx]] -= 1.0          # gradient of cross-entropy w.r.t. logits
            G /= len(idx)
            np.add.at(W, Xb.col, -lr * Xb.data[:, None] * G[Xb.row])   # W -= lr * X_b^T G  (sparse)
            b -= lr * G.sum(axis=0)
        if cfg.l2 > 0:
            W *= (1 - lr * cfg.l2) ** n_batches
        if cfg.mu > 0:                                           # FedProx: pull back towards the global model
            keep = (1 - lr * cfg.mu) ** n_batches
            W = W_global + (W - W_global) * keep
            b = b_global + (b - b_global) * keep
    return W, b


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_clients(kind="platform"):
    """kind='platform' -> 5 non-IID platform clients;  kind='iid' -> 5 random equal clients."""
    if kind == "platform":
        names = PLATFORMS
        files = [(DATA / "clients" / f"{p.lower()}_train.csv", DATA / "clients" / f"{p.lower()}_val.csv") for p in PLATFORMS]
    else:
        names = [f"Client {k}" for k in range(1, 6)]
        files = [(DATA / "clients_iid" / f"client{k}_train.csv", DATA / "clients_iid" / f"client{k}_val.csv") for k in range(1, 6)]
    hv = make_hashing()                       # every client builds features ON ITS OWN – no shared vocabulary
    clients = []
    for name, (ftr, fva) in zip(names, files):
        tr, va = pd.read_csv(ftr), pd.read_csv(fva)
        clients.append(dict(name=name, X=hv.transform(tr.clean_text.astype(str)).tocsr(),
                            y=tr.label.values, y_idx=np.searchsorted(LABELS, tr.label.values),
                            Xval=hv.transform(va.clean_text.astype(str)).tocsr(), yval=va.label.values))
    return clients


def load_test():
    te = pd.read_csv(DATA / "processed" / "test.csv")
    return make_hashing().transform(te.clean_text.astype(str)).tocsr(), te.label.values, te


# ---------------------------------------------------------------------------
# Training procedures
# ---------------------------------------------------------------------------
def fedavg(clients, Xte, yte, cfg=FLConfig(), verbose=True):
    rng = np.random.default_rng(cfg.seed)
    d = clients[0]["X"].shape[1]
    W, b = np.zeros((d, 3)), np.zeros(3)
    n_total = sum(c["X"].shape[0] for c in clients)
    history, mb_per_update = [], (W.size + b.size) * 4 / 1e6          # float32 bytes -> MB
    for r in range(1, cfg.rounds + 1):
        lr = cfg.lr / (1 + cfg.lr_decay * (r - 1))
        t0 = time.perf_counter()
        updates, weights = [], []
        for c in clients:                                              # (in reality these run in parallel)
            Wk, bk = local_train(c["X"], c["y_idx"], W, b, cfg, lr, rng, W, b)
            updates.append((Wk - W, bk - b))
            weights.append(c["X"].shape[0] / n_total)
        if cfg.dp_clip > 0:                                            # clip each client's update
            clipped = []
            for dW, db in updates:
                norm = np.sqrt((dW ** 2).sum() + (db ** 2).sum())
                f = min(1.0, cfg.dp_clip / (norm + 1e-12))
                clipped.append((dW * f, db * f))
            updates = clipped
        dW = sum(w * u[0] for w, u in zip(weights, updates))           # FedAvg: weighted average
        db = sum(w * u[1] for w, u in zip(weights, updates))
        if cfg.dp_noise > 0:                                           # Gaussian noise on the aggregate
            sigma = cfg.dp_noise * cfg.dp_clip / len(clients)
            dW = dW + rng.normal(0, sigma, dW.shape)
            db = db + rng.normal(0, sigma, db.shape)
        W, b = W + dW, b + db
        acc, f1 = scores(Xte, yte, W, b)
        row = dict(round=r, lr=round(lr, 4), test_accuracy=acc, test_macro_f1=f1,
                   uploaded_mb_total=round(mb_per_update * len(clients) * r, 1), seconds=round(time.perf_counter() - t0, 2))
        for c in clients:
            row[f"val_acc_{c['name']}"] = scores(c["Xval"], c["yval"], W, b)[0]
        history.append(row)
        if verbose and (r == 1 or r % 5 == 0):
            print(f"round {r:3d}  test acc {acc:.4f}  macro-F1 {f1:.4f}  ({row['seconds']}s)")
    return W, b, pd.DataFrame(history)


def centralized_softmax(clients, Xte, yte, cfg=FLConfig(), verbose=False):
    """Same model and optimiser, but all client data pooled (no privacy) – the fair upper bound."""
    from scipy.sparse import vstack
    X = vstack([c["X"] for c in clients]).tocsr()
    y_idx = np.concatenate([c["y_idx"] for c in clients])
    rng = np.random.default_rng(cfg.seed)
    W, b = np.zeros((X.shape[1], 3)), np.zeros(3)
    history = []
    for r in range(1, cfg.rounds + 1):                                 # one "round" = one epoch over all data
        lr = cfg.lr / (1 + cfg.lr_decay * (r - 1))
        W, b = local_train(X, y_idx, W, b, cfg, lr, rng)
        acc, f1 = scores(Xte, yte, W, b)
        history.append(dict(round=r, test_accuracy=acc, test_macro_f1=f1))
        if verbose and (r == 1 or r % 5 == 0):
            print(f"epoch {r:3d}  test acc {acc:.4f}")
    return W, b, pd.DataFrame(history)


def local_only(client, Xte, yte, cfg=FLConfig()):
    """A client that refuses to collaborate: trains only on its own data."""
    rng = np.random.default_rng(cfg.seed)
    W, b = np.zeros((client["X"].shape[1], 3)), np.zeros(3)
    for r in range(1, cfg.rounds + 1):
        lr = cfg.lr / (1 + cfg.lr_decay * (r - 1))
        W, b = local_train(client["X"], client["y_idx"], W, b, cfg, lr, rng)
    return W, b, scores(Xte, yte, W, b)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def main():
    cfg = FLConfig()
    print("Loading the 5 platform clients (each one hashes its own text)...")
    clients = load_clients("platform")
    Xte, yte, _ = load_test()
    for c in clients:
        print(f"  {c['name']:11s} {c['X'].shape[0]:6,d} comments")
    print(f"\nFedAvg: {cfg.rounds} rounds, {cfg.local_epochs} local epoch(s), lr {cfg.lr}")
    W, b, hist = fedavg(clients, Xte, yte, cfg)

    print("\nCentralized reference (same model, pooled data)...")
    Wc, bc, hist_c = centralized_softmax(clients, Xte, yte, cfg)

    MODELS.mkdir(exist_ok=True); TABLES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(MODELS / "fedavg_global_model.npz", W=W.astype(np.float32), b=b.astype(np.float32), labels=LABELS)
    np.savez_compressed(MODELS / "centralized_softmax_model.npz", W=Wc.astype(np.float32), b=bc.astype(np.float32), labels=LABELS)
    hist.to_csv(TABLES / "fedavg_history.csv", index=False)
    hist_c.to_csv(TABLES / "centralized_softmax_history.csv", index=False)

    final, final_c = hist.iloc[-1], hist_c.iloc[-1]
    summary = dict(config=asdict(cfg), clients={c["name"]: int(c["X"].shape[0]) for c in clients},
                   fedavg_test_accuracy=round(final.test_accuracy, 4), fedavg_test_macro_f1=round(final.test_macro_f1, 4),
                   centralized_test_accuracy=round(final_c.test_accuracy, 4), centralized_test_macro_f1=round(final_c.test_macro_f1, 4),
                   gap_percentage_points=round((final_c.test_accuracy - final.test_accuracy) * 100, 2),
                   model_size_mb=round((W.size + b.size) * 4 / 1e6, 2), total_uploaded_mb=float(final.uploaded_mb_total),
                   raw_comments_shared=0)
    (REPORTS / "fedavg_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nFedAvg      : accuracy {final.test_accuracy:.4f}  macro-F1 {final.test_macro_f1:.4f}")
    print(f"Centralized : accuracy {final_c.test_accuracy:.4f}  macro-F1 {final_c.test_macro_f1:.4f}")
    print(f"Gap         : {summary['gap_percentage_points']} percentage points   |   raw comments shared: 0")


if __name__ == "__main__":
    main()
