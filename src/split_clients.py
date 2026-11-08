"""
Step 6 – Split the training data into federated-learning clients.

Main setting (realistic, non-IID)
  One client per platform: YouTube, X, Instagram, Reddit, NewsPortal.
  Each client keeps ONLY its own comments. Clients differ in size, sentiment mix
  and language, as real platforms would.

Comparison setting (IID)
  The same training data shuffled into 5 equal random clients. Every client then
  looks like the whole dataset. Used in Step 8 to show how much the realistic
  non-IID split makes federated learning harder.

The global TEST set is never split: it stays with the evaluator and is only used
to score the final models.

Run:  python src/split_clients.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
CLIENTS = ROOT / "data" / "clients"
CLIENTS_IID = ROOT / "data" / "clients_iid"
PLATFORMS = ["YouTube", "X", "Instagram", "Reddit", "NewsPortal"]
LABELS = ["negative", "neutral", "positive"]
SEED = 42


def js_divergence(p, q):
    """Jensen–Shannon divergence (base 2): 0 = identical distributions, 1 = completely different."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = (p + q) / 2
    kl = lambda a, b: np.sum(np.where(a > 0, a * np.log2(a / b), 0))
    return float((kl(p, m) + kl(q, m)) / 2)


def main():
    train = pd.read_csv(DATA / "train.csv")
    val = pd.read_csv(DATA / "val.csv")
    CLIENTS.mkdir(parents=True, exist_ok=True)
    CLIENTS_IID.mkdir(parents=True, exist_ok=True)

    # ---- non-IID: one client per platform
    global_mix = train.label.value_counts(normalize=True).reindex(LABELS).values
    stats = []
    for p in PLATFORMS:
        tr, va = train[train.platform == p], val[val.platform == p]
        name = p.lower()
        tr.to_csv(CLIENTS / f"{name}_train.csv", index=False, encoding="utf-8-sig")
        va.to_csv(CLIENTS / f"{name}_val.csv", index=False, encoding="utf-8-sig")
        mix = tr.label.value_counts(normalize=True).reindex(LABELS).fillna(0)
        lang = tr.language.value_counts(normalize=True)
        stats.append(dict(client=p, train_rows=len(tr), val_rows=len(va), share_of_data=len(tr) / len(train),
                          **{f"pct_{l}": mix[l] for l in LABELS},
                          pct_hinglish_or_hindi=lang.reindex(["hinglish", "hi"]).fillna(0).sum(),
                          js_divergence_from_global=js_divergence(mix.values, global_mix)))

    # ---- IID: 5 equal random clients (same total data)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(train))
    vidx = rng.permutation(len(val))
    for k, (part, vpart) in enumerate(zip(np.array_split(idx, 5), np.array_split(vidx, 5)), start=1):
        train.iloc[part].to_csv(CLIENTS_IID / f"client{k}_train.csv", index=False, encoding="utf-8-sig")
        val.iloc[vpart].to_csv(CLIENTS_IID / f"client{k}_val.csv", index=False, encoding="utf-8-sig")
        mix = train.iloc[part].label.value_counts(normalize=True).reindex(LABELS).values
        stats.append(dict(client=f"IID client {k}", train_rows=len(part), val_rows=len(vpart), share_of_data=len(part) / len(train),
                          **{f"pct_{l}": v for l, v in zip(LABELS, mix)},
                          pct_hinglish_or_hindi=train.iloc[part].language.isin(["hinglish", "hi"]).mean(),
                          js_divergence_from_global=js_divergence(mix, global_mix)))

    table = pd.DataFrame(stats)
    out = ROOT / "results" / "tables"; out.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(out / "client_statistics.csv", index=False)
    summary = dict(non_iid_clients=PLATFORMS, iid_clients=5, train_rows=len(train), val_rows=len(val),
                   mean_js_non_iid=round(table.head(5).js_divergence_from_global.mean(), 4),
                   mean_js_iid=round(table.tail(5).js_divergence_from_global.mean(), 4))
    (ROOT / "reports" / "clients_summary.json").write_text(json.dumps(summary, indent=2))
    print(table.round(3).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
