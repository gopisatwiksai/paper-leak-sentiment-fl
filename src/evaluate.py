"""
Step 8 – Full comparison (Objective 3).

Experiments
  1. Main comparison (3 random seeds, mean ± std)
       Centralized   – same softmax model, all training data pooled (no privacy)
       FedAvg        – 5 platform clients, non-IID (our federated system)
       FedAvg IID    – 5 random equal clients (easy, artificial case)
       Local only    – each platform trains alone on its own data, no collaboration
  2. Cross-platform test: each local model (and the FedAvg model) scored on every platform's data
  3. Per-language accuracy: why collaboration helps the English-only clients
  4. FedProx: FedAvg plus a pull towards the global model (designed for non-IID data)
  5. Privacy noise: Gaussian noise added to the clients' updates (differential-privacy style)
     and its effect on accuracy

All models are scored on the same held-out TEST set (10,000 comments).
Runtime: about 10 minutes.   Run:  python src/evaluate.py
"""
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from federated import (FLConfig, LABELS, PLATFORMS, centralized_softmax, fedavg, load_clients,
                       load_test, local_only, predict)

ROOT = Path(__file__).resolve().parents[1]
TABLES, REPORTS = ROOT / "results" / "tables", ROOT / "reports"
SEEDS = [42, 7, 2026]


def metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return dict(accuracy=accuracy_score(y_true, y_pred), precision=p, recall=r, macro_f1=f)


def log(msg, t0):
    print(f"[{time.perf_counter() - t0:6.0f}s] {msg}", flush=True)


def privacy_experiment(clients, Xte, yte, base, t0):
    """Clip each client's update to norm 5, then add Gaussian noise (std = sigma per weight) on the server."""
    rows = []
    d = clients[0]["X"].shape[1] * 3 + 3
    for sigma in [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]:
        cfg = replace(base, dp_clip=5.0, dp_noise=sigma * len(clients) / 5.0)   # dp_noise is the multiplier; std = mult*clip/K
        W, b, _ = fedavg(clients, Xte, yte, cfg, verbose=False)
        rows.append(dict(noise_std_per_weight=sigma, noise_norm_per_round=sigma * np.sqrt(d),
                         **metrics(yte, predict(Xte, W, b))))
        log(f"privacy noise sigma={sigma} done", t0)
    pd.DataFrame(rows).to_csv(TABLES / "privacy_noise_tradeoff.csv", index=False)
    return rows


def main():
    t0 = time.perf_counter()
    TABLES.mkdir(parents=True, exist_ok=True)
    log("loading clients and test set", t0)
    clients = load_clients("platform")
    clients_iid = load_clients("iid")
    Xte, yte, test_df = load_test()
    base = FLConfig()

    # ------------------------------------------------------------ 1. main comparison, 3 seeds
    rows, curves, keep = [], [], {}
    for seed in SEEDS:
        cfg = replace(base, seed=seed)
        Wc, bc, hc = centralized_softmax(clients, Xte, yte, cfg)
        rows.append(dict(seed=seed, method="Centralized (pooled data)", **metrics(yte, predict(Xte, Wc, bc))))
        log(f"seed {seed}: centralized done", t0)

        Wf, bf, hf = fedavg(clients, Xte, yte, cfg, verbose=False)
        rows.append(dict(seed=seed, method="FedAvg – platform clients (non-IID)", **metrics(yte, predict(Xte, Wf, bf))))
        log(f"seed {seed}: FedAvg non-IID done", t0)

        Wi, bi, hi = fedavg(clients_iid, Xte, yte, cfg, verbose=False)
        rows.append(dict(seed=seed, method="FedAvg – random clients (IID)", **metrics(yte, predict(Xte, Wi, bi))))
        log(f"seed {seed}: FedAvg IID done", t0)

        local_models = {}
        for c in clients:
            Wl, bl, _ = local_only(c, Xte, yte, cfg)
            local_models[c["name"]] = (Wl, bl)
            rows.append(dict(seed=seed, method=f"Local only – {c['name']}", **metrics(yte, predict(Xte, Wl, bl))))
        log(f"seed {seed}: 5 local-only models done", t0)

        if seed == SEEDS[0]:
            keep = dict(central=(Wc, bc), fedavg=(Wf, bf), iid=(Wi, bi), local=local_models)
            for name, h in [("Centralized (pooled data)", hc), ("FedAvg – platform clients (non-IID)", hf),
                            ("FedAvg – random clients (IID)", hi)]:
                curves.append(h[["round", "test_accuracy"]].assign(method=name))

    runs = pd.DataFrame(rows)
    runs.to_csv(TABLES / "comparison_all_runs.csv", index=False)
    summary = runs.groupby("method", sort=False).agg(
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        precision=("precision", "mean"), recall=("recall", "mean"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std")).reset_index()
    local_rows = runs[runs.method.str.startswith("Local only")]
    avg_local = local_rows.groupby("seed")[["accuracy", "precision", "recall", "macro_f1"]].mean()
    summary.loc[len(summary)] = dict(method="Local only – average of 5 platforms",
                                     accuracy_mean=avg_local.accuracy.mean(), accuracy_std=avg_local.accuracy.std(),
                                     precision=avg_local.precision.mean(), recall=avg_local.recall.mean(),
                                     macro_f1_mean=avg_local.macro_f1.mean(), macro_f1_std=avg_local.macro_f1.std())
    summary.round(4).to_csv(TABLES / "comparison_main.csv", index=False)

    # ------------------------------------------------------------ 2. cross-platform matrix (seed 42 models)
    cross = {}
    for name, (W, b) in list(keep["local"].items()) + [("FedAvg (all platforms)", keep["fedavg"]),
                                                      ("Centralized", keep["central"])]:
        cross[name] = {c["name"]: accuracy_score(c["yval"], predict(c["Xval"], W, b)) for c in clients}
    cross = pd.DataFrame(cross).T[PLATFORMS]
    cross.index.name = "trained on"
    cross.round(4).to_csv(TABLES / "cross_platform_accuracy.csv")
    log("cross-platform matrix done", t0)

    # ------------------------------------------------------------ 3. per-language accuracy (seed 42 models)
    lang_rows = []
    named = {"Centralized": keep["central"], "FedAvg": keep["fedavg"],
             "Local only – YouTube": keep["local"]["YouTube"], "Local only – NewsPortal": keep["local"]["NewsPortal"]}
    for name, (W, b) in named.items():
        pred = predict(Xte, W, b)
        for lang, g in test_df.assign(pred=pred).groupby("language"):
            lang_rows.append(dict(model=name, language=lang, comments=len(g), accuracy=accuracy_score(g.label, g.pred)))
    pd.DataFrame(lang_rows).round(4).to_csv(TABLES / "accuracy_by_language.csv", index=False)

    # ------------------------------------------------------------ 4. FedProx (seed 42)
    prox_rows = [dict(mu=0.0, **metrics(yte, predict(Xte, *keep["fedavg"])))]
    for mu in [0.0003, 0.001, 0.003]:
        W, b, h = fedavg(clients, Xte, yte, replace(base, mu=mu), verbose=False)
        prox_rows.append(dict(mu=mu, **metrics(yte, predict(Xte, W, b))))
        curves.append(h[["round", "test_accuracy"]].assign(method=f"FedProx (mu={mu})"))
        log(f"FedProx mu={mu} done", t0)
    pd.DataFrame(prox_rows).round(4).to_csv(TABLES / "fedprox_results.csv", index=False)
    pd.concat(curves).to_csv(TABLES / "convergence_curves.csv", index=False)

    # ------------------------------------------------------------ 5. privacy noise (seed 42)
    dp_rows = privacy_experiment(clients, Xte, yte, base, t0)

    out = dict(seeds=SEEDS, main=summary.round(4).to_dict(orient="records"),
               fedprox=prox_rows, privacy_noise=dp_rows, runtime_seconds=round(time.perf_counter() - t0))
    (REPORTS / "comparison_summary.json").write_text(json.dumps(out, indent=2, default=float))
    print("\n" + summary.round(4).to_string(index=False))
    log("all experiments finished", t0)


if __name__ == "__main__":
    main()
