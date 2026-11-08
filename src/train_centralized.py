"""
Step 5 – Centralized baseline models.

"Centralized" = all training data is pooled in one place (no privacy).
These scores are the benchmark the federated model (Step 7) is compared against.

Models
  1. Multinomial Naive Bayes      – TF-IDF features
  2. Logistic Regression          – TF-IDF features
  3. Linear SVM                   – TF-IDF features
  4. SGD Logistic Regression      – Hashing features  <- the SAME model and features the
                                                          federated clients will train (FL reference)

Each model's main setting is tuned on the VALIDATION set; the TEST set is used once at the end.

Run:  python src/train_centralized.py
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from preprocess import make_hashing

ROOT = Path(__file__).resolve().parents[1]
DATA, MODELS, TABLES = ROOT / "data/processed", ROOT / "models", ROOT / "results/tables"
LABELS = ["negative", "neutral", "positive"]
SEED = 42

# Shared settings for the SGD model – Step 7 (federated) imports these so both use the same model
SGD_PARAMS = dict(loss="log_loss", alpha=1e-6, learning_rate="optimal", random_state=SEED)
SGD_EPOCHS = 5


def sgd_fit_epochs(X, y, epochs=SGD_EPOCHS, model=None, seed=SEED):
    """Train an SGD logistic regression for a fixed number of passes (same routine the FL clients use)."""
    rng = np.random.default_rng(seed)
    model = model or SGDClassifier(**SGD_PARAMS)
    for _ in range(epochs):
        idx = rng.permutation(X.shape[0])
        model.partial_fit(X[idx], y[idx], classes=np.array(LABELS))
    return model


def evaluate(name, features, model, X, y):
    pred = model.predict(X)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    per_class = f1_score(y, pred, average=None, labels=LABELS)
    return dict(model=name, features=features, accuracy=accuracy_score(y, pred), precision=p, recall=r, f1=f,
                **{f"f1_{c}": v for c, v in zip(LABELS, per_class)}), pred


def main():
    train = pd.read_csv(DATA / "train.csv")
    val = pd.read_csv(DATA / "val.csv")
    test = pd.read_csv(DATA / "test.csv")
    ytr, yva, yte = train.label.values, val.label.values, test.label.values

    tfidf = joblib.load(MODELS / "tfidf_vectorizer.joblib")
    Xtr, Xva, Xte = (tfidf.transform(d.clean_text) for d in (train, val, test))
    hv = make_hashing()
    Htr, Hva, Hte = (hv.transform(d.clean_text) for d in (train, val, test))

    candidates = {
        "Naive Bayes": ("TF-IDF", "alpha", [0.01, 0.05, 0.1, 0.5, 1.0],
                        lambda v: MultinomialNB(alpha=v), (Xtr, Xva, Xte)),
        "Logistic Regression": ("TF-IDF", "C", [0.5, 1, 2, 5, 10],
                                lambda v: LogisticRegression(C=v, max_iter=3000), (Xtr, Xva, Xte)),
        "Linear SVM": ("TF-IDF", "C", [0.05, 0.1, 0.5, 1.0],
                       lambda v: LinearSVC(C=v, random_state=SEED), (Xtr, Xva, Xte)),
        "SGD Logistic Regression (FL reference)": ("Hashing", "alpha", [1e-7, 1e-6, 1e-5, 1e-4],
                                                   lambda v: SGDClassifier(**{**SGD_PARAMS, "alpha": v}), (Htr, Hva, Hte)),
    }

    rows, tuning, preds, best_models = [], [], {}, {}
    for name, (feat, pname, grid, make, (A, B, C)) in candidates.items():
        scores = []
        for v in grid:
            m = make(v)
            m = sgd_fit_epochs(A, ytr, model=m) if feat == "Hashing" else m.fit(A, ytr)
            s = f1_score(yva, m.predict(B), average="macro")
            scores.append(s)
            tuning.append(dict(model=name, param=pname, value=v, val_macro_f1=round(s, 4)))
        best_v = grid[int(np.argmax(scores))]
        t0 = time.perf_counter()
        m = make(best_v)
        m = sgd_fit_epochs(A, ytr, model=m) if feat == "Hashing" else m.fit(A, ytr)
        train_sec = time.perf_counter() - t0
        res, pred = evaluate(name, feat, m, C, yte)
        res.update(best_param=f"{pname}={best_v}", val_macro_f1=max(scores), train_seconds=round(train_sec, 2))
        rows.append(res); preds[name] = pred; best_models[name] = m
        print(f"{name:42s} test acc {res['accuracy']:.4f}  macro-F1 {res['f1']:.4f}  ({pname}={best_v}, {train_sec:.1f}s)")

    results = pd.DataFrame(rows).sort_values("f1", ascending=False)
    TABLES.mkdir(parents=True, exist_ok=True)
    results.round(4).to_csv(TABLES / "centralized_results.csv", index=False)
    pd.DataFrame(tuning).to_csv(TABLES / "centralized_tuning.csv", index=False)

    # save every model + the predictions for the notebook / Step 8
    MODELS.mkdir(exist_ok=True)
    files = {"Naive Bayes": "centralized_naive_bayes.joblib", "Logistic Regression": "centralized_logreg.joblib",
             "Linear SVM": "centralized_linear_svm.joblib",
             "SGD Logistic Regression (FL reference)": "centralized_sgd_hashing.joblib"}
    for name, m in best_models.items():
        joblib.dump(m, MODELS / files[name])
    out = test[["comment_id", "platform", "language", "text", "label"]].copy()
    for name, p in preds.items():
        out[f"pred_{files[name].split('.')[0].replace('centralized_', '')}"] = p
    out.to_csv(TABLES / "centralized_test_predictions.csv", index=False, encoding="utf-8-sig")

    best = results.iloc[0]
    summary = dict(best_model=best.model, best_test_accuracy=round(best.accuracy, 4), best_test_macro_f1=round(best.f1, 4),
                   fl_reference_model="SGD Logistic Regression (FL reference)",
                   fl_reference_macro_f1=round(results.set_index("model").loc["SGD Logistic Regression (FL reference)", "f1"], 4),
                   sgd_params=SGD_PARAMS, sgd_epochs=SGD_EPOCHS,
                   best_confusion_matrix=confusion_matrix(yte, preds[best.model], labels=LABELS).tolist(),
                   best_classification_report=classification_report(yte, preds[best.model], labels=LABELS, output_dict=True))
    (ROOT / "reports/centralized_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nBest: {best.model}  acc {best.accuracy:.4f}  macro-F1 {best.f1:.4f}")


if __name__ == "__main__":
    main()
