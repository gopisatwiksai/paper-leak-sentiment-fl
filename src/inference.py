"""
Use the trained federated model on new comments.

    from inference import predict_sentiment
    predict_sentiment("Exam cancelled again, years of preparation wasted 😡")
    -> {'label': 'negative', 'confidence': 0.97, 'scores': {...}}
"""
from pathlib import Path

import numpy as np

from preprocess import clean_text, make_hashing

ROOT = Path(__file__).resolve().parents[1]
_MODEL, _HV = None, make_hashing()


def _load(path=ROOT / "models" / "fedavg_global_model.npz"):
    global _MODEL
    if _MODEL is None:
        m = np.load(path, allow_pickle=True)
        _MODEL = (m["W"].astype(np.float64), m["b"].astype(np.float64), [str(l) for l in m["labels"]])
    return _MODEL


def predict_sentiment(texts):
    """Accepts one string or a list of strings."""
    single = isinstance(texts, str)
    texts = [texts] if single else list(texts)
    W, b, labels = _load()
    X = _HV.transform([clean_text(t) for t in texts])
    Z = X @ W + b
    Z -= Z.max(axis=1, keepdims=True)
    P = np.exp(Z); P /= P.sum(axis=1, keepdims=True)
    out = [dict(text=t, label=labels[p.argmax()], confidence=round(float(p.max()), 3),
                scores={l: round(float(v), 3) for l, v in zip(labels, p)}) for t, p in zip(texts, P)]
    return out[0] if single else out


if __name__ == "__main__":
    for r in predict_sentiment(["NEET paper leaked again, years of preparation wasted 😡",
                                "Good decision to cancel the exam, a fair retest is better 👏",
                                "When will the new exam date be announced?"]):
        print(f"{r['label']:8s} {r['confidence']:.2f}  {r['text']}")
