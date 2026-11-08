"""Quality checks for the generated dataset. Run: python src/check_dataset.py"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data/processed/paper_leak_comments.csv")
inc = pd.read_csv(ROOT / "data/raw/paper_leaks.csv").set_index("incident_id")
rep = {}

# 1. size, balance, uniqueness
rep["rows"] = len(df)
rep["label_counts"] = df.label.value_counts().to_dict()
rep["exact_duplicate_texts"] = int(df.text.duplicated().sum())
rep["incidents_covered"] = int(df.incident_id.nunique())
rep["avg_words"] = round(df.text.str.split().str.len().mean(), 1)

# 2. surface diversity
words = df.text.str.lower().str.findall(r"\w+")
ngr = [tuple(w[i:i + 4]) for w in words for i in range(len(w) - 3)]
rep["distinct_4gram_ratio"] = round(len(set(ngr)) / len(ngr), 3)
rep["vocabulary_size"] = len({t for w in words for t in w})

# 3. near-duplicates: share of comments whose closest neighbour is almost identical
samp = df.sample(5000, random_state=0)
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit(df.text)
X = vec.transform(samp.text)
nn = NearestNeighbors(n_neighbors=2, metric="cosine").fit(vec.transform(df.text))
dist, _ = nn.kneighbors(X)
sim = 1 - dist[:, 1]
rep["near_duplicate_share_cos>0.9"] = round(float((sim > 0.9).mean()), 3)
rep["median_nearest_neighbour_cosine"] = round(float(np.median(sim)), 3)

# 4. fact consistency: arrest numbers / retest / cancel mentions match the incident
def bad(row):
    a = inc.loc[row.incident_id]
    t = row.text.lower().replace("re-exam paper", "").replace("cgl re-exam", "").replace("re-neet", "")
    if re.search(r"\bretest|re-exam|re exam|dobara pariksha|दोबारा परीक्षा", t) and "retest" not in str(a.action_taken).lower():
        return True
    nums = re.findall(r"\b(\d+) (?:people|log|arrest|were arrested|arrested|giraftari|गिरफ्तार|mandi)", t)
    if nums and (pd.isna(a.arrests) or int(a.arrests) != int(nums[0])):
        return True
    return False
rep["fact_mismatches"] = int(df.apply(bad, axis=1).sum())

# 5. difficulty: a simple TF-IDF + Logistic Regression baseline
tr, te = train_test_split(df, test_size=0.2, stratify=df.label, random_state=42)
v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
clf = LogisticRegression(max_iter=2000, C=1.0).fit(v.fit_transform(tr.text), tr.label)
p = clf.predict(v.transform(te.text))
rep["baseline_tfidf_lr_accuracy"] = round(accuracy_score(te.label, p), 4)
rep["baseline_tfidf_lr_macro_f1"] = round(f1_score(te.label, p, average="macro"), 4)

# 6. per-platform label mix (non-IID check for federated clients)
rep["platform_label_mix"] = pd.crosstab(df.platform, df.label, normalize="index").round(3).to_dict(orient="index")
rep["language_counts"] = df.language.value_counts().to_dict()

print(json.dumps(rep, indent=2, ensure_ascii=False))
(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports/quality_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
