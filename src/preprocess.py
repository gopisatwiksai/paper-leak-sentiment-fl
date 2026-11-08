"""
Step 4 – Preprocessing for Paper Leak Sentiment Analysis.

What it does
  1. Cleans every comment (clean_text): lowercase, remove links / mentions,
     turn emojis and ! ? into words, normalise slang (u -> you, pls -> please),
     keep hashtag words, shorten stretched letters (sooo -> soo), mask numbers.
  2. Keeps negations (not, no, never, nahi ...) – they flip sentiment.
  3. Splits the data ONCE into train / validation / test (80 / 10 / 10),
     stratified by label, with identical cleaned texts kept in the same split
     so nothing "leaks" from train into test.
  4. Builds the features:
       - TF-IDF (word 1–2 grams) for the centralized models (Step 5)
       - Hashing features for federated learning (Step 7): every client gets the
         same feature space without sharing any vocabulary or raw text.

Run:  python src/preprocess.py
"""
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
SEED = 42

# ---------------------------------------------------------------------------
# 1. Cleaning rules
# ---------------------------------------------------------------------------
EMOJI_WORDS = {
    "😡": "emo_angry", "🤬": "emo_angry", "😤": "emo_angry",
    "💔": "emo_sad", "😭": "emo_sad", "😞": "emo_sad", "😩": "emo_sad", "😢": "emo_sad",
    "🤦": "emo_facepalm", "👏": "emo_clap", "🙏": "emo_pray", "👍": "emo_thumbsup",
    "💪": "emo_strong", "❤": "emo_love", "🙌": "emo_celebrate", "🔥": "emo_fire",
    "🤔": "emo_think", "🙂": "emo_smile", "😅": "emo_relief", "😌": "emo_relief",
    "🫡": "emo_salute", "🤝": "emo_together",
}
SLANG = {
    "u": "you", "ur": "your", "r": "are", "pls": "please", "plz": "please", "ppl": "people",
    "coz": "because", "bcoz": "because", "rly": "really", "tho": "though", "thx": "thanks",
    "b4": "before", "smth": "something", "wat": "what", "k": "ok", "dont": "do not",
    "cant": "can not", "wont": "will not", "doesnt": "does not", "didnt": "did not",
    "isnt": "is not", "wasnt": "was not", "im": "i am", "ive": "i have", "thats": "that is",
    "its": "it is", "lets": "let us",
    # common Hinglish spelling variants
    "nhi": "nahi", "nai": "nahi", "ni": "nahi", "h": "hai", "hy": "hai", "kr": "kar", "rha": "raha",
}
NEGATIONS = {"not", "no", "never", "nor", "nothing", "none", "nobody", "nahi", "mat", "na"}
# Small stop-word list on purpose: sklearn's built-in English list removes "not", "no", "never"
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "for", "with", "is", "are", "was", "were",
    "be", "been", "it", "this", "that", "these", "those", "as", "by", "from", "so", "about", "there",
    "ka", "ki", "ke", "ko", "se", "me", "bhi", "hi", "ye", "wo", "aur",
} - NEGATIONS

URL = re.compile(r"https?://\S+|www\.\S+")
MENTION = re.compile(r"@\w+")
HASHTAG = re.compile(r"#(\w+)")
STRETCH = re.compile(r"([a-z])\1{2,}")                  # sooo -> soo
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
EMOJI_ANY = re.compile(r"[\U0001F300-\U0001FAFF☀-➿‍️]")


def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """Return a cleaned, space-separated version of one comment."""
    s = str(text)
    s = URL.sub(" ", s)
    s = MENTION.sub(" ", s)
    s = HASHTAG.sub(r" \1 ", s)                          # keep the hashtag word, drop '#'
    for emo, word in EMOJI_WORDS.items():               # emojis carry sentiment -> words
        s = s.replace(emo, f" {word} ")
    s = EMOJI_ANY.sub(" ", s)                            # any other emoji
    s = s.lower().replace("’", "'")
    s = re.sub(r"\bcan't\b", "can not", s)
    s = re.sub(r"\bwon't\b", "will not", s)
    s = re.sub(r"n't\b", " not", s)                     # don't -> do not, isn't -> is not
    s = re.sub(r"\bi'm\b", "i am", s)
    s = re.sub(r"\?+", " qmark ", s)                     # questions -> mostly neutral
    s = re.sub(r"!+", " exclaim ", s)                    # emphasis
    s = re.sub(r"\.{3,}|…", " ellipsis ", s)
    s = NUMBER.sub(" num ", s)
    s = STRETCH.sub(r"\1\1", s)
    s = s.replace("/s", " sarcasm_tag ")
    # keep letters of ANY script (Devanagari vowel signs included) plus _ and '
    s = re.sub(r"[^\wऀ-ॿ' ]+", " ", s)
    tokens = []
    for t in s.split():
        t = t.strip("'")
        t = SLANG.get(t, t)
        for w in t.split():
            if remove_stopwords and w in STOPWORDS:
                continue
            tokens.append(w)
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# 2. Feature extractors
# ---------------------------------------------------------------------------
TOKEN_PATTERN = r"(?u)\S+"          # text is already cleaned -> split on spaces (keeps Devanagari intact)


def make_tfidf(max_features=50_000):
    """TF-IDF features for the centralized baseline models."""
    return TfidfVectorizer(token_pattern=TOKEN_PATTERN, lowercase=False, ngram_range=(1, 2),
                           min_df=2, max_features=max_features, sublinear_tf=True)


def make_hashing(n_features=2 ** 18):
    """Hashing features for federated learning: no shared vocabulary needed, same space on every client."""
    return HashingVectorizer(token_pattern=TOKEN_PATTERN, lowercase=False, ngram_range=(1, 2),
                             n_features=n_features, alternate_sign=False, norm="l2")


# ---------------------------------------------------------------------------
# 3. Split once, save everything
# ---------------------------------------------------------------------------
def main():
    df = pd.read_csv(DATA / "paper_leak_comments.csv")
    df["clean_text"] = df.text.map(clean_text)
    df = df[df.clean_text.str.len() > 0].reset_index(drop=True)

    # identical cleaned texts must stay in the same split (otherwise test scores are inflated)
    groups = df.clean_text.factorize()[0]
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
    folds = np.zeros(len(df), dtype=int)
    for k, (_, idx) in enumerate(sgkf.split(df, df.label, groups)):
        folds[idx] = k
    df["split"] = np.select([folds == 0, folds == 1], ["test", "val"], "train")

    cols = ["comment_id", "incident_id", "platform", "language", "exam_category", "text", "clean_text", "label"]
    for name in ["train", "val", "test"]:
        df.loc[df.split == name, cols].to_csv(DATA / f"{name}.csv", index=False, encoding="utf-8-sig")

    # TF-IDF is fitted on TRAIN ONLY (no information from val/test)
    tfidf = make_tfidf().fit(df.loc[df.split == "train", "clean_text"])
    MODELS.mkdir(exist_ok=True)
    joblib.dump(tfidf, MODELS / "tfidf_vectorizer.joblib")

    tr, te = set(df.loc[df.split == "train", "clean_text"]), set(df.loc[df.split == "test", "clean_text"])
    stats = {
        "rows_after_cleaning": len(df),
        "split_sizes": df.split.value_counts().reindex(["train", "val", "test"]).to_dict(),
        "label_by_split": pd.crosstab(df.split, df.label).to_dict(orient="index"),
        "platform_by_split": pd.crosstab(df.split, df.platform).to_dict(orient="index"),
        "texts_shared_by_train_and_test": len(tr & te),
        "avg_tokens_before": round(df.text.str.split().str.len().mean(), 2),
        "avg_tokens_after": round(df.clean_text.str.split().str.len().mean(), 2),
        "tfidf_vocabulary_size": len(tfidf.vocabulary_),
        "hashing_feature_size": 2 ** 18,
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "preprocessing_report.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
