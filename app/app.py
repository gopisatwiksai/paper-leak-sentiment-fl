"""
Paper Leak Sentiment Analysis Using Federated Learning – demo app (Step 9).

Run from the project folder:
    streamlit run app/app.py
"""
import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from preprocess import clean_text, make_hashing  # noqa: E402

SENT_COLORS = {"negative": "#e34948", "neutral": "#898781", "positive": "#2a78d6"}
ORDER = ["negative", "neutral", "positive"]
PLATFORMS = ["YouTube", "X", "Instagram", "Reddit", "NewsPortal"]
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"

st.set_page_config(page_title="Paper Leak Sentiment – Federated Learning", page_icon="📝", layout="wide")


# ---------------------------------------------------------------------------
# Loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    models = {}
    for key, file in [("FedAvg (federated)", "fedavg_global_model.npz"), ("Centralized", "centralized_softmax_model.npz")]:
        m = np.load(ROOT / "models" / file, allow_pickle=True)
        models[key] = (m["W"].astype(np.float64), m["b"].astype(np.float64), [str(l) for l in m["labels"]])
    return models, make_hashing()


@st.cache_data
def load_data():
    comments = pd.read_csv(ROOT / "data/processed/paper_leak_comments.csv")
    incidents = pd.read_csv(ROOT / "data/raw/paper_leaks.csv")
    return comments, incidents


def predict(texts, model_key):
    models, hv = load_models()
    W, b, labels = models[model_key]
    cleaned = [clean_text(t) for t in texts]
    Z = hv.transform(cleaned) @ W + b
    Z -= Z.max(axis=1, keepdims=True)
    P = np.exp(Z)
    P /= P.sum(axis=1, keepdims=True)
    return P, labels, cleaned


def prob_chart(probs, labels):
    df = pd.DataFrame({"sentiment": labels, "probability": probs})
    return (alt.Chart(df).mark_bar(cornerRadiusEnd=4, height=22)
            .encode(x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", title=None)),
                    y=alt.Y("sentiment:N", sort=ORDER, title=None),
                    color=alt.Color("sentiment:N", scale=alt.Scale(domain=ORDER, range=list(SENT_COLORS.values())), legend=None),
                    tooltip=["sentiment", alt.Tooltip("probability:Q", format=".1%")])
            .properties(height=120))


def share_chart(df, group, title):
    """100% stacked bars of sentiment per group."""
    agg = df.groupby([group, "sentiment"]).size().reset_index(name="comments")
    return (alt.Chart(agg).mark_bar()
            .encode(y=alt.Y(f"{group}:N", title=None, sort=PLATFORMS if group == "platform" else None),
                    x=alt.X("comments:Q", stack="normalize", axis=alt.Axis(format="%", title=None)),
                    color=alt.Color("sentiment:N", scale=alt.Scale(domain=ORDER, range=list(SENT_COLORS.values())),
                                    legend=alt.Legend(orient="bottom", title=None)),
                    order=alt.Order("sentiment:N", sort="ascending"),
                    tooltip=[group, "sentiment", "comments"])
            .properties(title=title, height=200))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("📝 Paper Leak Sentiment")
st.sidebar.caption("Sentiment analysis of public reactions to exam paper leaks, trained with **Federated Learning**.")
page = st.sidebar.radio("Go to", ["🔍 Analyze a comment", "📚 Incident explorer", "📊 Results", "⚙️ How it works"])
st.sidebar.divider()
st.sidebar.markdown("**Model:** softmax regression on hashing features, trained with FedAvg across 5 platform clients "
                    "(YouTube, X, Instagram, Reddit, NewsPortal). No raw comment ever left a client.")
st.sidebar.caption("Dataset: synthetic comments grounded in 110 real, reported paper-leak incidents (2004–2026).")

# ---------------------------------------------------------------------------
# Page 1 – analyze a comment
# ---------------------------------------------------------------------------
if page.startswith("🔍"):
    st.title("Analyze a comment")
    st.write("Type a public comment about a paper leak, in English, Hinglish or Hindi, and the federated model will classify it.")
    examples = {
        "– pick an example –": "",
        "Anger (English)": "NEET paper leaked again, years of preparation wasted 😡",
        "Relief (English)": "Good decision to cancel the exam, a fair retest is better than an unfair result 👏",
        "Question (English)": "When will the new exam date be announced? Is the old admit card valid?",
        "Anger (Hinglish)": "bhai phir se paper leak, kya fayda padhai ka",
        "Praise (Hinglish)": "police ne badhiya kaam kiya, gang pakda gaya 👏",
        "Question (Hinglish)": "retest ki date kab aayegi koi batao?",
        "Praise (Hindi)": "पुलिस ने अच्छा काम किया, दोषियों को पकड़ना सराहनीय है",
        "Sarcasm (hard case)": "Great job NTA, another leak 👏",
    }
    choice = st.selectbox("Examples", list(examples))
    text = st.text_area("Comment", value=examples[choice], height=100, placeholder="e.g. Exam cancelled again, no one is accountable")
    if st.button("Analyze", type="primary") and text.strip():
        P, labels, cleaned = predict([text], "FedAvg (federated)")
        p = P[0]
        label = labels[int(p.argmax())]
        emoji = {"negative": "😠", "neutral": "😐", "positive": "🙂"}[label]
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f"### {emoji} {label.capitalize()}")
            st.metric("Confidence", f"{p.max():.0%}")
        with c2:
            st.altair_chart(prob_chart(p, labels), width="stretch")
        with st.expander("What the model actually sees (after preprocessing)"):
            st.code(cleaned[0] or "(empty)")
            st.caption("Emojis become words (😡 → emo_angry), ? becomes qmark, slang is normalised, negations are kept.")
        Pc, _, _ = predict([text], "Centralized")
        lc = labels[int(Pc[0].argmax())]
        agree = "✅ agrees" if lc == label else "⚠️ disagrees"
        st.caption(f"Centralized model (trained on pooled data) predicts **{lc}** ({Pc[0].max():.0%}), so it {agree} with the federated model.")
        if choice.startswith("Sarcasm"):
            st.info("Sarcasm is a known limitation: a word-based model sees 'great job' and 👏 and may call it positive.")

# ---------------------------------------------------------------------------
# Page 2 – incident explorer
# ---------------------------------------------------------------------------
elif page.startswith("📚"):
    st.title("Incident explorer")
    comments, incidents = load_data()
    counts = comments.groupby("incident_id").size()
    incidents = incidents.assign(n_comments=incidents.incident_id.map(counts).fillna(0).astype(int))
    incidents["option"] = incidents.apply(lambda r: f"{r.date[:4]} · {r.exam_name[:70]}  ({r.n_comments:,} comments)", axis=1)
    default = int(incidents.n_comments.idxmax())
    opt = st.selectbox("Choose one of the 110 real incidents", incidents.option.tolist(), index=default)
    inc = incidents[incidents.option == opt].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Date", inc.date)
    c2.metric("Where", inc.area.split("(")[0].strip())
    c3.metric("Leak status", inc.leak_status)
    c4.metric("Arrests reported", "—" if pd.isna(inc.arrests) else f"{int(inc.arrests):,}")
    st.markdown(f"**Action taken:** {inc.action_taken}  \n**Summary:** {inc.note}")
    if isinstance(inc.source_url, str) and inc.source_url.startswith("http"):
        st.caption(f"Source: [{inc.source_name}]({inc.source_url}) · confidence {inc.confidence}")

    sub = comments[comments.incident_id == inc.incident_id].copy()
    P, labels, _ = predict(sub.text.tolist(), "FedAvg (federated)")
    sub["sentiment"] = np.array(labels)[P.argmax(axis=1)]
    st.subheader("Public sentiment predicted by the federated model")
    m1, m2, m3 = st.columns(3)
    for col, s in zip((m1, m2, m3), ORDER):
        col.metric(s.capitalize(), f"{(sub.sentiment == s).mean():.0%}")
    st.altair_chart(share_chart(sub, "platform", "Sentiment by platform"), width="stretch")
    st.subheader("Sample comments")
    show = sub.sample(min(12, len(sub)), random_state=1)[["platform", "language", "text", "sentiment"]]
    st.dataframe(show, hide_index=True, width="stretch")
    st.caption("Comments are synthetic but grounded in this incident's real facts.")

# ---------------------------------------------------------------------------
# Page 3 – results
# ---------------------------------------------------------------------------
elif page.startswith("📊"):
    st.title("Results")
    main = pd.read_csv(TAB / "comparison_main.csv").set_index("method")
    fed = main.loc["FedAvg – platform clients (non-IID)", "accuracy_mean"]
    cen = main.loc["Centralized (pooled data)", "accuracy_mean"]
    loc = main.loc["Local only – average of 5 platforms", "accuracy_mean"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Federated (FedAvg)", f"{fed:.2%}", help="5 platform clients, no comments shared")
    c2.metric("Centralized", f"{cen:.2%}", delta=f"{(cen - fed) * 100:+.2f} pts vs FedAvg", delta_color="off")
    c3.metric("Local only (avg)", f"{loc:.2%}", delta=f"{(loc - fed) * 100:+.2f} pts vs FedAvg", delta_color="off")
    c4.metric("Raw comments shared", "0")
    final = pd.read_csv(TAB / "final_comparison_table.csv")
    st.dataframe(final.style.format({c: "{:.2%}" for c in ["Accuracy", "Precision", "Recall", "Macro-F1"]}),
                 hide_index=True, width="stretch")
    tabs = st.tabs(["Main comparison", "Cross-platform", "By language", "Convergence", "Privacy noise"])
    for tab, (img, cap) in zip(tabs, [
            ("G1_main_comparison.png", "FedAvg nearly matches centralized training and beats every platform training alone (mean of 3 runs)."),
            ("G2_cross_platform.png", "Local models drop on unfamiliar platforms; the FedAvg model is strong everywhere."),
            ("G3_accuracy_by_language.png", "The English-only NewsPortal client gains most on Hinglish (+7.6 points) by collaborating."),
            ("F2_fedavg_convergence.png", "FedAvg passes 90% at round 4 and ends 0.35 points below the same model trained centrally."),
            ("G5_privacy_noise_tradeoff.png", "Small noise on updates is free; strong noise costs accuracy with only 5 clients.")]):
        with tab:
            if (FIG / img).exists():
                st.image(str(FIG / img), width="stretch")
            st.caption(cap)

# ---------------------------------------------------------------------------
# Page 4 – how it works
# ---------------------------------------------------------------------------
else:
    st.title("How it works")
    if (FIG / "F1_fl_architecture.png").exists():
        st.image(str(FIG / "F1_fl_architecture.png"), width="stretch")
    st.markdown("""
**Federated Averaging (FedAvg), repeated for 40 rounds**
1. The server sends the current global model to the 5 platform clients.
2. Each client cleans and hashes **its own** comments and trains the model for 2 local epochs.
3. Each client sends back **only the updated weights** (3.15 MB of numbers, no text).
4. The server averages the weights, giving larger clients more weight: **W = Σ (nₖ / n) · Wₖ**.

**Why hashing features?** A normal TF-IDF vocabulary has to be built from everyone's text, which would break privacy.
The hashing vectorizer maps words to columns with a fixed function, so every client builds identical features on its own.

**Pipeline:** data → preprocessing (emojis → words, slang normalisation, negations kept, Hindi kept intact)
→ train / validation / test split (80 / 10 / 10) → 5 platform clients → FedAvg → evaluation.
""")
    st.info("**About the data:** no public labeled dataset on paper-leak sentiment exists, so the 1,00,002 comments are synthetic. "
            "They are grounded in 110 real, reported incidents (2004–2026) and include simulated annotator disagreement "
            "(7.6% of labels). See `reports/DATASET_CARD.md`.")
