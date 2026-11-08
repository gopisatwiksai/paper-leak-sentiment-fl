# Dataset Card: Paper Leak Public Comments (synthetic, incident-grounded)

**File:** `data/processed/paper_leak_comments.csv`  **Rows:** 100,002  **Classes:** 3 (exactly balanced)
**Snapshot date:** 6 Nov 2026  **Generator:** `src/generate_dataset.py` (seed 42, fully reproducible)

## 1. What this dataset is

This is a **realistic synthetic dataset**: 1 lakh labeled public-style comments about examination paper-leak incidents in India.

- Every comment is tied to **one of 110 real, sourced paper-leak incidents (2004–2026)** from `data/raw/paper_leaks.csv`.
- The comments were **generated, not collected**. No real user wrote them and no personal data is included.

**How to describe it in the report or review:**
> "A realistic synthetic dataset of 1,00,002 labeled comments, grounded in 110 real, documented paper-leak incidents (2004–2026), with simulated annotator disagreement on ambiguous comments."

Do not describe it as collected or scraped data.

## 2. Label definitions

| Label | Meaning in the paper-leak context | Count |
|---|---|---|
| **Positive** | Support for action taken (arrests, cancellation, retest, probe), relief, encouragement, praise for whistleblowers | 33,334 |
| **Negative** | Anger, frustration, distrust, hopelessness, sarcasm aimed at the system or the conducting body | 33,334 |
| **Neutral** | Questions, factual updates, sharing of information, undecided or reported opinions | 33,334 |

## 3. Columns

| Column | Description |
|---|---|
| `comment_id` | Unique ID (C000001 …) |
| `incident_id` | Links to `paper_leaks.csv` (PL-0001 … PL-0110) |
| `exam_name`, `exam_category`, `state`, `incident_year` | Facts about the incident (`exam_category`: recruit / entrance / board) |
| `platform` | YouTube, X, Instagram, Reddit, NewsPortal. **These are the 5 federated-learning clients.** |
| `posted_at` | Simulated comment time (after the incident, never after the snapshot date) |
| `user_type` | aspirant, parent, teacher, coaching, general |
| `language` | `en`, `hinglish` (Hindi in Roman script), `hi` (Devanagari), `te-en` (Telugu-English) |
| `text` | The comment |
| `label` | positive, negative or neutral (the label used for training and testing) |
| `likes` | Simulated like count |

## 4. Key statistics

| Metric | Value |
|---|---|
| Incidents covered | 110 of 110 (135 to 4,941 comments each; bigger and more recent exams get more) |
| Most-discussed incidents | NEET-UG 2026, NEET-UG 2024, UP Police Constable 2024, UGC-NET 2024 |
| Platforms | X 26.1k · YouTube 22.0k · Instagram 19.4k · Reddit 16.7k · NewsPortal 15.9k |
| Languages | English 70.4% · Hinglish 27.7% · Hindi (Devanagari) 1.7% · Telugu-English 0.2% |
| Exam categories | Recruitment 75.2% · Entrance 19.3% · Board 5.5% |
| Average length | 12.2 words |
| Exact duplicate texts | 0 |
| Fact mismatches (automatic check) | 0 |

**Per-platform differences (non-IID), designed to resemble real platforms; useful for federated learning:**

| Platform | Negative | Neutral | Positive | Hinglish share |
|---|---|---|---|---|
| YouTube | 41% | 24% | 35% | 54% |
| X | 37% | 32% | 31% | 26% |
| Instagram | 35% | 21% | 44% | 40% |
| Reddit | 28% | 45% | 27% | 5% |
| NewsPortal | 20% | 52% | 28% | 3% |

These differences are **design assumptions**, not measurements of real platforms.

## 5. How it was generated

1. **Incident facts.** For each incident, the generator reads which facts are true: cancelled, retest, arrests (and the exact count), probe and agency, Confirmed/Alleged/Denied status, aspirants affected, state, and whether it came after the 2024 anti-paper-leak law.
2. **Grounding.** A comment only mentions facts its incident actually has. The checker in `src/check_dataset.py` found 0 mismatches.
3. **Human-style text.** There are two hand-written sources:
   - `src/comment_bank.py`: about 450 full comments in English, Hinglish, Hindi and Telugu-English.
   - `src/grammar.py`: a phrase grammar of reactions, facts, reasons, questions and wishes.
   Fact phrases are shared by all three labels. The data includes sarcasm, mixed feelings, reported opinions and negation pairs ("I trust the CBI" vs. "I don't trust the CBI").
4. **Wording variety.** Synonyms and spelling variants are swapped in (students/aspirants/candidates, nahi/nhi, bahut/bohot …). Typos, txt-speak and platform styles are added at realistic rates.
5. **Balancing.** 38,000 comments are generated per class. After step 6, exactly 33,334 per final label are kept.

## 6. Simulated annotator disagreement (label noise)

In real sentiment datasets, human annotators do not always agree, especially on short, sarcastic or mixed comments. A dataset where every label is perfectly consistent is unrealistic, and models trained on it score near 100%.

To reflect this, each comment gets a probability that an annotator would label it differently:

| Comment type | Chance of a different label |
|---|---|
| Clear, strongly worded opinion ("Shameful!", "Well done 💪") | about 1.5% |
| Plain factual question or update (neutral) | 1–2% |
| Typical opinion comment | about 5% |
| Extra for very short comments (≤5 words) | +15% |
| Extra for mixed feelings ("but", "par", "though") | +15% |
| Extra for opinion phrased as a question | +12% |
| Extra for neutral comments that report others' feelings | +15% |
| Extra for possible sarcasm in negative comments | +20% (usually read as positive) |

The new label is a plausible neighbour (mostly opinion↔neutral). **Result: 7,603 comments (7.6%) carry the alternative label.** Every changed label is listed in `reports/label_noise_log.csv`, with the generated label and the final label, so the process is fully transparent.

## 7. Quality checks (`reports/quality_report.json`)

| Check | Result |
|---|---|
| Baseline TF-IDF + Logistic Regression on raw text | 90.4% accuracy (macro-F1 0.904) |
| Best centralized model after preprocessing (Step 5) | Linear SVM, 92.4% accuracy |
| Comments with a near-duplicate (cosine > 0.9) | 18.1% |
| Distinct 4-gram ratio | 0.20 |
| Vocabulary size | about 6,100 words |

## 8. Limitations (state these honestly in the report)

- **It is synthetic.** Its phrasing comes from hand-written patterns, so it is more regular than real social media. The difficulty comes mainly from ambiguous comments and simulated disagreement, not from the full messiness of real text.
- **Some fields are simulated.** Timestamps, likes, user types, platform/language shares and the noise rates are assumptions, not measurements.
- **Language coverage is uneven.** Hindi (Devanagari) and Telugu-English are small; Tamil, Bengali, Marathi and other languages are not covered.
- **Facts are only as good as the source.** Incident facts come from the source CSV, which is a manually compiled log of *reported* incidents.
- **Future work:** validate the model on a small set of real, hand-labeled comments (for example, 500 from Reddit or YouTube).

## 9. Source

Incident backbone: `paper_leaks.csv` from <https://github.com/sujaibanerji/paper-leaks-analysis> (110 incidents, each with a source URL and confidence rating). No license was found in that repository, so cite it as the source.
