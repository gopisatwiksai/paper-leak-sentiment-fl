# Paper Leak Sentiment Analysis Using Federated Learning

B.Tech (CSE – AI & ML) capstone project. Public comments about examination paper-leak incidents in India are classified as **positive, negative or neutral**, using a model trained with **Federated Learning (FedAvg)**: five social-media platforms train together while every platform keeps its own comments.

## Key results (test set of 10,000 comments, mean of 3 runs)

| Method | Comments shared | Accuracy | Macro-F1 |
|---|---|---|---|
| Centralized (all data pooled) | all | 92.25% | 0.9222 |
| **FedAvg – 5 platform clients (non-IID)** | **none** | **91.94%** | **0.9191** |
| FedAvg – 5 random clients (IID) | none | 92.00% | 0.9197 |
| Local only – average of 5 platforms | none | 90.50% | 0.9047 |
| Best centralized baseline (Linear SVM, TF-IDF) | all | 92.42% | 0.9239 |

- Privacy costs only **0.31 percentage points** – with zero comments shared.
- Every platform gains from collaboration: **+1.44 points** on average, **+7.6 points on Hinglish** for the news-portal client.

## How it works

1. **Data** – 110 real, documented paper-leak incidents (2004–2026) and 1,00,002 labelled comments grounded in them (English, Hinglish, Hindi, Telugu-English).
2. **Preprocessing** – emojis to words, Hinglish slang normalised, negations kept, Devanagari kept intact.
3. **Clients** – one client per platform: YouTube, X, Instagram, Reddit, NewsPortal (naturally non-IID).
4. **FedAvg** – implemented from scratch in NumPy: softmax regression on hashed features (no shared vocabulary), 40 rounds × 2 local epochs; the server averages the weights, weighted by client size.
5. **Evaluation** – centralized vs federated vs local-only, cross-platform, per language, FedProx, and a clipping + noise privacy experiment.

## Project structure

```
paper_leak_fl/
├── app/app.py                  Streamlit demo (4 pages)
├── data/
│   ├── raw/paper_leaks.csv     110 real incidents (source below)
│   ├── processed/              comments, train / val / test splits
│   ├── clients/                5 platform clients (non-IID)
│   └── clients_iid/            5 random clients (IID)
├── models/                     trained models (FedAvg global model, baselines)
├── notebooks/01–06             EDA, preprocessing, baselines, clients, FL, comparison
├── reports/                    dataset card, quality and result summaries
├── results/                    figures and tables
├── src/                        all project code (see below)
└── requirements.txt
```

| Script | Purpose |
|---|---|
| `src/generate_dataset.py` | Builds the comment dataset (fixed seed, reproducible) |
| `src/check_dataset.py` | Quality checks |
| `src/preprocess.py` | Cleaning, split, TF-IDF and hashing features |
| `src/train_centralized.py` | Naive Bayes, Logistic Regression, Linear SVM, SGD baselines |
| `src/split_clients.py` | Builds the federated clients |
| `src/federated.py` | FedAvg, FedProx, clipping + noise |
| `src/evaluate.py` | All comparison experiments (3 seeds) |
| `src/inference.py` | `predict_sentiment()` for new comments |

## Setup and run

```bash
py -m venv venv
venv\Scripts\activate            # Windows  (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt

python src/preprocess.py
python src/train_centralized.py
python src/split_clients.py
python src/federated.py
python src/evaluate.py           # about 13 minutes
streamlit run app/app.py         # demo at http://localhost:8501
```

The processed data and trained models are included, so the demo runs straight after installation.

## Dataset

- **Incidents (real):** `paper_leaks.csv` from [sujaibanerji/paper-leaks-analysis](https://github.com/sujaibanerji/paper-leaks-analysis) – 110 incidents, each with a source URL and confidence rating.
- **Comments (synthetic):** a realistic synthetic dataset of 1,00,002 labelled comments, grounded in the 110 real incidents, with simulated annotator disagreement on ambiguous comments (7.6%). No real user data is included. Details: [`reports/DATASET_CARD.md`](reports/DATASET_CARD.md).

## Team

- Gopi Satwik Sai – Federated Learning & System Integration
- Thanvi Priya – Dataset Collection & Preprocessing
- R. Harsha Vardhan Reddy – Sentiment Analysis Model
- M. V. Sai Praneeth – Application, Evaluation & Visualization

Guide: [Guide Name], Department of CSE (AI & ML), [College Name]
