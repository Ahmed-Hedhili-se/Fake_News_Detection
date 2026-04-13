# Fake News Detection — From Research to Applied AI

> A complete three-approach NLP project: **Machine Learning → Deep Learning → Transformers**  
> Grounded in two peer-reviewed papers (2025) and implemented as production-ready Kaggle notebooks.

---

## Project Overview

This project implements and benchmarks three generations of fake news detection models on the [ISOT Fake News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset), following the exact taxonomy established by **Tian et al. (2025)** — *ML → DL → Transformers* — and replicating the hybrid BERT architecture studied by **Lilhore et al. (2025)**.

| Task | Approach | Best Model | F1-Score |
|------|----------|------------|----------|
| Task 1 | Classical ML | Linear SVM + TF-IDF | 99.10% |
| Task 2 | Deep Learning | Bi-GRU (PyTorch) | 98.10% |
| Task 3 | Transformer | BERT fine-tuned | **99.98%** |

---

## Repository Structure

```
fake-news-detection/
│
├── notebooks/
│   ├── fake_news_detection_ML.ipynb        # Task 1 — Machine Learning
│   ├── fake_news_detection_DL.ipynb        # Task 2 — Deep Learning (RNNs)
│   └── fake_news_detection_BERT.ipynb      # Task 3 — BERT Transformer
│
├── README.md                               # This file
├── project_explanation.md                  # Full technical write-up
└── requirements.txt                        # Python dependencies
```

---

## Research Foundation

### Paper 1 — Lilhore et al. (2025)
**"Fake News Detection Using BERT and Bi-LSTM with Grid Search Hyperparameter Optimization"**
- Proposes a hybrid BERT + Bi-LSTM architecture
- Achieves 99.908% accuracy on Twitter Fake News Dataset
- Our implementation: fine-tuned `bert-base-uncased` with deep classification head

### Paper 2 — Tian et al. (2025)
**"An Empirical Comparison of Machine Learning and Deep Learning Models for Automated Fake News Detection"**
- Benchmarks ML vs DL vs Transformer across ISOT and LIAR datasets
- ALBERT achieves Macro F1 = 0.99 (best Transformer)
- Our project directly mirrors this three-generation structure

---

## Dataset

**Fake and Real News Dataset** by Clément Bisaillon  
- 44,898 articles total (21,417 REAL from Reuters / 23,481 FAKE)  
- Columns: `title`, `text`, `subject`, `date`, `label`  
- Download: [kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

---

## Quickstart

### Run on Kaggle (recommended)

1. Open each notebook directly on Kaggle
2. Add the dataset: *Notebook settings → Add data → search "fake-and-real-news-dataset"*
3. For Task 3: enable GPU — *Notebook settings → Accelerator → GPU T4 x2*
4. Run all cells top to bottom

### Run locally

```bash
git clone https://github.com/YOUR_USERNAME/fake-news-detection.git
cd fake-news-detection
pip install -r requirements.txt

# Download NLTK assets (Task 1 & 2)
python -c "import nltk; [nltk.download(p) for p in ['stopwords','wordnet','punkt','punkt_tab','omw-1.4']]"

# Launch notebooks
jupyter notebook notebooks/
```

> **Note:** Task 3 (BERT) requires a CUDA-capable GPU. On CPU, training will take several hours.  
> Set `DATA_DIR` to your local dataset path before running.

---

## Results

### Final Comparison — All Three Approaches

| Approach | Model | Accuracy | Precision | Recall | F1-Score |
|----------|-------|----------|-----------|--------|----------|
| ML | Naïve Bayes | 93.60% | 93.10% | 92.10% | 92.60% |
| ML | Logistic Regression | 98.70% | 98.80% | 98.50% | 98.60% |
| ML | Linear SVM | 99.10% | 99.20% | 99.00% | 99.10% |
| DL | Bi-RNN | 97.30% | 97.10% | 96.80% | 97.00% |
| DL | Bi-LSTM | 98.10% | 98.00% | 97.90% | 98.00% |
| DL | Bi-GRU | 98.30% | 98.20% | 98.10% | 98.10% |
| **Transformer** | **BERT** | **99.98%** | **100.00%** | **99.95%** | **99.98%** |

### BERT Training Summary
- Best checkpoint: **step 1,200** (mid epoch 1)
- Early stopping triggered: step 1,800
- Total training time: **9.4 minutes** (Tesla T4)
- Test errors: **1 out of 4,490** articles

---

## Key Design Decisions

### Task 1 — Machine Learning
- TF-IDF with unigrams + bigrams (`ngram_range=(1,2)`, `max_features=50,000`)
- `sublinear_tf=True` to dampen high-frequency terms
- Three models compared: Logistic Regression, Multinomial Naïve Bayes, Linear SVM

### Task 2 — Deep Learning
- Unified `FakeNewsRNN` class — one architecture, three variants (RNN / LSTM / GRU)
- Bidirectional, 2-layer stacked (`bidirectional=True`, `n_layers=2`)
- Vocabulary built on training data only (30,000 tokens)
- Custom `encode_and_pad()` with left-padding and right-truncation

### Task 3 — BERT Transformer
- `bert-base-uncased` fine-tuned end-to-end (110M parameters)
- Title + body combined as input — maximises signal
- Deep classification head: `LayerNorm → Dropout → Linear(768→256) → GELU → Dropout → Linear(256→1)`
- `BCEWithLogitsLoss` (autocast-safe) — Sigmoid fused into loss, not the model
- AdamW with decoupled weight decay (bias and LayerNorm excluded)
- Linear warmup (10%) + linear decay scheduler
- Mid-epoch validation every 200 steps + early stopping (patience=3)
- Mixed precision training (`torch.cuda.amp`) — 2× speed

---

## Limitations

- ISOT is a "clean" benchmark — real news is exclusively Reuters-style, fake news comes from known misinformation sites. All three approaches achieve near-ceiling performance because stylistic separation is strong.
- On harder, adversarial datasets (LIAR, FakeNewsNet, GossipCop), scores drop to 70–85% and the Transformer advantage becomes decisive.
- Models learn journalistic style, not reasoning — they may fail on adversarial fake news designed to mimic real reporting.

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Data manipulation | `pandas`, `numpy` |
| NLP preprocessing | `nltk` |
| ML models | `scikit-learn` |
| DL models | `torch` |
| Transformers | `transformers` (Hugging Face) |
| Visualisation | `matplotlib`, `seaborn` |
| Environment | Kaggle (GPU T4), Python 3.10+ |

---

## License

MIT License — free to use, modify, and distribute with attribution.
