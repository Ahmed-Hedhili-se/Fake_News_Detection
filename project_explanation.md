# Project Explanation — Fake News Detection: Research to Applied AI

> **Orbyx Project** | February 2026  
> A complete NLP pipeline from academic research to working code, across three generations of models.

---

## 1. Project Context & Motivation

Misinformation spreads faster than corrections on social media. Automated fake news detection is one of the most active NLP research areas, with applications in content moderation, journalism, and platform governance.

This project follows a deliberate research-to-code pipeline:

1. Read two real 2025 papers
2. Understand the terminology and models they describe
3. Implement each approach from scratch
4. Compare results to the published benchmarks

The three-notebook structure directly mirrors the taxonomy established by **Tian et al. (2025)**, who compared ML, DL, and Transformer approaches on the same class of dataset.

---

## 2. Research Papers

### Paper 1 — Lilhore et al. (January 2025)

**Title:** *Fake News Detection Using BERT and Bi-LSTM with Grid Search Hyperparameter Optimization*

**Core idea:** Combine a pretrained Transformer (BERT) for contextual word embeddings with a Bidirectional LSTM for sequential classification. Use Grid Search to find optimal hyperparameters.

**What we implemented from this paper:**
- `bert-base-uncased` as the encoder backbone (same model family)
- Bidirectional design (our BERT head + the Bi-LSTM in Task 2)
- Accuracy / Precision / Recall / F1 as evaluation metrics (exact same set)
- Classification head after BERT's `[CLS]` token

**Where we diverged (and why):**
- We do not stack a Bi-LSTM on top of BERT. Modern practice shows the `[CLS]` token already captures bidirectional context from all 12 attention layers — an additional Bi-LSTM adds parameters without a meaningful accuracy gain on clean datasets like ISOT.
- Instead of Grid Search, we use AdamW + linear warmup + `ReduceLROnPlateau` — a more principled hyperparameter schedule.

**Their result:** 99.908% accuracy on Twitter Fake News Dataset  
**Our result:** 99.98% F1 on ISOT (different dataset, comparable difficulty level)

---

### Paper 2 — Tian et al. (June 2025)

**Title:** *An Empirical Comparison of Machine Learning and Deep Learning Models for Automated Fake News Detection*

**Core idea:** Benchmark three generations of models — Logistic Regression / Random Forest (ML), RNNs / CNNs (DL), and BERT / ALBERT (Transformer) — on standard datasets (ISOT, LIAR).

**What we implemented from this paper:**
- The exact same three-generation structure: Task 1 (ML) → Task 2 (DL) → Task 3 (Transformer)
- Same model families: Logistic Regression, Naïve Bayes (ML); RNN, LSTM, GRU (DL); BERT (Transformer)
- Same dataset family: ISOT-derived data
- Final comparison table that mirrors their benchmark structure

**Their result:** ALBERT Macro F1 = 0.99 (best Transformer)  
**Our result:** BERT F1 = 99.98% (consistent with their finding that Transformers dominate)

---

## 3. Dataset Description

**Name:** Fake and Real News Dataset (Bisaillon, Kaggle)  
**Size:** 44,898 articles  
**Classes:** REAL (21,417) from Reuters, FAKE (23,481) from known misinformation sites  
**Balance:** 47.7% REAL / 52.3% FAKE — nearly balanced, so Accuracy is a reliable metric  
**Split used:** 80% train / 10% val / 10% test (stratified)

**Why this dataset is easy:**  
ISOT has strong stylistic separation — Reuters articles have a recognisable formal structure, while fake articles use sensationalist language. Even TF-IDF + SVM reaches 99.1%. This makes it ideal for learning and comparison, but not for testing adversarial robustness.

---

## 4. Task 1 — Machine Learning Approach

### Pipeline

```
Raw text
  → Lowercase
  → Remove URLs, HTML, punctuation, digits (regex)
  → Tokenize (NLTK word_tokenize)
  → Remove stopwords
  → Lemmatize (WordNetLemmatizer)
  → TF-IDF vectorization (50k features, unigrams + bigrams)
  → Classifier
```

### Why TF-IDF?

TF-IDF (Term Frequency — Inverse Document Frequency) converts words into numbers by weighting how important a word is to a specific document relative to the whole corpus. A word like "reuters" that appears in every real news article but almost never in fake articles gets a high discriminative weight.

$$\text{TF-IDF}(t, d) = \frac{f_{t,d}}{\sum_k f_{k,d}} \times \log\frac{N}{|{d : t \in d}|}$$

### Models Compared

| Model | Core Mechanism | Strength |
|-------|---------------|----------|
| Logistic Regression | Linear decision boundary in TF-IDF space | Interpretable coefficients |
| Naïve Bayes | Probabilistic word frequency model | Fastest training |
| Linear SVM | Maximum-margin hyperplane | Best for high-dim sparse features |

### Results (Task 1)

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Naïve Bayes | 93.60% | 92.60% |
| Logistic Regression | 98.70% | 98.60% |
| Linear SVM | **99.10%** | **99.10%** |

### Limitation
TF-IDF ignores word order and context. "not good" and "good" produce identical feature vectors — the negation is invisible. This is where DL and Transformers have a structural advantage.

---

## 5. Task 2 — Deep Learning Approach (RNNs)

### Pipeline

```
Raw text
  → Lowercase + clean (regex)
  → Tokenize (NLTK)
  → Remove stopwords
  → Build vocabulary from training data (30,000 tokens)
  → Encode: word → integer index (UNK for unseen words)
  → Pad / truncate to MAX_LEN
  → Embedding layer (128-dim, trainable)
  → Bidirectional RNN / LSTM / GRU
  → [CLS-equivalent: last hidden state]
  → Dropout → Linear(256→1) → Sigmoid
```

### Architecture

```
Input (batch × seq_len)
    ↓
Embedding(vocab=30k, dim=128, padding_idx=0)
    ↓
Bi-{RNN/LSTM/GRU} (hidden=128, layers=2, bidirectional=True)
    ↓
Concatenate [forward_last, backward_last]  →  (batch, 256)
    ↓
Dropout(0.5) → Linear(256→1) → Sigmoid
    ↓
P(REAL) ∈ [0, 1]
```

### Why bidirectional?

A unidirectional RNN reading "the story was not true" processes each word left-to-right and may weight "true" strongly before seeing "not". A bidirectional model reads the sequence in both directions simultaneously, so "not" and "true" are always processed together. This directly mirrors the Bi-LSTM design in Lilhore et al.

### Models Compared

| Model | Memory Mechanism | Vanishing Gradient |
|-------|-----------------|-------------------|
| Simple RNN | Single hidden state | Severe on long sequences |
| LSTM | Cell state + 3 gates (forget, input, output) | Largely solved |
| GRU | 2 gates (reset, update) | Largely solved, fewer params |

### Training Best Practices Applied
- Vocabulary built on training data only (no test leakage)
- Left-padding (recent context is more informative for classification)
- `ReduceLROnPlateau` scheduler
- Gradient clipping (`max_norm=1.0`) — critical for Simple RNN
- Best checkpoint restore

### Results (Task 2)

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Bi-RNN | 97.30% | 97.00% |
| Bi-LSTM | 98.10% | 98.00% |
| Bi-GRU | **98.30%** | **98.10%** |

### Why GRU > LSTM here?
GRU has fewer parameters (2 gates vs 3) which reduces overfitting on a dataset where the signal is already very strong. On longer, noisier sequences, LSTM's cell state would give it an edge.

---

## 6. Task 3 — Transformer Approach (BERT)

### Why BERT is fundamentally different

| Approach | Context window | Word order | Pretrained knowledge |
|----------|---------------|------------|---------------------|
| TF-IDF | None (bag of words) | Ignored | None |
| Bi-GRU | Sequential | Left-to-right + right-to-left | None |
| BERT | Full sequence simultaneously | Full bidirectional self-attention | 3.3B words of Wikipedia + BooksCorpus |

BERT reads every word in context of every other word at the same time, via self-attention. This means "not true" and "not false" produce different representations — something neither TF-IDF nor RNNs can do properly.

### Architecture

```
Input: "TITLE BODY" (title + body concatenated, max 256 BERT tokens)
    ↓
WordPiece tokenizer → [CLS] token1 token2 ... [SEP] [PAD]...
    ↓
BERT Encoder (12 transformer layers, 768-dim hidden)
    ↓
[CLS] hidden state (position 0)  →  (batch, 768)
    ↓
LayerNorm(768)
    ↓
Dropout(0.3) → Linear(768→256) → GELU
    ↓
Dropout(0.3) → Linear(256→1)
    ↓
BCEWithLogitsLoss (training) / Sigmoid (inference)
    ↓
P(REAL) ∈ [0, 1]
```

### Why title + body combined?
Most implementations use only the article body. The title carries the strongest stylistic signal — fake news headlines are deliberately sensationalist ("BREAKING:", all-caps, exclamation marks). Feeding both to BERT with a single `[SEP]` boundary lets the model learn to weight title and body jointly.

### Training Recipe

| Hyperparameter | Value | Source |
|---------------|-------|--------|
| Learning rate | 2e-5 | Devlin et al. (2019) BERT paper |
| Optimizer | AdamW | Standard for fine-tuning |
| Weight decay | 0.01 (not on bias/LayerNorm) | Prevents catastrophic forgetting |
| Warmup | 10% of total steps | Gradual LR ramp to protect pretrained weights |
| Scheduler | Linear warmup + linear decay | Standard BERT fine-tuning |
| Gradient clip | 1.0 | Prevents explosive updates |
| Mixed precision | `torch.cuda.amp` | 2× speed, 50% VRAM |
| Epochs | 2 (early stopping at step 1,800) | ISOT memorises in <300 steps |
| Eval frequency | Every 200 steps | Mid-epoch visibility |
| Early stopping | Patience = 3 evals | Auto-halt if val_loss rises |

### Key Engineering Decisions

**`BCEWithLogitsLoss` instead of `BCELoss + Sigmoid`**  
PyTorch's `BCELoss` is unsafe inside `torch.cuda.amp.autocast` — float16 precision causes numerical underflow in the sigmoid output. `BCEWithLogitsLoss` fuses the sigmoid and the cross-entropy in a single numerically-stable operation that works correctly under mixed precision.

**Logit threshold at 0, not 0.5**  
Since the model outputs raw logits (not probabilities), the decision boundary is at `logit = 0` (which corresponds to `sigmoid(0) = 0.5`). Applying `>= 0.5` to raw logits would produce wrong predictions for the majority of samples.

**Separate weight decay groups**  
BERT's bias parameters and LayerNorm weights should not be regularised — they are scale/shift parameters, not weights representing learned knowledge. Applying weight decay to them hurts fine-tuning performance. We explicitly exclude them from the AdamW weight decay group.

### Results (Task 3)

| Metric | Score |
|--------|-------|
| Accuracy | 99.98% |
| Precision | 100.00% |
| Recall | 99.95% |
| F1-Score | 99.98% |
| Test errors | 1 / 4,490 |
| Best checkpoint | Step 1,200 |
| Training time | 9.4 min (Tesla T4) |

---

## 7. Three-Approach Comparison

### Why the progression ML → DL → Transformer?

| Limitation removed | By |
|---|---|
| No word order awareness | DL (RNNs process sequences) |
| No long-range dependencies | LSTM / GRU gates |
| Embeddings trained from scratch | BERT (pretrained on 3.3B words) |
| Context limited to one direction | BERT (full bidirectional attention) |
| No world knowledge | BERT (encyclopaedic pretraining) |

### Final Results Table

| Approach | Best Model | F1-Score | Gap to BERT |
|----------|------------|----------|-------------|
| ML | Linear SVM | 99.10% | -0.88% |
| DL | Bi-GRU | 98.10% | -1.88% |
| **Transformer** | **BERT** | **99.98%** | — |

On ISOT, the gap between approaches is small because the dataset is stylistically clean. On harder benchmarks (LIAR: 6 classes, short sentences; FakeNewsNet: social context required), BERT's advantage grows to 10–20% over ML and 5–10% over DL.

---

## 8. Pipeline Integrity Audit

| Check | Status | Evidence |
|-------|--------|---------|
| No data leakage (TF-IDF) | Pass | `tfidf.fit_transform(X_train)` only, `tfidf.transform(X_test)` |
| No data leakage (vocab) | Pass | `Counter` built on `X_train_raw` only |
| No data leakage (BERT tokenizer) | Pass | Pretrained tokenizer — no fit step |
| Stratified splits | Pass | `stratify=y` used in all three tasks |
| Best checkpoint restored | Pass | `model.load_state_dict(torch.load(CKPT_PATH))` before test eval |
| No overfitting | Pass | val_loss range 0.0025–0.0061, no divergence |
| Correct loss function | Pass | `BCEWithLogitsLoss`, Sigmoid only at inference |
| Correct decision boundary | Pass | `logit >= 0` during training, `sigmoid(logit) >= 0.5` at inference |

---

## 9. Limitations & Future Work

### Current limitations

- **Dataset ceiling effect:** ISOT is too clean for discriminating between approaches. All models cluster near 99%. Use LIAR or GossipCop for a more informative benchmark.
- **Style, not reasoning:** All models detect journalistic style, not logical consistency. Adversarial fake news that mimics Reuters style would fool all three.
- **English only:** All models are English-language only. Cross-lingual fake news detection requires multilingual BERT (mBERT) or XLM-RoBERTa.
- **Static model:** Articles from after the training cutoff may use new language patterns or reference new events that shift the decision boundary.

### Potential improvements

| Improvement | Expected gain | Effort |
|---|---|---|
| Use `roberta-base` instead of `bert-base-uncased` | +0.1–0.3% on ISOT; larger on harder datasets | Low |
| MAX_LEN = 512 instead of 256 | +0.1–0.2% (captures full article) | Medium (needs more VRAM) |
| Pretrained GloVe/FastText embeddings (Task 2) | +1–3% on harder datasets | Low |
| Test on LIAR dataset | Real benchmark; scores drop to 70–75% | Medium |
| Ensemble (BERT + SVM) | +0.1–0.5% on adversarial data | Medium |
| Domain-adapted BERT (news corpus) | +0.5–2% generalisation | High |

---

## 10. Glossary

| Term | Definition |
|------|-----------|
| TF-IDF | Term Frequency — Inverse Document Frequency. Converts words to numbers by measuring how distinctive a word is to a document versus the full corpus. |
| Tokenization | Breaking text into smaller units (words or subwords) for processing. |
| Lemmatization | Reducing a word to its base dictionary form (`running` → `run`). |
| Padding | Adding zeros to shorter sequences so all inputs are the same length. |
| Embedding | Dense vector representation of a word (e.g. 128-dimensional float vector). |
| RNN | Recurrent Neural Network. Processes sequences word-by-word, maintaining a hidden state. |
| LSTM | Long Short-Term Memory. An RNN with gates that control what to remember and forget. |
| GRU | Gated Recurrent Unit. A lighter LSTM variant with fewer parameters. |
| Transformer | Neural architecture using self-attention to process entire sequences simultaneously. |
| BERT | Bidirectional Encoder Representations from Transformers. Google's pretrained Transformer model. |
| `[CLS]` token | Special BERT token prepended to every input. Its final hidden state represents the whole sequence. |
| WordPiece | BERT's subword tokenizer. Splits unknown words into known subword units (`playing` → `play`, `##ing`). |
| BCEWithLogitsLoss | Binary cross-entropy loss with Sigmoid fused in — numerically stable under mixed precision. |
| AdamW | Adam optimizer with corrected weight decay (excludes bias and LayerNorm parameters). |
| Warmup | Gradually increasing the learning rate from 0 to the target value over the first N steps. |
| Early stopping | Halting training when validation loss stops improving, to prevent overfitting. |
| F1-Score | Harmonic mean of Precision and Recall. The most informative single metric for classification. |
| Mixed precision | Using float16 for forward pass and float32 for gradient accumulation — doubles training speed. |
