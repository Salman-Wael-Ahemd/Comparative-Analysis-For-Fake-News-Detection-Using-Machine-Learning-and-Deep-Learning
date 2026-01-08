# Comparative Analysis For Fake News Detection Using Machine Learning And Deep Learning

This repository contains an NLP project for **fake-news detection** using two approaches:

- **Machine Learning (ML):** TF–IDF + POS-based features → **Linear SVC (C=2.0)**
- **Deep Learning (DL):** Transformer fine-tuning using **DistilBERT (distilbert-base-uncased)**

The models are trained and evaluated on two datasets:
- **WELFake** (primary dataset)
- **ISOT** (used to validate generalizability)

---

## Project Overview

Fake news spreads quickly on social media, making manual fact-checking difficult at scale. This project builds automated fake-news classifiers and compares:
1) A  classical baseline (LinearSVC with engineered text features)
2) A transformer-based model (DistilBERT) that captures contextual meaning

---

## Datasets (Included)

This repository includes the datasets in the `data/` directory.

- **WELFake** (Verma et al.)
- **ISOT** (Ahmed et al.)

---
## Preprocessing

### Shared preprocessing (applied to both datasets)
- Missing values check using `df.info()`
  - **WELFake:** missing values in `title` and `text`
    - Missing `title` → replaced with `""`
    - Missing `text` → row dropped
  - **ISOT:** no missing values
- Text normalization:
  - lowercase
  - whitespace cleanup
  - remove URLs
  - remove HTML tags
  - remove emojis
- Combine `title + text` into a single input column

### ML-only preprocessing
- Remove punctuation **except** hyphens (`-`) and apostrophes (`'`)
- Remove stopwords
- POS-aware lemmatization

### DL preprocessing
- No extra text cleaning beyond the shared steps (tokenizer handles text)

---

## Text Representation

### ML representation
- **POS features (proportions):** noun / verb / adjective / adverb (+ modal verbs)
- **TF–IDF Vectorization:**
  - `max_features = 50000`
  - `ngram_range = (1, 2)`
  - `min_df = 5`
  - `max_df = 0.8`

### DL representation (DistilBERT)
Text is tokenized into:
- `input_ids` (token indices used to retrieve embeddings)
- `attention_mask` (marks real tokens vs padding)
- `max_length = 512`

---

## Models

### 1) Machine Learning Model — LinearSVC
- Model: `LinearSVC(C=2.0)`
- Trained on: **TF–IDF + POS-proportion features**

### 2) Deep Learning Model — DistilBERT
- Model: `distilbert-base-uncased` (`AutoModelForSequenceClassification`, `num_labels=2`)
- Training configuration:
  - `EPOCHS = 3`
  - `BATCH_SIZE = 32`
  - `MAX_LEN = 512`
  - Optimizer: `AdamW(lr=2e-5)`
  - Scheduler: linear warmup (`warmup = 10% of total steps`)
  - Gradient clipping: `max_norm = 1.0`
  - Best checkpoint selected by **validation performance**

---

## Results (Test Set)

### WELFake
| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| LinearSVC | 0.9735 | 0.9666 | 0.9825 | 0.9744 |
| DistilBERT | 0.9953 | 0.9957 | 0.9951 | 0.9954 |

**Confusion Matrix (WELFake):**
- LinearSVC: **72 FP**, **20 FN**
- DistilBERT: **3 FP**, **3 FN**

### ISOT
| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| LinearSVC | 0.9980 | 0.9991 | 0.9969 | 0.9980 |
| DistilBERT | 0.9999 | 0.9999 | 0.9999 | 0.9999 |

---

## Installation

### 1) Create a virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
