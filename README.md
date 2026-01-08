# Comparative Analysis For Fake News Detection Using Machine Learning And Deep Learning

This project builds and compares two approaches for fake-news detection using Natural Language Processing (NLP):
1) A **classical machine-learning model** using **Linear SVC** with **TF–IDF + POS-based features**
2) A **deep-learning transformer model** using **DistilBERT** fine-tuning

The project is evaluated on **WELFake** (primary dataset) and validated on **ISOT** (generalization test).

---

## Datasets
- **WELFake** (Verma et al.) — combined from multiple sources (Kaggle, McIntire, Reuters, BuzzFeed Political)
- **ISOT** (Ahmed et al.) — used as an external dataset to assess generalizability

> Note: Datasets are not included in this repository. Please download them from their official sources and place them in the paths described below.

---

## Methods Overview

### 1) Machine Learning Pipeline (Linear SVC)
- Data cleaning (lowercasing, URL/HTML/emoji removal, whitespace normalization)
- Combine `title + text` into one input column
- POS tagging → compute POS **proportions** (noun/verb/adj/adv + modal)
- POS-aware lemmatization
- TF–IDF vectorization:
  - max_features = 50,000
  - ngram_range = (1, 2)
  - min_df = 5
  - max_df = 0.8
- Model: **LinearSVC (C = 2.0)**

### 2) Deep Learning Pipeline (DistilBERT)
- Tokenization using `distilbert-base-uncased`
- Inputs: `input_ids` + `attention_mask` (max length = 512)
- Model: `AutoModelForSequenceClassification` (num_labels = 2)
- Training setup:
  - epochs = 3
  - batch size = 32
  - optimizer = AdamW (lr = 2e-5)
  - scheduler = linear warmup (10% warmup steps)
  - gradient clipping = 1.0
  - best checkpoint selected by validation performance

---

## Results (Test Set)

### WELFake
| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| LinearSVC | 0.9735 | 0.9666 | 0.9825 | 0.9744 |
| DistilBERT | 0.9953 | 0.9957 | 0.9951 | 0.9954 |

**Confusion Matrix (WELFake):**
- LinearSVC: 72 FP, 20 FN  
- DistilBERT: 3 FP, 3 FN  

### ISOT
| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| LinearSVC | 0.9980 | 0.9991 | 0.9969 | 0.9980 |
| DistilBERT | 0.9999 | 0.9999 | 0.9999 | 0.9999 |

---

## Repository Structure
Suggested structure (edit based on your actual files):
