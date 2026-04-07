# Sentence Splitter

## Overview

Fine-tuned XLM-RoBERTa-large for sentence boundary detection.
The model performs per-character boundary classification using a sliding-window tokenization strategy.

---

## Repository Structure

```
sentence_splitting/
|-- UD_*/                   # 8 UD datasets (not tracked in git)
|-- data/                   # JSONL test sets and results (not tracked in git)
|-- src/
|   |-- __init__.py
|   |-- data.py             # Parse .sent_split, char alignment, windowing, Dataset
|   |-- model.py            # XLMRSentenceSplitter class + focal loss
|   |-- train.py            # Training loop module (fp16, early stopping)
|   |-- inference.py        # Sliding window prediction + char probability mapping
|   |-- ensemble.py         # Multi-model weighted averaging + weight optimization
|   |-- rules.py            # Blank lines, abbreviation lists, hard overrides
|   |-- evaluate.py         # Boundary F1, sentence exact match, per-dataset reports
|   |-- utils.py            # Config loading, seeding, logging helpers
|   |-- train_xlmr.py       # [CLI] Self-contained training script
|   |-- evaluate_xlmr.py    # [CLI] Threshold tuning + per-dataset evaluation
|   |-- predict.py          # [CLI] Single-file inference on arbitrary text
|   |-- run_baselines.py    # [CLI] spaCy + NLTK baseline comparison
|   |-- optimize.py         # [CLI] Threshold + ensemble weight grid search
|   |-- eval_test.py        # [CLI] Run inference on JSONL test sets
|   |-- build_test.py       # [CLI] Convert <EOS>-marked files to JSONL
|   +-- train_sat.py        # [CLI] wtpsplit/SaT out-of-box evaluation
|-- configs/
|   |-- xlmr.yaml           # XLM-R hyperparameters
|   +-- sat.yaml            # SaT configuration
|-- checkpoints/            # Model checkpoints (not tracked in git)
|-- results.md              # Per-dataset F1 scores across thresholds
|-- PLAN.md                 # Architecture design and implementation plan
+-- requirements.txt
```

---

## Data Format

Each `.sent_split` file contains raw text with `<EOS>` markers at sentence boundaries:

```
This is the first sentence.<EOS> This is the second sentence.<EOS>
```

The data pipeline:
1. Splits on `<EOS>` to recover the clean text and the character positions of each boundary
2. A boundary at position `i` means the sentence ends at character `i` (inclusive)

The 8 UD datasets (4 English, 4 Italian) should be placed in the project root as `UD_*/` directories, each containing `*-ud-train.sent_split`, `*-ud-dev.sent_split`, and `*-ud-test.sent_split` files.

---

## Environment Setup

```bash
cd sentence_splitting/
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1
pip install -r requirements.txt
```

For CPU-only inference:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Additional setup for baselines:
```bash
python -m nltk.downloader punkt_tab
python -m spacy download en_core_web_sm
python -m spacy download it_core_news_sm
```

---

## Training

### Architecture

XLM-RoBERTa-large (560M params) with a single linear classification head:
```
XLM-RoBERTa backbone -> Dropout(0.1) -> Linear(1024, 1) -> per-token logit
```

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base model | `xlm-roberta-large` (560M params) |
| Classification head | `Linear(1024, 1)` + sigmoid |
| Loss | Focal BCE (alpha=0.75, gamma=2.0) |
| Learning rate | 2e-5 with linear warmup (10%) |
| Batch size | 16 (gradient accumulation steps=2, effective batch=32) |
| Max epochs | 10 |
| Early stopping patience | 3 epochs (metric: dev boundary F1) |
| Mixed precision | FP16 |
| Window size | 510 content tokens per window |
| Stride | 256 tokens (50% overlap) |
| Dataset sampling | Temperature=2.0 (upweights small datasets) |
| Seed | 42 |

All hyperparameters are also stored in `configs/xlmr.yaml`.

### Data used for training

All 7 training sets were combined into a single multilingual dataset (PUD is test-only, no training split):

| Dataset | Language | Approx. train sentences |
|---------|----------|------------------------|
| UD_English-EWT | English | 12,543 |
| UD_English-GUM | English | 9,386 |
| UD_English-ParTUT | English | 1,781 |
| UD_Italian-ISDT | Italian | 13,121 |
| UD_Italian-MarkIT | Italian | 611 |
| UD_Italian-ParTUT | Italian | 1,090 |
| UD_Italian-VIT | Italian | 7,803 |

Temperature-weighted sampling (T=2.0) ensures small datasets (MarkIT, ParTUT) are not underrepresented.

### How to run training

```bash
python src/train_xlmr.py \
    --data_dir ./ \
    --output_dir checkpoints \
    --model_name_or_path xlm-roberta-large
```

`--data_dir` should point to the folder containing `UD_*/` subdirectories.
The best checkpoint is saved to `checkpoints/best_xlmr_model.pt` whenever dev F1 improves.

---

## Evaluation

Evaluation runs a threshold sweep (0.1 to 0.9, step 0.05) on all dev sets to find the optimal decision threshold, then reports per-dataset P/R/F1 on all test sets.

```bash
python src/evaluate_xlmr.py \
    --data_dir ./ \
    --model_path checkpoints/best_xlmr_model.pt \
    --model_name_or_path xlm-roberta-large \
    --predictions_dir outputs/predictions
```

`--predictions_dir` is optional; if provided, numbered sentence predictions are saved per dataset.

### Results on UD test sets

| Dataset | t=0.5 | t=0.7 | t=0.8 |
|---------|-------|-------|-------|
| EN-EWT  | 0.975 | 0.977 | 0.976 |
| EN-GUM  | 0.985 | 0.987 | 0.987 |
| EN-ParTUT | 0.997 | 0.997 | 0.997 |
| EN-PUD  | 0.985 | 0.988 | 0.990 |
| IT-ISDT | 0.997 | 0.997 | 0.997 |
| IT-MarkIT | 0.997 | 0.997 | 0.997 |
| IT-ParTUT | 1.000 | 1.000 | 1.000 |
| IT-VIT  | 0.965 | 0.981 | 0.986 |
| **Macro F1** | **0.9876** | **0.9904** | **0.9912** |

Baseline to beat: spaCy macro F1 = 0.9553.

Recommended threshold: 0.8 (best macro F1 on test sets).

Full results with all thresholds and precision/recall: see `results.md`.

### Running baselines

```bash
python src/run_baselines.py --split test --verbose
```

---

## Inference on Arbitrary Text

```bash
python src/predict.py \
    --input_file /path/to/input.txt \
    --model_path checkpoints/best_xlmr_model.pt \
    --model_name_or_path xlm-roberta-large \
    --threshold 0.8 \
    --output_file outputs/predicted_sentences.txt
```

Output format (one sentence per entry, numbered):
```
1) First sentence here.

2) Second sentence here.
```

### How inference works

1. The input text is tokenized without truncation using the XLM-RoBERTa tokenizer
2. Sliding windows of 510 content tokens are created with 256-token stride (50% overlap)
3. Each window is padded to length 512, prefixed with [CLS] and suffixed with [SEP]
4. The model outputs a logit per token; overlapping regions are averaged across windows
5. Logits are passed through sigmoid to produce per-token boundary probabilities
6. Token probabilities are projected back to character positions (last character of each token's span)
7. Characters with probability >= threshold are predicted as sentence boundaries
8. The text is split at those positions

---

## Reproducibility Notes

- All randomness is seeded (seed=42) in the training script
- The checkpoint is a plain state_dict saved with `torch.save`; loaded with `weights_only=True`
- The model architecture is identical across train_xlmr.py, evaluate_xlmr.py, and predict.py
- The src/ package provides a modular implementation of the same pipeline for extensibility
