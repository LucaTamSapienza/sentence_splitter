---
language:
  - en
  - it
license: mit
tags:
  - sentence-segmentation
  - sentence-boundary-detection
  - xlm-roberta
  - token-classification
datasets:
  - universal_dependencies
base_model: xlm-roberta-large
---

# XLM-RoBERTa Sentence Splitter

Fine-tuned **XLM-RoBERTa-large** (560M params) for sentence boundary detection on 8 Universal Dependencies treebanks (4 English, 4 Italian).

## Results

| | NLTK | spaCy | **This model** |
|---|---|---|---|
| **Macro F1** | 0.9411 | 0.9519 | **0.9863** |

## Usage

```bash
git clone https://github.com/LucaTamSapienza/sentence_splitter.git
cd sentence_splitter
pip install -r requirements.txt
python download_model.py
python src/predict.py --input input/ --output output/ --model_path checkpoints/best_xlmr_model.pt
```

## Architecture

```
XLM-RoBERTa-large -> Dropout(0.1) -> Linear(1024, 1) -> per-token sigmoid
```

Trained with focal loss (alpha=0.75, gamma=2.0), sliding windows (510 tokens, stride 256), FP16, AdamW (lr=2e-5).

## Citation

```
@misc{tam2026sentencesplitter,
  author = {Luca Tam},
  title = {Sentence Splitter: Fine-tuning XLM-RoBERTa for Sentence Boundary Detection},
  year = {2026},
  url = {https://github.com/LucaTamSapienza/sentence_splitter}
}
```
