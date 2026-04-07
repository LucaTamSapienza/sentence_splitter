"""
src/evaluate_xlmr.py — Tune decision threshold on dev sets and evaluate on test sets.

Usage:
    python src/evaluate_xlmr.py --data_dir ./ --model_path checkpoints/best_xlmr_model.pt
"""

import argparse
import logging
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.WARNING)

MODEL_NAME = 'xlm-roberta-large'
WINDOW_SIZE = 510
STRIDE = 256
EOS_TAG = '<EOS>'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_sent_split_file(filepath) -> Tuple[str, List[int]]:
    raw = Path(filepath).read_text(encoding='utf-8')
    parts = raw.split(EOS_TAG)
    clean = ''.join(parts)
    boundaries, offset = [], 0
    for part in parts[:-1]:
        offset += len(part)
        if offset > 0:
            boundaries.append(offset - 1)
    return clean, boundaries

class XLMRSentenceSplitter(nn.Module):
    def __init__(self, model_name=MODEL_NAME):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, 1)
    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(out.last_hidden_state).squeeze(-1)
        return logits, None

def predict_xlmr(model, tokenizer, text, window_size=510, stride=256, batch_size=32):
    enc = tokenizer(text, return_offsets_mapping=True, truncation=False, add_special_tokens=False)
    token_ids, offsets = enc['input_ids'], enc['offset_mapping']
    if not token_ids: return np.zeros(len(text))

    cls_id, sep_id, pad_id = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
    windows, N, start = [], len(token_ids), 0
    while start < N:
        end = min(start + window_size, N)
        ids = [cls_id] + token_ids[start:end] + [sep_id]
        mask = [1] * len(ids)
        pad = window_size + 2 - len(ids)
        ids += [pad_id] * pad
        mask += [0] * pad
        windows.append((torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long), start))
        if end == N: break
        start += stride

    logits_sum, logits_cnt = np.zeros(N, dtype=np.float64), np.zeros(N, dtype=np.int32)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = windows[i:i+batch_size]
            b_ids = torch.stack([w[0] for w in batch]).to(device)
            b_mask = torch.stack([w[1] for w in batch]).to(device)
            logits, _ = model(b_ids, b_mask)
            lg_np = logits.cpu().float().numpy()
            for j, (_, _, doc_start) in enumerate(batch):
                clen = min(window_size, N - doc_start)
                logits_sum[doc_start:doc_start+clen] += lg_np[j, 1:1+clen]
                logits_cnt[doc_start:doc_start+clen] += 1

    avg = logits_sum / np.maximum(logits_cnt, 1)
    tok_prob = 1.0 / (1.0 + np.exp(-avg))
    char_probs = np.zeros(len(text), dtype=np.float32)
    for i, (s, e) in enumerate(offsets):
        if e > s and i < len(tok_prob): char_probs[e-1] = max(char_probs[e-1], float(tok_prob[i]))
    return char_probs

def boundary_f1(pred: set, gold: set):
    tp, fp, fn = len(pred & gold), len(pred - gold), len(gold - pred)
    p = tp / (tp+fp) if (tp+fp) else 0.0
    r = tp / (tp+fn) if (tp+fn) else 0.0
    return p, r, 2*p*r/(p+r) if (p+r) else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data dir containing UD_* folders")
    parser.add_argument("--model_path", type=str, required=True, help="Path to best_xlmr_model.pt")
    parser.add_argument("--model_name_or_path", type=str, default="xlm-roberta-large", help="HuggingFace base model name or offline path")
    parser.add_argument("--predictions_dir", type=str, default=None, help="Directory to save predicted sentence splits")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=True)

    model = XLMRSentenceSplitter(model_name=args.model_name_or_path).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()
    print("Model loaded successfully.")

    # 1. Tune threshold on dev sets
    print("\n--- Tuning Threshold on Dev Sets ---")
    all_dev = sorted(data_dir.glob('UD_*/*-ud-dev.sent_split'))
    dev_probs, dev_golds = [], []
    for fpath in all_dev:
        text, gold = parse_sent_split_file(fpath)
        if not text: continue
        dev_probs.append(predict_xlmr(model, tokenizer, text))
        dev_golds.append(set(gold))

    best_t, best_f1 = 0.5, 0.0
    print(f"{'Threshold':>10}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    print("-" * 44)
    for t in np.arange(0.1, 0.91, 0.05):
        ps, rs, f1s = [], [], []
        for probs, gold in zip(dev_probs, dev_golds):
            pred = {i for i, p in enumerate(probs) if p >= t}
            p, r, f1 = boundary_f1(pred, gold)
            ps.append(p); rs.append(r); f1s.append(f1)
        avg_f1 = sum(f1s)/len(f1s) if f1s else 0.0
        marker = ' <- BEST' if avg_f1 > best_f1 else ''
        print(f"{t:>10.2f}  {sum(ps)/len(ps):>10.4f}  {sum(rs)/len(rs):>8.4f}  {avg_f1:>8.4f}{marker}")
        if avg_f1 > best_f1:
            best_f1, best_t = avg_f1, t

    print(f"\nBest threshold selected: {best_t:.2f} (dev F1: {best_f1:.4f})")

    # 2. Evaluate on test sets
    print("\n--- Final Evaluation on Test Sets ---")
    all_test = sorted(data_dir.glob('UD_*/*-ud-test.sent_split'))
    print(f"{'Dataset':<40} {'P':>7} {'R':>7} {'F1':>7} {'#Gold':>7}")
    print("-" * 68)

    all_f1s = []
    for fpath in all_test:
        text, gold = parse_sent_split_file(fpath)
        if not text: continue
        char_probs = predict_xlmr(model, tokenizer, text)
        pred = {i for i, p in enumerate(char_probs) if p >= best_t}
        p, r, f1 = boundary_f1(pred, set(gold))
        all_f1s.append(f1)
        label = f"{fpath.parent.name}/{fpath.name}"[:40]
        print(f"{label:<40} {p:>7.4f} {r:>7.4f} {f1:>7.4f} {len(gold):>7}")

        if args.predictions_dir:
            out_file = Path(args.predictions_dir) / f"{fpath.parent.name}_{fpath.stem}_preds.txt"
            out_file.parent.mkdir(parents=True, exist_ok=True)

            boundaries = sorted(list(pred))
            sentences, prev = [], 0
            for b in boundaries:
                sentences.append(text[prev:b+1].strip())
                prev = b+1
            if prev < len(text):
                sentences.append(text[prev:].strip())

            with open(out_file, "w", encoding="utf-8") as f:
                idx = 1
                for s in sentences:
                    if s.strip():
                        f.write(f"{idx}) {s}\n")
                        idx += 1

    print("-" * 68)
    macro_avg = sum(all_f1s)/len(all_f1s) if all_f1s else 0.0
    print(f"{'MACRO AVG':<40} {'':>7} {'':>7} {macro_avg:>7.4f}")
    print(f"\nBaseline to beat: NLTK = 0.9411, spaCy = 0.9519")

if __name__ == '__main__':
    main()
