import argparse
import logging
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

logging.basicConfig(level=logging.WARNING)

# Configuration defaults
MODEL_NAME = 'xlm-roberta-large'
WINDOW_SIZE = 510
STRIDE = 256
BATCH_SIZE = 16
GRAD_ACCUM = 2
NUM_EPOCHS = 10
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
FP16 = True
PATIENCE = 3
SEED = 42
FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 2.0
TEMPERATURE = 2.0
THRESHOLD = 0.5
EOS_TAG = '<EOS>'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

def build_windows(text: str, boundaries: List[int], tokenizer, window_size=510, stride=256):
    enc = tokenizer(text, return_offsets_mapping=True, truncation=False, add_special_tokens=False)
    token_ids, offsets = enc['input_ids'], enc['offset_mapping']
    if not token_ids: return [], [], [], []
    char_to_tok = [-1] * len(text)
    for tok_i, (s, e) in enumerate(offsets):
        for c in range(s, min(e, len(text))): char_to_tok[c] = tok_i
    token_labels = [0] * len(token_ids)
    for b in boundaries:
        if 0 <= b < len(char_to_tok) and char_to_tok[b] >= 0:
            token_labels[char_to_tok[b]] = 1

    cls_id, sep_id, pad_id = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
    windows, N, start = [], len(token_ids), 0
    while start < N:
        end = min(start + window_size, N)
        ids = [cls_id] + token_ids[start:end] + [sep_id]
        labs = [-100] + token_labels[start:end] + [-100]
        mask = [1] * len(ids)
        pad = window_size + 2 - len(ids)
        ids += [pad_id] * pad
        labs += [-100] * pad
        mask += [0] * pad
        windows.append({
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'labels': torch.tensor(labs, dtype=torch.long),
            'doc_token_start': start,
        })
        if end == N: break
        start += stride
    return windows, token_ids, offsets, token_labels

def augment_text(text, boundaries, rng):
    if len(text) < 200: return text, boundaries
    if rng.random() < 0.2:
        cut = rng.randint(1, min(100, len(text)//4))
        text = text[cut:]
        boundaries = [b-cut for b in boundaries if b-cut >= 0]
    if rng.random() < 0.2:
        cut = rng.randint(1, min(100, len(text)//4))
        text = text[:len(text)-cut]
        boundaries = [b for b in boundaries if b < len(text)]
    return text, boundaries

class SentSplitDataset(Dataset):
    def __init__(self, file_paths, tokenizer, window_size=510, stride=256, augment=False, seed=42):
        self.windows = []
        rng = random.Random(seed)
        for fp in file_paths:
            text, bounds = parse_sent_split_file(fp)
            if not text: continue
            if augment: text, bounds = augment_text(text, bounds, rng)
            wins, _, _, _ = build_windows(text, bounds, tokenizer, window_size, stride)
            self.windows.extend(wins)
    def __len__(self): return len(self.windows)
    def __getitem__(self, idx):
        w = self.windows[idx]
        return {'input_ids': w['input_ids'], 'attention_mask': w['attention_mask'], 'labels': w['labels']}

def build_combined_loader(train_file_groups, tokenizer, batch_size, window_size=510, stride=256, temperature=2.0):
    per_ds_windows = []
    for i, files in enumerate(train_file_groups):
        ds = SentSplitDataset(files, tokenizer, window_size, stride, augment=True, seed=SEED+i)
        per_ds_windows.append(ds.windows)

    sizes = [len(w) for w in per_ds_windows]
    total = sum(sizes)
    raw_w = [(n/total)**(1.0/temperature) if total > 0 else 0 for n in sizes]
    z = sum(raw_w)
    dw = [w/z for w in raw_w] if z > 0 else []

    sample_weights = []
    for w, size in zip(dw, sizes):
        sample_weights.extend([w/size if size else 0] * size)

    combined = SentSplitDataset.__new__(SentSplitDataset)
    combined.windows = [w for group in per_ds_windows for w in group]

    sampler = WeightedRandomSampler(sample_weights, len(combined), replacement=True)
    loader = DataLoader(combined, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    return loader

def focal_bce_loss(logits, labels, alpha=0.75, gamma=2.0):
    logits, labels = logits.reshape(-1), labels.reshape(-1)
    mask = labels != -100
    logits, targets = logits[mask], labels[mask].float()
    if targets.numel() == 0: return logits.sum() * 0.0
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1 - probs) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    weight = alpha_t * (1 - p_t).pow(gamma)
    return (weight * bce).mean()

class XLMRSentenceSplitter(nn.Module):
    def __init__(self, model_name=MODEL_NAME, dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, 1)
    def forward(self, input_ids, attention_mask, labels=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(self.dropout(out.last_hidden_state)).squeeze(-1)
        loss = focal_bce_loss(logits, labels, FOCAL_ALPHA, FOCAL_GAMMA) if labels is not None else None
        return logits, loss

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

def evaluate_on_dev(model, tokenizer, dev_files, threshold=0.5):
    all_f1 = []
    for fpath in dev_files:
        text, gold = parse_sent_split_file(fpath)
        if not text: continue
        pred = {i for i, p in enumerate(predict_xlmr(model, tokenizer, text)) if p >= threshold}
        _, _, f1 = boundary_f1(pred, set(gold))
        all_f1.append(f1)
    return sum(all_f1)/len(all_f1) if all_f1 else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data dir containing UD_* folders")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Where to save model")
    parser.add_argument("--model_name_or_path", type=str, default="xlm-roberta-large", help="HuggingFace model name or local path")
    args = parser.parse_args()

    set_seed(SEED)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    train_groups = []
    for d in sorted(data_dir.glob('UD_*/')):
        files = sorted(d.glob('*-ud-train.sent_split'))
        if files: train_groups.append(files)

    all_dev = sorted(data_dir.glob('UD_*/*-ud-dev.sent_split'))

    train_loader = build_combined_loader(train_groups, tokenizer, BATCH_SIZE, WINDOW_SIZE, STRIDE, TEMPERATURE)
    model = XLMRSentenceSplitter(model_name=args.model_name_or_path).to(device)

    no_decay = {'bias', 'LayerNorm.weight'}
    optimizer = torch.optim.AdamW([
        {'params': [p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': WEIGHT_DECAY},
        {'params': [p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ], lr=LEARNING_RATE)

    total_steps = (len(train_loader) // GRAD_ACCUM) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * WARMUP_RATIO), total_steps)
    scaler = GradScaler('cuda') if FP16 and torch.cuda.is_available() else None

    best_f1, patience_counter = 0.0, 0
    checkpoint_path = out_dir / 'best_xlmr_model.pt'

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        t0 = time.time()
        for step, batch in enumerate(train_loader):
            ids, masks, labs = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            if scaler:
                with autocast('cuda'): _, loss = model(ids, masks, labs)
                scaler.scale(loss / GRAD_ACCUM).backward()
            else:
                _, loss = model(ids, masks, labs)
                (loss / GRAD_ACCUM).backward()
            epoch_loss += loss.item()
            if (step + 1) % GRAD_ACCUM == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                if scaler:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if (step + 1) % 100 == 0:
                print(f'  Epoch {epoch} | step {step+1}/{len(train_loader)} | loss {epoch_loss/(step+1):.4f}', end='\r')

        avg_loss = epoch_loss / len(train_loader)
        dev_f1 = evaluate_on_dev(model, tokenizer, all_dev)
        print(f'\nEpoch {epoch:02d}/{NUM_EPOCHS} | loss {avg_loss:.4f} | dev F1 {dev_f1:.4f} | {(time.time()-t0)/60:.1f} min')

        if dev_f1 > best_f1:
            best_f1, patience_counter = dev_f1, 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f'  → New best! Checkpoint saved')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f'Early stopping. Best dev F1: {best_f1:.4f}')
                break

if __name__ == '__main__':
    main()
