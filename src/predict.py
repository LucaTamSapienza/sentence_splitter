import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
import logging

logging.basicConfig(level=logging.WARNING)

class XLMRSentenceSplitter(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(out.last_hidden_state).squeeze(-1)
        return logits, None

def predict_xlmr(model, tokenizer, text, window_size=510, stride=256, batch_size=32, device="cpu"):
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
        if e > s and i < len(tok_prob):
            char_probs[e-1] = max(char_probs[e-1], float(tok_prob[i]))

    return char_probs

def split_text(text, char_probs, threshold):
    boundaries = [i for i, p in enumerate(char_probs) if p >= threshold]
    sentences, prev = [], 0
    for b in boundaries:
        sentences.append(text[prev:b+1].strip())
        prev = b+1
    if prev < len(text):
        sentences.append(text[prev:].strip())
    return [s for s in sentences if s]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Raw TXT file to process")
    parser.add_argument("--model_path", type=str, required=True, help="Path to best_xlmr_model.pt")
    parser.add_argument("--model_name_or_path", type=str, default="xlm-roberta-large", help="HuggingFace base model")
    parser.add_argument("--threshold", type=float, default=0.65, help="Decision threshold")
    parser.add_argument("--output_file", type=str, required=True, help="Output TXT with splits")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading tokenizer via {args.model_name_or_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=True)

    print("Loading model weights...")
    model = XLMRSentenceSplitter(model_name=args.model_name_or_path).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()

    print(f"Reading input from {args.input_file}...")
    raw_text = Path(args.input_file).read_text(encoding='utf-8')
    if not raw_text.strip():
        print("Input file is empty!")
        return

    print("Running inference...")
    char_probs = predict_xlmr(model, tokenizer, raw_text, device=device)

    sentences = split_text(raw_text, char_probs, args.threshold)

    print(f"Writing {len(sentences)} sentences to {args.output_file}...")
    with open(args.output_file, "w", encoding="utf-8") as f:
        for idx, s in enumerate(sentences, 1):
            f.write(f"{idx}) {s}\n\n")

    print("Done!")

if __name__ == "__main__":
    main()
