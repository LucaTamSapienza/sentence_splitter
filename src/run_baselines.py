"""
scripts/run_baselines.py — Evaluate NLTK and spaCy sentence splitters on all
dev and test sets. Establishes the performance floor needed for the +3 bonus.

Usage:
    python scripts/run_baselines.py
    python scripts/run_baselines.py --split test   # evaluate on test sets
    python scripts/run_baselines.py --verbose       # show per-file breakdown

Requirements:
    pip install nltk spacy
    python -m nltk.downloader punkt_tab
    python -m spacy download en_core_web_sm
    python -m spacy download it_core_news_sm
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import nltk
import spacy

from src.data import parse_sent_split_file, get_all_dev_files, get_all_test_files, detect_lang
from src.evaluate import boundary_f1, format_report

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)


# ---------------------------------------------------------------------------
# NLTK Punkt baseline
# ---------------------------------------------------------------------------


def predict_nltk(text: str, lang: str = "en") -> List[int]:
    """
    Sentence-tokenise text using NLTK Punkt and return boundary positions.

    Strategy: run sent_tokenize to get sentences, then find each sentence's
    end position in the original text (important: preserves original spacing).
    """
    nltk_lang = "english" if lang == "en" else "italian"

    sentences = nltk.sent_tokenize(text, language=nltk_lang)

    boundaries: List[int] = []
    search_from = 0
    for sent in sentences:
        # Find this sentence in the original text starting from search_from
        pos = text.find(sent, search_from)
        if pos == -1:
            # Fallback: stripped search
            pos = text.find(sent.strip(), search_from)
        if pos == -1:
            continue
        end = pos + len(sent) - 1
        boundaries.append(end)
        search_from = pos + len(sent)

    return sorted(set(boundaries))


# ---------------------------------------------------------------------------
# spaCy baseline
# ---------------------------------------------------------------------------


_SPACY_MODELS: Dict[str, str] = {
    "en": "en_core_web_sm",
    "it": "it_core_news_sm",
}
_SPACY_CACHE: Dict[str, object] = {}


def _get_spacy(lang: str):
    if lang not in _SPACY_CACHE:
        model_name = _SPACY_MODELS[lang]
        try:
            _SPACY_CACHE[lang] = spacy.load(model_name, disable=["ner", "lemmatizer"])
        except OSError:
            logger.error(
                f"spaCy model '{model_name}' not found. "
                f"Run: python -m spacy download {model_name}"
            )
            return None
    return _SPACY_CACHE[lang]


def predict_spacy(text: str, lang: str = "en") -> List[int]:
    """
    Sentence-tokenise text using spaCy and return boundary positions.
    """
    nlp = _get_spacy(lang)
    if nlp is None:
        return []

    doc = nlp(text)

    boundaries: List[int] = []
    for sent in doc.sents:
        end_char = sent.end_char - 1  # last char of the sentence
        # Walk back past trailing whitespace
        while end_char > 0 and text[end_char] in " \t\n":
            end_char -= 1
        if end_char >= 0:
            boundaries.append(end_char)

    return sorted(set(boundaries))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_baseline(
    predictor_fn,
    files: List[Path],
    name: str,
) -> Tuple[List[Dict], float]:
    """
    Evaluate a baseline predictor on a list of .sent_split files.

    Returns (per-file results list, macro-average F1).
    """
    results = []
    for fpath in files:
        text, gold_boundaries = parse_sent_split_file(fpath)
        if not text:
            continue

        lang = detect_lang(fpath)

        try:
            pred_boundaries = predictor_fn(text, lang)
        except Exception as e:
            logger.warning(f"  [{name}] failed on {fpath.name}: {e}")
            pred_boundaries = []

        metrics = boundary_f1(set(pred_boundaries), set(gold_boundaries))
        metrics["dataset"] = fpath.parent.name
        metrics["file"]    = fpath.name
        metrics["n_gold"]  = float(len(gold_boundaries))
        metrics["n_pred"]  = float(len(pred_boundaries))
        results.append(metrics)

    avg_f1 = (
        sum(r["f1"] for r in results) / len(results) if results else 0.0
    )
    return results, avg_f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate NLTK and spaCy baselines on UD sent_split data."
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test", "both"],
        default="dev",
        help="Which data split to evaluate (default: dev)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print per-file breakdown"
    )
    args = parser.parse_args()

    # Download NLTK data if needed
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        logger.info("Downloading NLTK punkt_tab...")
        nltk.download("punkt_tab", quiet=True)

    # Collect files
    if args.split in ("dev", "both"):
        dev_files = get_all_dev_files()
    if args.split in ("test", "both"):
        test_files = get_all_test_files()

    for split_name, files in [
        ("dev",  dev_files  if args.split in ("dev",  "both") else []),
        ("test", test_files if args.split in ("test", "both") else []),
    ]:
        if not files:
            continue

        print(f"\n{'#' * 78}")
        print(f"  SPLIT: {split_name.upper()} ({len(files)} files)")
        print(f"{'#' * 78}")

        for name, fn in [("NLTK", predict_nltk), ("spaCy", predict_spacy)]:
            results, avg_f1 = evaluate_baseline(fn, files, name)
            if args.verbose:
                print(format_report(results, title=f"{name} — {split_name}"))
            else:
                print(f"\n  [{name}]  macro-avg F1 = {avg_f1:.4f}")
                for r in results:
                    print(
                        f"    {r['dataset']}/{r['file']:<35}  "
                        f"P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1']:.3f}"
                    )

    print(
        "\nNote: Your trained model must beat BOTH of these to claim the +3 bonus.\n"
    )


if __name__ == "__main__":
    main()
