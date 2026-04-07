"""
src/ — Sentence boundary detection package.

Core library (imported by other modules):
    model.py        XLM-RoBERTa classifier architecture + focal loss
    data.py         Data parsing, windowing, dataset construction
    train.py        Training loop (fp16, early stopping)
    inference.py    Sliding-window prediction pipeline
    evaluate.py     Metrics: boundary F1, sentence exact match
    ensemble.py     Multi-model blending & weight optimization
    rules.py        Rule-based boundary heuristics (abbreviations, blank lines)
    utils.py        Config loading, seeding, logging helpers

CLI scripts (run directly with python src/<script>.py):
    train_xlmr.py       Train an XLM-R sentence splitter on UD data
    evaluate_xlmr.py    Tune threshold on dev & evaluate on test sets
    predict.py          Run inference on files or folders
    run_baselines.py    NLTK + spaCy baseline evaluation
    optimize.py         Grid-search ensemble weights & threshold
    eval_test.py        Evaluate on JSONL gold test sets
    build_test.py       Convert <EOS>-annotated files to JSONL format
    train_sat.py        Evaluate wtpsplit / SaT models
"""

from .model import XLMRSentenceSplitter, focal_bce_loss
from .data import (
    parse_sent_split_file,
    detect_lang,
    get_all_dev_files,
    get_all_test_files,
    get_all_train_file_groups,
    SentSplitDataset,
)
from .evaluate import boundary_f1, sentence_exact_match, format_report
from .inference import predict_xlmr, predict_sat, probs_to_boundaries
from .ensemble import EnsemblePredictor, optimise_ensemble, threshold_sweep
from .rules import rule_based_probs, get_hard_boundaries
from .utils import load_config, set_seed, get_logger

__all__ = [
    # model
    "XLMRSentenceSplitter",
    "focal_bce_loss",
    # data
    "parse_sent_split_file",
    "detect_lang",
    "get_all_dev_files",
    "get_all_test_files",
    "get_all_train_file_groups",
    "SentSplitDataset",
    # evaluate
    "boundary_f1",
    "sentence_exact_match",
    "format_report",
    # inference
    "predict_xlmr",
    "predict_sat",
    "probs_to_boundaries",
    # ensemble
    "EnsemblePredictor",
    "optimise_ensemble",
    "threshold_sweep",
    # rules
    "rule_based_probs",
    "get_hard_boundaries",
    # utils
    "load_config",
    "set_seed",
    "get_logger",
]
