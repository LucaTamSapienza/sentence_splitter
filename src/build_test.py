"""
scripts/build_test.py — Convert any text file with <EOS> markers to JSONL.

Each output line is a self-contained JSON object:
  id              — derived from the input filename stem
  source          — original file path
  lang            — detected language ("en" | "it"), overridable
  text            — clean text with <EOS> removed
  gold_boundaries — sorted list of char indices where sentences end
  gold_sentences  — list of sentence strings (human-readable)

The JSONL is the contract between this script and eval_test.py.
You can run this once and eval as many times as you want with different models.

Usage:
    # Single file:
    python scripts/build_test.py \\
        --input data/hard_test.txt \\
        --output data/hard_test.jsonl

    # Multiple files → one JSONL (one entry per file):
    python scripts/build_test.py \\
        --input UD_English-EWT/en_ewt-ud-test.sent_split \\
                UD_Italian-ISDT/it_isdt-ud-test.sent_split \\
        --output data/ud_test.jsonl

    # Preview without writing:
    python scripts/build_test.py \\
        --input data/hard_test.txt \\
        --output data/hard_test.jsonl \\
        --dry-run
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import detect_lang

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

EOS_TAG = "<EOS>"


# ---------------------------------------------------------------------------
# Core parsing — works on any file with <EOS> markers, any extension
# ---------------------------------------------------------------------------


def parse_eos_file(filepath: Path) -> Tuple[str, List[int]]:
    """
    Parse any text file containing <EOS> markers.

    Removes all <EOS> tags and returns:
      - clean_text:         the raw text without <EOS>
      - boundary_positions: sorted list of 0-based char indices of the last
                            character of each sentence in clean_text

    This is the same logic as src.data.parse_sent_split_file but standalone,
    so build_test.py has no dependency on the rest of src/.
    """
    raw = filepath.read_text(encoding="utf-8")
    parts = raw.split(EOS_TAG)

    clean_text = "".join(parts)

    boundary_positions: List[int] = []
    offset = 0
    for part in parts[:-1]:
        offset += len(part)
        if offset > 0:
            pos = offset - 1
            if not boundary_positions or boundary_positions[-1] != pos:
                boundary_positions.append(pos)

    return clean_text, boundary_positions


def boundaries_to_sentences(text: str, boundaries: List[int]) -> List[str]:
    """Reconstruct sentence strings from boundary char positions."""
    sentences: List[str] = []
    prev = 0
    for b in sorted(boundaries):
        sentences.append(text[prev : b + 1])
        prev = b + 1
    if prev < len(text) and text[prev:].strip():
        sentences.append(text[prev:])
    return sentences


def build_entry(filepath: Path, lang_override: Optional[str] = None) -> dict:
    """Build one JSONL entry from a file with <EOS> markers."""
    text, boundaries = parse_eos_file(filepath)
    lang = lang_override or detect_lang(filepath)
    sentences = boundaries_to_sentences(text, boundaries)
    return {
        "id":               filepath.stem,
        "source":           str(filepath),
        "lang":             lang,
        "text":             text,
        "gold_boundaries":  boundaries,
        "gold_sentences":   sentences,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert text files with <EOS> markers to JSONL."
    )
    parser.add_argument(
        "--input", nargs="+", required=True, metavar="FILE",
        help="One or more files with <EOS> markers (.txt, .sent_split, etc.).",
    )
    parser.add_argument(
        "--output", required=True, metavar="FILE",
        help="Output .jsonl path.",
    )
    parser.add_argument(
        "--lang", default=None, choices=["en", "it"],
        help="Override language detection. Applied to all inputs.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print entries to stdout without writing the output file.",
    )
    args = parser.parse_args()

    entries = []
    for raw_path in args.input:
        fpath = Path(raw_path)
        if not fpath.exists():
            logger.error(f"File not found: {fpath}")
            sys.exit(1)

        entry = build_entry(fpath, lang_override=args.lang)
        entries.append(entry)
        logger.info(
            f"{fpath.name}: {len(entry['gold_sentences'])} sentences, "
            f"{len(entry['text'])} chars, lang={entry['lang']}"
        )

    lines = [json.dumps(e, ensure_ascii=False) for e in entries]

    if args.dry_run:
        for line in lines:
            print(line)
        logger.info("Dry run — nothing written.")
        return

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Written {len(entries)} entr{'y' if len(entries)==1 else 'ies'} → {out_path}")


if __name__ == "__main__":
    main()
