from __future__ import annotations

import csv
import difflib
import re
import time
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = REPO_ROOT / "data" / "raw"
TRAIN_DIR = REPO_ROOT / "train_folder"
TRAIN_DIR.mkdir(exist_ok=True)

OUTPUT_SRC = TRAIN_DIR / "try4_train.src"
OUTPUT_TGT = TRAIN_DIR / "try4_train.tgt"
OUTPUT_ADDED = TRAIN_DIR / "try4_added_only.csv"
OUTPUT_READ = TRAIN_DIR / "try4_read.md"
OUTPUT_LOG = TRAIN_DIR / "try4_process.log"


def log(message: str) -> None:
    print(message, flush=True)
    with OUTPUT_LOG.open("a", encoding="utf-8") as file:
        file.write(message + "\n")


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def normalize_transliteration(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("ḫ", "h").replace("Ḫ", "H")
    text = text.replace("sz", "š").replace("SZ", "Š")
    text = re.sub(r"\[\.\.\.\]", "<gap>", text)
    text = re.sub(r"x x x", "<gap>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def only_punct_or_nums(text: str) -> bool:
    return not bool(re.search(r"[a-zšṣṭḫāīū]", text, re.I))


def english_sentence_split(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"“])', text)
    return [part.strip() for part in parts if part.strip()] or [text]


def detect_noise(src: str, tgt: str) -> bool:
    if len(tgt) < 3:
        return True
    if len(src) == 0:
        return True
    ratio = len(tgt) / len(src)
    if ratio < 0.2 or ratio > 3.5:
        return True
    similarity = difflib.SequenceMatcher(None, src, tgt).ratio()
    if similarity > 0.85:
        return True
    return False


def is_clean_pair(src: str, tgt: str) -> bool:
    src = clean_text(src)
    tgt = clean_text(tgt)
    if len(src.split()) < 2 or len(src.split()) > 35:
        return False
    if len(tgt.split()) < 3 or len(tgt.split()) > 45:
        return False
    token_ratio = len(tgt.split()) / max(1, len(src.split()))
    if token_ratio < 0.3 or token_ratio > 4.2:
        return False
    if only_punct_or_nums(src) or only_punct_or_nums(tgt):
        return False
    if detect_noise(src, tgt):
        return False
    if src.count(".") > 1:
        return False
    if tgt.count('"') % 2 == 1:
        return False
    return True


def baseline_pairs(train_path: Path) -> list[tuple[str, str]]:
    pairs = []
    with train_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            src_lines = [x.strip() for x in re.split(r"[\n\.]", str(row["transliteration"])) if x.strip()]
            tgt_lines = [x.strip() for x in re.split(r"[\n\.]", str(row["translation"])) if x.strip()]
            for src, tgt in zip(src_lines, tgt_lines):
                if len(src) >= 5 and len(tgt) >= 5:
                    src_clean = clean_text(src)
                    tgt_clean = clean_text(tgt)
                    if only_punct_or_nums(src_clean) or only_punct_or_nums(tgt_clean):
                        continue
                    pairs.append((src_clean, tgt_clean))
    return list(dict.fromkeys(pairs))


def heuristic_extra_pairs(train_path: Path, baseline_set: set[tuple[str, str]]) -> list[tuple[str, str]]:
    extras = []
    with train_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            src = normalize_transliteration(row["transliteration"])
            tgt = normalize_transliteration(row["translation"])
            if not src or not tgt:
                continue

            tgt_sentences = english_sentence_split(tgt)
            if len(tgt_sentences) <= 1:
                continue

            src_words = src.split()
            total_src = len(src_words)
            total_tgt_chars = max(1, len(tgt))
            cursor = 0
            chunks = []

            for sentence in tgt_sentences:
                proportional_len = max(1, int(round(total_src * (len(sentence) / total_tgt_chars))))
                end = min(cursor + proportional_len, total_src)
                chunks.append((" ".join(src_words[cursor:end]), sentence))
                cursor = end

            if cursor < total_src and chunks:
                last_src, last_tgt = chunks[-1]
                chunks[-1] = ((last_src + " " + " ".join(src_words[cursor:])).strip(), last_tgt)

            for chunk_src, chunk_tgt in chunks:
                src_clean = clean_text(chunk_src)
                tgt_clean = clean_text(chunk_tgt)
                pair = (src_clean, tgt_clean)
                if pair in baseline_set:
                    continue
                if is_clean_pair(src_clean, tgt_clean):
                    extras.append(pair)

    return list(dict.fromkeys(extras))


def write_outputs(baseline: list[tuple[str, str]], extras: list[tuple[str, str]], elapsed: float) -> None:
    full_pairs = baseline + extras
    with OUTPUT_SRC.open("w", encoding="utf-8") as src_file:
        src_file.write("\n".join(src for src, _ in full_pairs) + "\n")
    with OUTPUT_TGT.open("w", encoding="utf-8") as tgt_file:
        tgt_file.write("\n".join(tgt for _, tgt in full_pairs) + "\n")

    with OUTPUT_ADDED.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["akkadian", "english"])
        writer.writeheader()
        for src, tgt in extras:
            writer.writerow({"akkadian": src, "english": tgt})

    with OUTPUT_READ.open("w", encoding="utf-8") as file:
        file.write("# Try 4 Train Expansion Log\n\n")
        file.write("## Scope\n")
        file.write("- Attempt: Try 4\n")
        file.write("- Goal: expand sentence pairs from `train.csv` beyond the original 6,052 clean pairs\n")
        file.write("- Strategy: keep the original baseline intact, then add only new high-confidence pairs from a second sentence-splitting pass\n\n")

        file.write("## Techniques Used\n")
        file.write("- Preserved the original newline/period baseline split as the backbone corpus\n")
        file.write("- Applied transliteration normalization before the second pass (`ḫ/h`, `sz/š`, gap cleanup)\n")
        file.write("- Split English documents into sentence-like units using punctuation anchors\n")
        file.write("- Allocated Akkadian source spans proportionally based on English sentence lengths\n")
        file.write("- Appended leftover Akkadian tokens to the last chunk to avoid truncation\n")
        file.write("- Re-filtered new pairs using minimum token thresholds, source-target length ratio, punctuation/numeric noise checks, and copy-noise detection\n")
        file.write("- Added only pairs that were genuinely new compared with the original 6,052 baseline\n\n")

        file.write("## Results\n")
        file.write(f"- Original clean baseline pairs: {len(baseline)}\n")
        file.write(f"- Additional high-confidence Try 4 pairs: {len(extras)}\n")
        file.write(f"- Final expanded training pairs: {len(full_pairs)}\n")
        file.write(f"- Runtime: {elapsed:.2f} seconds\n\n")

        file.write("## Sample Added Pairs\n")
        for src, tgt in extras[:10]:
            file.write(f"- SRC: {src}\n")
            file.write(f"- TGT: {tgt}\n")


def main() -> None:
    start = time.time()
    OUTPUT_LOG.write_text("", encoding="utf-8")
    train_path = RAW_DATA_DIR / "train.csv"

    log("TRY 4: Expanding train.csv sentence pairs")
    log("Phase 1: rebuilding original clean baseline")
    baseline = baseline_pairs(train_path)
    baseline_set = set(baseline)
    log(f"  baseline pairs={len(baseline)}")

    log("Phase 2: generating extra pairs via English-anchored proportional splitting")
    extras = heuristic_extra_pairs(train_path, baseline_set)
    log(f"  additional pairs={len(extras)}")

    elapsed = time.time() - start
    write_outputs(baseline, extras, elapsed)
    log(f"TRY 4 COMPLETE: final expanded pair count={len(baseline) + len(extras)} in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
