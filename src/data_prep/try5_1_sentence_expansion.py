from __future__ import annotations

import csv
import difflib
import re
import statistics
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = REPO_ROOT / "train_folder"

INPUT_ARCHIVE = TRAIN_DIR / "try5_train_plus.csv"
BASELINE_SRC = TRAIN_DIR / "train.src"
BASELINE_TGT = TRAIN_DIR / "train.tgt"

OUTPUT_SRC = TRAIN_DIR / "try5_1_train.src"
OUTPUT_TGT = TRAIN_DIR / "try5_1_train.tgt"
OUTPUT_ADDED = TRAIN_DIR / "try5_1_added_only.csv"
OUTPUT_READ = TRAIN_DIR / "try5_1_read.md"
OUTPUT_LOG = TRAIN_DIR / "try5_1_process.log"

COMMON_ENGLISH = {
    "the", "and", "to", "of", "in", "for", "that", "with", "is", "was", "on", "as",
    "he", "she", "it", "they", "his", "her", "from", "this", "by", "be", "or", "an",
    "you", "we", "i", "my", "your", "their", "will", "if", "not", "have", "has",
    "do", "are", "me", "this", "at", "all", "here", "there", "then", "when",
}
TRANSLIT_RE = re.compile(r"[šṣṭḫāīū]|[a-z]+-[a-z]+", re.IGNORECASE)


def log(message: str) -> None:
    print(message, flush=True)
    with OUTPUT_LOG.open("a", encoding="utf-8") as file:
        file.write(message + "\n")


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def english_sentence_split(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z"“])', text)
    parts = [part.strip() for part in parts if part.strip()]
    return parts or [text]


def only_punct_or_nums(text: str) -> bool:
    return not bool(re.search(r"[a-zšṣṭḫāīū]", text, re.IGNORECASE))


def detect_noise(src: str, tgt: str) -> bool:
    if len(tgt) < 3 or len(src) == 0:
        return True
    ratio = len(tgt) / len(src)
    if ratio < 0.25 or ratio > 3.2:
        return True
    if difflib.SequenceMatcher(None, src, tgt).ratio() > 0.80:
        return True
    return False


def english_quality_score(text: str) -> int:
    text = clean_text(text)
    words = re.findall(r"[a-z]+", text)
    stop_hits = sum(word in COMMON_ENGLISH for word in words)
    chars = [char for char in text if not char.isspace()]
    alpha_ratio = sum(char.isalpha() for char in chars) / len(chars) if chars else 0.0
    score = 0
    if stop_hits >= 2:
        score += 2
    if stop_hits >= 4:
        score += 1
    if alpha_ratio >= 0.78:
        score += 2
    if text.endswith((".", "!", "?")):
        score += 1
    if '"' not in text or text.count('"') % 2 == 0:
        score += 1
    if ":" in text:
        score -= 1
    if re.search(r"[^\x00-\x7Fšṣṭḫāīū\"'.,;:!?() \-]", text):
        score -= 3
    return score


def translit_quality_score(text: str) -> int:
    text = clean_text(text)
    tokens = text.split()
    hyphen_tokens = sum("-" in token for token in tokens)
    score = 0
    if TRANSLIT_RE.search(text):
        score += 2
    if hyphen_tokens >= 2:
        score += 2
    if hyphen_tokens / max(1, len(tokens)) >= 0.25:
        score += 1
    if "." not in text:
        score += 1
    return score


def load_baseline() -> list[tuple[str, str]]:
    src_lines = BASELINE_SRC.read_text(encoding="utf-8").splitlines()
    tgt_lines = BASELINE_TGT.read_text(encoding="utf-8").splitlines()
    return [(clean_text(src), clean_text(tgt)) for src, tgt in zip(src_lines, tgt_lines)]


def baseline_stats(baseline: list[tuple[str, str]]) -> dict[str, float]:
    src_lens = sorted(len(src.split()) for src, _ in baseline)
    tgt_lens = sorted(len(tgt.split()) for _, tgt in baseline)
    ratios = sorted((len(tgt.split()) / max(1, len(src.split()))) for src, tgt in baseline)
    n = len(baseline)
    return {
        "src_min": src_lens[int(0.15 * n)],
        "src_max": src_lens[int(0.85 * n)],
        "tgt_min": tgt_lens[int(0.15 * n)],
        "tgt_max": tgt_lens[int(0.85 * n)],
        "ratio_med": statistics.median(ratios),
    }


def confidence_score(src: str, tgt: str, stats: dict[str, float]) -> tuple[int, dict[str, float]]:
    src_len = len(src.split())
    tgt_len = len(tgt.split())
    ratio = tgt_len / max(1, src_len)

    score = translit_quality_score(src) + english_quality_score(tgt)
    if stats["src_min"] <= src_len <= stats["src_max"]:
        score += 2
    if stats["tgt_min"] <= tgt_len <= stats["tgt_max"]:
        score += 2
    if abs(ratio - stats["ratio_med"]) <= 0.6:
        score += 3
    elif abs(ratio - stats["ratio_med"]) <= 1.0:
        score += 1

    if src_len < 5:
        score -= 3
    if tgt_len < 4:
        score -= 3
    if src_len > 35:
        score -= 1
    if tgt_len > 35:
        score -= 1

    return score, {"src_len": src_len, "tgt_len": tgt_len, "ratio": ratio}


def proportional_chunks(src_text: str, tgt_sentences: list[str]) -> list[tuple[str, str]]:
    src_words = src_text.split()
    total_src = len(src_words)
    total_tgt_chars = max(1, sum(len(sentence) for sentence in tgt_sentences))
    cursor = 0
    chunks: list[tuple[str, str]] = []

    for sentence in tgt_sentences:
        proportional_len = max(1, int(round(total_src * (len(sentence) / total_tgt_chars))))
        end = min(cursor + proportional_len, total_src)
        chunks.append((" ".join(src_words[cursor:end]), sentence))
        cursor = end

    if cursor < total_src and chunks:
        last_src, last_tgt = chunks[-1]
        chunks[-1] = ((last_src + " " + " ".join(src_words[cursor:])).strip(), last_tgt)

    return chunks


def build_try5_1_pairs() -> tuple[list[tuple[str, str]], list[dict[str, object]], dict[str, int]]:
    baseline = load_baseline()
    baseline_set = set(baseline)
    stats = baseline_stats(baseline)

    with INPUT_ARCHIVE.open("r", encoding="utf-8", newline="") as file:
        archive_rows = list(csv.DictReader(file))

    extras: list[dict[str, object]] = []
    total_candidates = 0

    log(f"TRY 5.1: Splitting {len(archive_rows)} archive-level pairs into sentence additions")
    for index, row in enumerate(archive_rows, start=1):
        sentences = english_sentence_split(row["translation"])
        chunks = proportional_chunks(row["transliteration"], sentences)
        progress = f"[{index:02d}/{len(archive_rows):02d}]"
        log(f"{progress} {row['match_key']}: {len(sentences)} english sentences -> {len(chunks)} aligned chunks")

        for chunk_src, chunk_tgt in chunks:
            total_candidates += 1
            src_clean = clean_text(chunk_src)
            tgt_clean = clean_text(chunk_tgt)
            pair = (src_clean, tgt_clean)

            if pair in baseline_set:
                continue
            if only_punct_or_nums(src_clean) or only_punct_or_nums(tgt_clean):
                continue
            if detect_noise(src_clean, tgt_clean):
                continue

            score, meta = confidence_score(src_clean, tgt_clean, stats)
            if score >= 16 and meta["src_len"] >= 5 and meta["tgt_len"] >= 4:
                extras.append(
                    {
                        "oare_id": row["oare_id"],
                        "source": row["source"],
                        "match_key": row["match_key"],
                        "akkadian": src_clean,
                        "english": tgt_clean,
                        "confidence": score,
                        "src_len": meta["src_len"],
                        "tgt_len": meta["tgt_len"],
                        "ratio": round(meta["ratio"], 3),
                    }
                )

    deduped: list[dict[str, object]] = []
    seen = set()
    for row in sorted(
        extras,
        key=lambda item: (-int(item["confidence"]), abs(float(item["ratio"]) - stats["ratio_med"]), int(item["src_len"])),
    ):
        pair = (str(row["akkadian"]), str(row["english"]))
        if pair in seen:
            continue
        seen.add(pair)
        deduped.append(row)

    summary = {
        "archive_pairs": len(archive_rows),
        "candidate_chunks": total_candidates,
        "kept_pairs": len(deduped),
    }
    return baseline, deduped, summary


def write_outputs(baseline: list[tuple[str, str]], extras: list[dict[str, object]], summary: dict[str, int], elapsed: float) -> None:
    full_pairs = baseline + [(str(row["akkadian"]), str(row["english"])) for row in extras]

    OUTPUT_SRC.write_text("\n".join(src for src, _ in full_pairs) + "\n", encoding="utf-8")
    OUTPUT_TGT.write_text("\n".join(tgt for _, tgt in full_pairs) + "\n", encoding="utf-8")

    with OUTPUT_ADDED.open("w", encoding="utf-8", newline="") as csv_file:
        fieldnames = ["oare_id", "source", "match_key", "akkadian", "english", "confidence", "src_len", "tgt_len", "ratio"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(extras)

    sample_rows = extras[:5]
    with OUTPUT_READ.open("w", encoding="utf-8") as file:
        file.write("# Try 5.1 Sentence Expansion\n\n")
        file.write("## Scope\n")
        file.write("- Attempt: Try 5.1\n")
        file.write("- Input: archive-level metadata matches from `try5_train_plus.csv`\n")
        file.write("- Goal: convert the 18 document-level additions into cleaner sentence-level training pairs\n\n")
        file.write("## Techniques Used\n")
        file.write("- Split the English side into sentence-like units using punctuation boundaries\n")
        file.write("- Allocated Akkadian chunks proportionally by English sentence length\n")
        file.write("- Scored each candidate chunk against the original 6,052-pair baseline distribution\n")
        file.write("- Kept only chunks that passed transliteration quality, English quality, and length-ratio checks\n")
        file.write("- Stored the additions separately before mixing them into the merged corpus\n\n")
        file.write("## Results\n")
        file.write(f"- Archive-level input pairs: {summary['archive_pairs']}\n")
        file.write(f"- Sentence candidates considered: {summary['candidate_chunks']}\n")
        file.write(f"- New sentence-level additions kept: {summary['kept_pairs']}\n")
        file.write(f"- Final merged total (baseline + Try 5.1): {len(full_pairs)}\n")
        file.write(f"- Runtime: {elapsed:.2f} seconds\n\n")
        if sample_rows:
            file.write("## Sample Pairs\n")
            for row in sample_rows:
                file.write(f"- KEY: {row['match_key']} (confidence={row['confidence']})\n")
                file.write(f"  akkadian={row['akkadian']}\n")
                file.write(f"  english={row['english']}\n")


def main() -> None:
    start = time.time()
    OUTPUT_LOG.write_text("", encoding="utf-8")
    baseline, extras, summary = build_try5_1_pairs()
    elapsed = time.time() - start
    write_outputs(baseline, extras, summary, elapsed)
    log(
        "TRY 5.1 COMPLETE: "
        f"archive_pairs={summary['archive_pairs']}, "
        f"candidate_chunks={summary['candidate_chunks']}, "
        f"kept_pairs={summary['kept_pairs']}, "
        f"runtime={elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
