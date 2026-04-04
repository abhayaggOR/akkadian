from __future__ import annotations

import csv
import difflib
import re
import statistics
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = REPO_ROOT / "data" / "raw"
TRAIN_DIR = REPO_ROOT / "train_folder"
TRAIN_DIR.mkdir(exist_ok=True)

OUTPUT_SRC = TRAIN_DIR / "try4_2_train.src"
OUTPUT_TGT = TRAIN_DIR / "try4_2_train.tgt"
OUTPUT_ADDED = TRAIN_DIR / "try4_2_added_only.csv"
OUTPUT_READ = TRAIN_DIR / "try4_2_read.md"
OUTPUT_LOG = TRAIN_DIR / "try4_2_process.log"

COMMON_ENGLISH = {
    "the", "and", "to", "of", "in", "for", "that", "with", "is", "was", "on", "as",
    "he", "she", "it", "they", "his", "her", "from", "this", "by", "be", "or", "an",
    "you", "we", "i", "my", "your", "their", "will", "if", "not", "have", "has",
}
TRANSLIT_RE = re.compile(r"[šṣṭḫāīū]|[a-z]+-[a-z]+", re.IGNORECASE)


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
    if len(tgt) < 3 or len(src) == 0:
        return True
    ratio = len(tgt) / len(src)
    if ratio < 0.25 or ratio > 3.2:
        return True
    similarity = difflib.SequenceMatcher(None, src, tgt).ratio()
    if similarity > 0.80:
        return True
    return False


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


def baseline_stats(baseline: list[tuple[str, str]]) -> dict[str, float]:
    src_lens = sorted(len(src.split()) for src, _ in baseline)
    tgt_lens = sorted(len(tgt.split()) for _, tgt in baseline)
    ratios = sorted((len(tgt.split()) / max(1, len(src.split()))) for src, tgt in baseline)
    n = len(baseline)
    return {
        "src_med": statistics.median(src_lens),
        "tgt_med": statistics.median(tgt_lens),
        "src_min": src_lens[int(0.15 * n)],
        "src_max": src_lens[int(0.85 * n)],
        "tgt_min": tgt_lens[int(0.15 * n)],
        "tgt_max": tgt_lens[int(0.85 * n)],
        "ratio_min": ratios[int(0.15 * n)],
        "ratio_max": ratios[int(0.85 * n)],
        "ratio_med": statistics.median(ratios),
    }


def english_quality_score(text: str) -> int:
    text = clean_text(text)
    words = re.findall(r"[a-z]+", text)
    score = 0
    stop_hits = sum(word in COMMON_ENGLISH for word in words)
    alpha_chars = [char for char in text if not char.isspace()]
    alpha_ratio = sum(char.isalpha() for char in alpha_chars) / len(alpha_chars) if alpha_chars else 0.0
    if stop_hits >= 2:
        score += 2
    if stop_hits >= 3:
        score += 1
    if alpha_ratio >= 0.80:
        score += 2
    if text.endswith((".", "!", "?")):
        score += 1
    if '"' not in text or text.count('"') % 2 == 0:
        score += 1
    if "<gap>" in text:
        score -= 2
    if ":" in text:
        score -= 1
    if re.search(r"[^\x00-\x7Fšṣṭḫāīū<>\"'.,;:!?() -]", text):
        score -= 3
    if text.count('""') > 0:
        score -= 3
    return score


def translit_quality_score(text: str) -> int:
    text = clean_text(text)
    tokens = text.split()
    score = 0
    if TRANSLIT_RE.search(text):
        score += 2
    hyphen_tokens = sum("-" in token for token in tokens)
    if hyphen_tokens >= 2:
        score += 2
    if hyphen_tokens / max(1, len(tokens)) >= 0.30:
        score += 1
    if not text.startswith("<gap>") and not text.endswith("<gap>"):
        score += 1
    if sum(token == "<gap>" for token in tokens) == 0:
        score += 1
    if "." not in text:
        score += 1
    return score


def confidence_score(src: str, tgt: str, stats: dict[str, float]) -> tuple[int, dict[str, float]]:
    src = clean_text(src)
    tgt = clean_text(tgt)
    src_len = len(src.split())
    tgt_len = len(tgt.split())
    ratio = tgt_len / max(1, src_len)

    score = 0
    score += translit_quality_score(src)
    score += english_quality_score(tgt)

    if stats["src_min"] <= src_len <= stats["src_max"]:
        score += 2
    if stats["tgt_min"] <= tgt_len <= stats["tgt_max"]:
        score += 2
    if abs(ratio - stats["ratio_med"]) <= 0.5:
        score += 3
    elif abs(ratio - stats["ratio_med"]) <= 1.0:
        score += 1

    if src_len < 6:
        score -= 2
    if tgt_len < 5:
        score -= 2
    if tgt_len > 24:
        score -= 1

    return score, {"src_len": src_len, "tgt_len": tgt_len, "ratio": ratio}


def generate_highconf_extras(train_path: Path, baseline_set: set[tuple[str, str]], stats: dict[str, float]) -> list[dict[str, object]]:
    candidates = []
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
                if only_punct_or_nums(src_clean) or only_punct_or_nums(tgt_clean):
                    continue
                if detect_noise(src_clean, tgt_clean):
                    continue

                score, meta = confidence_score(src_clean, tgt_clean, stats)
                if meta["src_len"] >= 6 and meta["tgt_len"] >= 5 and score >= 15:
                    candidates.append(
                        {
                            "akkadian": src_clean,
                            "english": tgt_clean,
                            "confidence": score,
                            "src_len": meta["src_len"],
                            "tgt_len": meta["tgt_len"],
                            "ratio": round(meta["ratio"], 3),
                        }
                    )

    deduped = []
    seen = set()
    for row in sorted(candidates, key=lambda item: (-item["confidence"], abs(item["ratio"] - stats["ratio_med"]), item["src_len"])):
        pair = (row["akkadian"], row["english"])
        if pair in seen:
            continue
        seen.add(pair)
        deduped.append(row)
    return deduped


def write_outputs(baseline: list[tuple[str, str]], extras: list[dict[str, object]], elapsed: float, stats: dict[str, float]) -> None:
    full_pairs = baseline + [(row["akkadian"], row["english"]) for row in extras]
    with OUTPUT_SRC.open("w", encoding="utf-8") as src_file:
        src_file.write("\n".join(src for src, _ in full_pairs) + "\n")
    with OUTPUT_TGT.open("w", encoding="utf-8") as tgt_file:
        tgt_file.write("\n".join(tgt for _, tgt in full_pairs) + "\n")

    with OUTPUT_ADDED.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["akkadian", "english", "confidence", "src_len", "tgt_len", "ratio"])
        writer.writeheader()
        writer.writerows(extras)

    with OUTPUT_READ.open("w", encoding="utf-8") as file:
        file.write("# Try 4.2 High-Confidence Expansion Log\n\n")
        file.write("## Scope\n")
        file.write("- Attempt: Try 4.2\n")
        file.write("- Goal: produce a smaller, higher-confidence train.csv expansion set than Try 4.1\n")
        file.write("- Strategy: keep only additions that are close to the core baseline distribution and score strongly on both source and target quality\n\n")

        file.write("## Techniques Used\n")
        file.write("- Reused the baseline 6,052 clean pairs without modification\n")
        file.write("- Reused English-anchored proportional splitting only to propose candidate additions\n")
        file.write("- Tightened the acceptable source/target distribution to the middle band of the baseline corpus\n")
        file.write("- Penalized suspicious short-source / polished-English combinations more aggressively\n")
        file.write("- Required stronger transliteration-like surface patterns and stronger English sentence quality\n")
        file.write("- Kept only high-confidence additions\n\n")

        file.write("## Baseline Core Band\n")
        file.write(f"- Source token band: {stats['src_min']}..{stats['src_max']}\n")
        file.write(f"- Target token band: {stats['tgt_min']}..{stats['tgt_max']}\n")
        file.write(f"- Ratio band: {stats['ratio_min']:.3f}..{stats['ratio_max']:.3f}\n")
        file.write(f"- Median ratio: {stats['ratio_med']:.3f}\n\n")

        file.write("## Results\n")
        file.write(f"- Original clean baseline pairs: {len(baseline)}\n")
        file.write(f"- Additional high-confidence Try 4.2 pairs: {len(extras)}\n")
        file.write(f"- Final expanded training pairs: {len(full_pairs)}\n")
        file.write(f"- Runtime: {elapsed:.2f} seconds\n\n")

        file.write("## Sample Added Pairs\n")
        for row in extras[:10]:
            file.write(f"- SRC: {row['akkadian']}\n")
            file.write(f"- TGT: {row['english']}\n")
            file.write(f"  confidence={row['confidence']}, src_len={row['src_len']}, tgt_len={row['tgt_len']}, ratio={row['ratio']}\n")


def main() -> None:
    start = time.time()
    OUTPUT_LOG.write_text("", encoding="utf-8")
    train_path = RAW_DATA_DIR / "train.csv"

    log("TRY 4.2: Building high-confidence train.csv expansion")
    baseline = baseline_pairs(train_path)
    baseline_set = set(baseline)
    stats = baseline_stats(baseline)
    log(f"  baseline pairs={len(baseline)}")

    extras = generate_highconf_extras(train_path, baseline_set, stats)
    log(f"  high-confidence additional pairs={len(extras)}")

    elapsed = time.time() - start
    write_outputs(baseline, extras, elapsed, stats)
    log(f"TRY 4.2 COMPLETE: final expanded pair count={len(baseline) + len(extras)} in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
