from __future__ import annotations

import csv
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = REPO_ROOT / "data" / "raw"
TRAIN_DIR = REPO_ROOT / "train_folder"
TRAIN_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = TRAIN_DIR / "try3_1_extracted.csv"
OUTPUT_MD = TRAIN_DIR / "try3_1_read.md"
OUTPUT_LOG = TRAIN_DIR / "try3_1_process.log"

COMMON_ENGLISH = {
    "the", "and", "to", "of", "in", "for", "that", "with", "is", "was", "on", "as",
    "he", "she", "it", "they", "his", "her", "from", "this", "by", "be", "or", "an",
}


def log(message: str) -> None:
    print(message, flush=True)
    with OUTPUT_LOG.open("a", encoding="utf-8") as file:
        file.write(message + "\n")


def cleaned_lines(path: Path):
    with path.open("r", encoding="utf-8", errors="replace", newline="") as file:
        for line in file:
            yield line.replace("\x00", "")


def clean_text(text: str) -> str:
    text = str(text).replace("\x00", " ").replace("\ufeff", " ")
    return re.sub(r"\s+", " ", text.lower()).strip()


def split_transliteration(text: str) -> list[str]:
    return [clean_text(part) for part in re.split(r"[\n\.]", str(text)) if clean_text(part)]


def lexical_tokens(text: str) -> list[str]:
    tokens = []
    for token in clean_text(text).split():
        token = re.sub(r"^[^\w<]+|[^\w>]+$", "", token)
        if token:
            tokens.append(token)
    return tokens


def render_progress(current: int, total: int, start_time: float) -> str:
    width = 24
    ratio = current / total if total else 1.0
    filled = min(width, int(width * ratio))
    bar = "#" * filled + "-" * (width - filled)
    elapsed = max(time.time() - start_time, 1e-9)
    rate = current / elapsed
    remaining = (total - current) / rate if rate > 0 else 0.0
    return f"[{bar}] {current}/{total} ({ratio * 100:5.1f}%) elapsed={elapsed:6.1f}s eta={remaining:6.1f}s"


def english_like(text: str) -> bool:
    words = re.findall(r"[a-z]+", text.lower())
    if len(words) < 3:
        return False
    stopword_hits = sum(word in COMMON_ENGLISH for word in words)
    alpha_chars = [char for char in text if not char.isspace()]
    alpha_ratio = sum(char.isalpha() for char in alpha_chars) / len(alpha_chars) if alpha_chars else 0.0
    return stopword_hits >= 1 and alpha_ratio >= 0.55


def looks_sentence_like(text: str) -> bool:
    words = text.split()
    if not (3 <= len(words) <= 55):
        return False
    if english_like(text):
        return True
    if '"' in text or "'" in text or "“" in text or "”" in text:
        return True
    alpha_chars = [char for char in text if not char.isspace()]
    alpha_ratio = sum(char.isalpha() for char in alpha_chars) / len(alpha_chars) if alpha_chars else 0.0
    return alpha_ratio >= 0.60


def similarity_score(left: str, right: str) -> float:
    left_tokens = set(lexical_tokens(left))
    right_tokens = set(lexical_tokens(right))
    overlap = len(left_tokens & right_tokens) / len(left_tokens | right_tokens) if left_tokens and right_tokens else 0.0
    char_ratio = SequenceMatcher(None, left, right).ratio()
    return 0.55 * char_ratio + 0.45 * overlap


@dataclass
class LearnedStructure:
    common_sentence_starters: set[str]
    frequent_tokens: set[str]
    min_length: int
    max_length: int
    avg_length: float
    detect_min: int
    detect_max: int


def learn_akkadian_structure(train_path: Path) -> LearnedStructure:
    first_words = Counter()
    token_freq = Counter()
    lengths = []

    with train_path.open("r", encoding="utf-8", errors="replace", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            for sentence in split_transliteration(row.get("transliteration", "")):
                tokens = lexical_tokens(sentence)
                if not tokens:
                    continue
                first_words[tokens[0]] += 1
                token_freq.update(tokens)
                lengths.append(len(tokens))

    ordered = sorted(lengths)
    return LearnedStructure(
        common_sentence_starters={token for token, _ in first_words.most_common(75)},
        frequent_tokens={token for token, _ in token_freq.most_common(750)},
        min_length=min(lengths),
        max_length=max(lengths),
        avg_length=statistics.mean(lengths),
        detect_min=ordered[int(0.03 * len(ordered))],
        detect_max=ordered[int(0.97 * len(ordered))],
    )


def build_lexicon_forms() -> set[str]:
    forms = set()
    for file_name, column in [("OA_Lexicon_eBL.csv", "form"), ("eBL_Dictionary.csv", "word")]:
        with (RAW_DATA_DIR / file_name).open("r", encoding="utf-8", errors="replace", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                value = clean_text(row.get(column, ""))
                if value:
                    forms.add(value)
                    forms.update(lexical_tokens(value))
    return forms


def detect_akkadian(line: str, structure: LearnedStructure, lexicon_forms: set[str]) -> tuple[bool, dict[str, float]]:
    tokens = lexical_tokens(line)
    if not tokens:
        return False, {"score": 0.0}

    score = 0
    if any("-" in token for token in tokens):
        score += 2

    coverage = sum(token in lexicon_forms for token in tokens) / len(tokens)
    if coverage >= 0.40:
        score += 2

    if structure.detect_min <= len(tokens) <= min(structure.detect_max, 40):
        score += 1

    if tokens[0] in structure.common_sentence_starters:
        score += 1

    if any(token in structure.frequent_tokens for token in tokens[:3]):
        score += 1

    return score >= 4, {"score": score, "coverage": coverage, "length": len(tokens)}


def build_reference_assets() -> tuple[dict[str, list[str]], dict[str, list[str]], list[str]]:
    alias_map: dict[str, list[str]] = defaultdict(list)
    first_token_index: dict[str, list[str]] = defaultdict(list)
    reference_lines: list[str] = []

    with (RAW_DATA_DIR / "published_texts.csv").open("r", encoding="utf-8", errors="replace", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            transliteration = row.get("transliteration", "")
            aliases = row.get("aliases", "") or ""

            split_lines = split_transliteration(transliteration)
            for ref_line in split_lines:
                tokens = lexical_tokens(ref_line)
                if not tokens:
                    continue
                reference_lines.append(ref_line)
                first_token_index[tokens[0]].append(ref_line)

            for alias in aliases.split("|"):
                alias_clean = clean_text(alias)
                if alias_clean and split_lines:
                    alias_map[alias_clean].extend(split_lines)

    return alias_map, first_token_index, reference_lines


def normalize_candidate_spacing(line: str, preferred_forms: set[str]) -> str:
    tokens = line.split()
    normalized = []
    idx = 0
    while idx < len(tokens):
        if idx + 1 < len(tokens):
            merged = f"{tokens[idx]}-{tokens[idx + 1]}"
            if merged in preferred_forms:
                normalized.append(merged)
                idx += 2
                continue
        normalized.append(tokens[idx])
        idx += 1
    return clean_text(" ".join(normalized))


def choose_best_reference(
    candidate: str,
    page_text_clean: str,
    alias_map: dict[str, list[str]],
    first_token_index: dict[str, list[str]],
    reference_lines: list[str],
) -> tuple[bool, bool, float, str]:
    tokens = lexical_tokens(candidate)
    if not tokens:
        return False, False, 0.0, ""

    alias_match = False
    candidate_pool: list[str] = []
    for alias, refs in alias_map.items():
        if alias in page_text_clean:
            alias_match = True
            candidate_pool.extend(refs)

    candidate_pool.extend(first_token_index.get(tokens[0], []))
    candidate_pool.extend(reference_lines[:3000])

    deduped_pool = []
    seen = set()
    target_len = len(tokens)
    for ref in candidate_pool:
        if ref in seen:
            continue
        seen.add(ref)
        if abs(len(lexical_tokens(ref)) - target_len) <= 18:
            deduped_pool.append(ref)

    if len(deduped_pool) > 500:
        deduped_pool = deduped_pool[:500]

    best_score = 0.0
    best_ref = ""
    for ref in deduped_pool:
        score = similarity_score(candidate, ref)
        if score > best_score:
            best_score = score
            best_ref = ref

    matched = best_score >= 0.58 or (alias_match and best_score >= 0.46)
    return matched, alias_match, best_score, best_ref


def collect_translation_candidates(lines: list[str], pivot: int) -> list[tuple[int, str]]:
    start = max(0, pivot - 5)
    end = min(len(lines), pivot + 6)
    candidates = []
    for idx in range(start, end):
        if idx == pivot:
            continue
        line = lines[idx]
        if looks_sentence_like(line):
            candidates.append((idx, line))
    return candidates


def extract_try3_1() -> dict[str, object]:
    start_time = time.time()
    OUTPUT_LOG.write_text("", encoding="utf-8")

    log("TRY 3.1: Starting refined anchor extraction pipeline")
    log("PHASE 1: Learning Akkadian structure from train.csv")
    structure = learn_akkadian_structure(RAW_DATA_DIR / "train.csv")
    log(
        f"  learned range={structure.min_length}..{structure.max_length} "
        f"(avg={structure.avg_length:.2f}, detect_window={structure.detect_min}..{structure.detect_max})"
    )

    log("PHASE 2: Building lexicon-backed detection system")
    lexicon_forms = build_lexicon_forms()
    lexicon_forms.update(structure.frequent_tokens)
    log(f"  lexicon-backed tokens/forms={len(lexicon_forms)}")

    log("PHASE 3: Building sentence-level reference anchors from published_texts.csv")
    alias_map, first_token_index, reference_lines = build_reference_assets()
    log(f"  alias keys={len(alias_map)}")
    log(f"  reference lines={len(reference_lines)}")

    preferred_forms = lexicon_forms | structure.frequent_tokens
    extracted_rows = []
    seen_pairs = set()
    metrics = Counter()
    total_rows = 31286
    processed_true_rows = 0
    last_progress = time.time()

    reader = csv.DictReader(cleaned_lines(RAW_DATA_DIR / "publications.csv"))
    for row in reader:
        if clean_text(row.get("has_akkadian", "")) != "true":
            continue

        processed_true_rows += 1
        page_text = str(row.get("page_text", ""))
        page_text_clean = clean_text(page_text)
        lines = [clean_text(line) for line in page_text.split("\n") if clean_text(line)]

        for idx, line in enumerate(lines):
            normalized = normalize_candidate_spacing(line, preferred_forms)
            is_akkadian, detect_meta = detect_akkadian(normalized, structure, lexicon_forms)
            if not is_akkadian:
                continue

            metrics["akkadian_candidates"] += 1
            matched_reference, alias_match, similarity, best_ref = choose_best_reference(
                normalized,
                page_text_clean,
                alias_map,
                first_token_index,
                reference_lines,
            )
            if not matched_reference:
                continue

            metrics["reference_matches"] += 1
            translation_candidates = collect_translation_candidates(lines, idx)
            best_translation = None
            best_distance = None

            for t_idx, candidate in translation_candidates:
                if not english_like(candidate):
                    continue
                distance = abs(t_idx - idx)
                if distance > 4:
                    continue
                if best_translation is None or distance < best_distance:
                    best_translation = candidate
                    best_distance = distance

            if not best_translation:
                continue

            score = 0
            if alias_match:
                score += 5
            if matched_reference:
                score += 3
            if best_distance is not None and best_distance <= 4:
                score += 2
            if english_like(best_translation):
                score += 2
            if 3 <= len(lexical_tokens(normalized)) <= 45 and 3 <= len(best_translation.split()) <= 55:
                score += 1

            if score < 7:
                continue

            pair = (normalized.strip().lower(), best_translation.strip().lower())
            if pair in seen_pairs:
                continue
            if len(pair[0]) < 5 or len(pair[1]) < 5 or len(pair[0]) > 500 or len(pair[1]) > 500:
                continue

            seen_pairs.add(pair)
            extracted_rows.append(
                {
                    "akkadian": pair[0],
                    "english": pair[1],
                    "score": score,
                    "distance": best_distance,
                    "reference_similarity": round(similarity, 4),
                    "reference": best_ref,
                    "alias_match": alias_match,
                    "pdf_name": row.get("\ufeffpdf_name", ""),
                    "page": row.get("page", ""),
                }
            )
            metrics["kept_pairs"] += 1

        now = time.time()
        if processed_true_rows == 1 or processed_true_rows == total_rows or now - last_progress >= 2:
            log(render_progress(processed_true_rows, total_rows, start_time))
            last_progress = now

    extracted_rows.sort(key=lambda row: (-row["score"], row["distance"], -row["reference_similarity"]))

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "akkadian",
                "english",
                "score",
                "distance",
                "reference_similarity",
                "reference",
                "alias_match",
                "pdf_name",
                "page",
            ],
        )
        writer.writeheader()
        writer.writerows(extracted_rows)

    elapsed = time.time() - start_time
    with OUTPUT_MD.open("w", encoding="utf-8") as file:
        file.write("# Try 3.1 OCR Extraction Log\n\n")
        file.write("## Scope\n")
        file.write("- Attempt: Try 3.1\n")
        file.write("- Goal: relax the anchor stage from Try 3 without abandoning precision\n")
        file.write("- Core refinement: compare OCR candidates against sentence-split reference lines instead of full transliterations\n\n")
        file.write("## Key Changes From Try 3\n")
        file.write("- Lowered lexicon coverage gate from ultra-strict to moderate (`>= 40%`)\n")
        file.write("- Indexed sentence-level references from `published_texts.csv`\n")
        file.write("- Allowed alias-backed weaker similarity matches when the page anchor is strong\n")
        file.write("- Expanded translation search window to `+-5` lines, while keeping nearest English-valid line only\n")
        file.write("- Preserved precision-first final confidence threshold\n\n")
        file.write("## Metrics\n")
        file.write(f"- Pages scanned with `has_akkadian == true`: {processed_true_rows}\n")
        file.write(f"- Akkadian candidates detected: {metrics['akkadian_candidates']}\n")
        file.write(f"- Reference matches retained: {metrics['reference_matches']}\n")
        file.write(f"- Final extracted pairs: {len(extracted_rows)}\n")
        file.write(f"- Runtime: {elapsed:.2f} seconds\n")
        file.write("- Multilingual note: this runtime still keeps only English-valid nearby lines because automatic translation is not available locally.\n\n")
        file.write("## Sample Pairs\n")
        for row in extracted_rows[:5]:
            file.write(f"- SRC: {row['akkadian']}\n")
            file.write(f"- TGT: {row['english']}\n")
            file.write(
                f"  score={row['score']}, distance={row['distance']}, similarity={row['reference_similarity']}, alias_match={row['alias_match']}\n"
            )

    return {"total_pairs": len(extracted_rows), "elapsed_seconds": elapsed, "samples": extracted_rows[:5]}


if __name__ == "__main__":
    summary = extract_try3_1()
    log(f"TRY 3.1 COMPLETE: extracted {summary['total_pairs']} pairs in {summary['elapsed_seconds']:.2f}s")
