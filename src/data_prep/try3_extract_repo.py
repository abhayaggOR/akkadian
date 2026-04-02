from __future__ import annotations

import csv
import math
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

OUTPUT_CSV = TRAIN_DIR / "try3_extracted.csv"
OUTPUT_MD = TRAIN_DIR / "try3_read.md"
OUTPUT_LOG = TRAIN_DIR / "try3_process.log"


COMMON_ENGLISH = {
    "the", "and", "to", "of", "in", "for", "that", "with", "is", "was", "on", "as",
    "he", "she", "it", "they", "his", "her", "from", "this", "by", "be", "or", "an",
}
GERMAN_HINTS = {"der", "die", "das", "und", "ist", "mit", "nicht", "ein", "eine", "zu", "von", "den"}
FRENCH_HINTS = {"le", "la", "les", "de", "des", "et", "est", "dans", "une", "un", "pour", "que"}
SPANISH_HINTS = {"el", "la", "los", "las", "de", "y", "es", "una", "un", "por", "que", "con"}


def log(message: str) -> None:
    print(message, flush=True)
    with OUTPUT_LOG.open("a", encoding="utf-8") as log_file:
        log_file.write(message + "\n")


def cleaned_lines(path: Path):
    with path.open("r", encoding="utf-8", errors="replace", newline="") as file:
        for line in file:
            yield line.replace("\x00", "")


def clean_text(text: str) -> str:
    text = str(text).replace("\x00", " ").replace("\ufeff", " ")
    text = re.sub(r"\s+", " ", text.lower()).strip()
    return text


def split_transliteration(text: str) -> list[str]:
    return [clean_text(part) for part in re.split(r"[\n\.]", str(text)) if clean_text(part)]


def line_word_tokens(text: str) -> list[str]:
    return [token for token in text.split() if token]


def normalized_token(token: str) -> str:
    token = clean_text(token)
    token = re.sub(r"^[^\w<]+|[^\w>]+$", "", token)
    return token


def lexical_tokens(text: str) -> list[str]:
    return [normalized_token(token) for token in line_word_tokens(text) if normalized_token(token)]


def percent(part: int, total: int) -> float:
    return (part / total) * 100 if total else 0.0


def render_progress(current: int, total: int, start_time: float) -> str:
    width = 24
    ratio = current / total if total else 1.0
    filled = min(width, int(width * ratio))
    bar = "#" * filled + "-" * (width - filled)
    elapsed = max(time.time() - start_time, 1e-9)
    rate = current / elapsed
    remaining = (total - current) / rate if rate > 0 else 0.0
    return f"[{bar}] {current}/{total} ({ratio * 100:5.1f}%) elapsed={elapsed:6.1f}s eta={remaining:6.1f}s"


def mostly_alphabetic(text: str) -> bool:
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False
    alpha = sum(c.isalpha() for c in chars)
    return alpha / len(chars) >= 0.55


def looks_sentence_like(text: str) -> bool:
    words = text.split()
    if not (3 <= len(words) <= 45):
        return False
    if '"' in text or "'" in text or "“" in text or "”" in text:
        return True
    if mostly_alphabetic(text):
        return True
    if text.endswith((".", "!", "?", ":", ";")):
        return True
    return False


def english_like(text: str) -> bool:
    words = re.findall(r"[a-z]+", text.lower())
    if len(words) < 3:
        return False
    stopword_hits = sum(word in COMMON_ENGLISH for word in words)
    return stopword_hits >= 1 and (stopword_hits / len(words)) >= 0.06


def detect_language(text: str) -> str:
    words = re.findall(r"[a-zà-ÿ]+", text.lower())
    if len(words) < 3:
        return "unknown"

    def score(hints: set[str]) -> int:
        return sum(word in hints for word in words)

    scores = {
        "en": score(COMMON_ENGLISH),
        "de": score(GERMAN_HINTS),
        "fr": score(FRENCH_HINTS),
        "es": score(SPANISH_HINTS),
    }
    best_lang = max(scores, key=scores.get)
    return best_lang if scores[best_lang] > 0 else "unknown"


def maybe_translate_to_english(text: str) -> tuple[str | None, str]:
    language = detect_language(text)
    if language in {"en", "unknown"}:
        return text if english_like(text) else None, language
    return None, language


def similarity_score(left: str, right: str) -> float:
    ratio = SequenceMatcher(None, left, right).ratio()
    left_tokens = set(lexical_tokens(left))
    right_tokens = set(lexical_tokens(right))
    overlap = len(left_tokens & right_tokens) / len(left_tokens | right_tokens) if left_tokens and right_tokens else 0.0
    return 0.65 * ratio + 0.35 * overlap


def choose_best_reference(candidate: str, reference_index: dict[str, list[str]]) -> tuple[bool, float, str]:
    tokens = lexical_tokens(candidate)
    if not tokens:
        return False, 0.0, ""

    candidate_pool = []
    first = tokens[0]
    candidate_pool.extend(reference_index.get(first, []))
    candidate_pool.extend(reference_index.get("__fallback__", []))

    if len(candidate_pool) > 300:
        target_len = len(tokens)
        candidate_pool = sorted(candidate_pool, key=lambda ref: abs(len(lexical_tokens(ref)) - target_len))[:300]

    best_ref = ""
    best_score = 0.0
    for ref in candidate_pool:
        score = similarity_score(candidate, ref)
        if score > best_score:
            best_score = score
            best_ref = ref

    return best_score >= 0.72, best_score, best_ref


def normalize_candidate_spacing(text: str, preferred_forms: set[str]) -> str:
    tokens = line_word_tokens(text)
    if not tokens:
        return text

    fixed = []
    idx = 0
    while idx < len(tokens):
        if idx + 1 < len(tokens):
            merged = f"{tokens[idx]}-{tokens[idx + 1]}"
            if merged in preferred_forms:
                fixed.append(merged)
                idx += 2
                continue
        fixed.append(tokens[idx])
        idx += 1
    return " ".join(fixed)


@dataclass
class LearnedStructure:
    common_sentence_starters: set[str]
    length_range: dict[str, float]
    frequent_tokens: set[str]
    detection_length_min: int
    detection_length_max: int


def learn_akkadian_structure(train_path: Path) -> LearnedStructure:
    first_word_counter = Counter()
    token_counter = Counter()
    lengths = []

    with train_path.open("r", encoding="utf-8", errors="replace", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            for sentence in split_transliteration(row.get("transliteration", "")):
                tokens = lexical_tokens(sentence)
                if not tokens:
                    continue
                first_word_counter[tokens[0]] += 1
                token_counter.update(tokens)
                lengths.append(len(tokens))

    avg_length = statistics.mean(lengths) if lengths else 0.0
    sorted_lengths = sorted(lengths)
    lower_idx = int(0.05 * len(sorted_lengths))
    upper_idx = min(len(sorted_lengths) - 1, int(0.95 * len(sorted_lengths)))
    detection_min = sorted_lengths[lower_idx]
    detection_max = sorted_lengths[upper_idx]

    return LearnedStructure(
        common_sentence_starters={token for token, _ in first_word_counter.most_common(50)},
        length_range={
            "min_length": min(lengths) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
            "avg_length": avg_length,
        },
        frequent_tokens={token for token, _ in token_counter.most_common(500)},
        detection_length_min=detection_min,
        detection_length_max=detection_max,
    )


def build_lexicon_forms(lexicon_path: Path) -> set[str]:
    forms = set()
    with lexicon_path.open("r", encoding="utf-8", errors="replace", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            form = clean_text(row.get("form", ""))
            if form:
                forms.add(form)
                for token in lexical_tokens(form):
                    forms.add(token)
    return forms


def build_dictionary_lexemes(dictionary_path: Path) -> set[str]:
    lexemes = set()
    with dictionary_path.open("r", encoding="utf-8", errors="replace", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            word = clean_text(row.get("word", ""))
            if word:
                lexemes.update(lexical_tokens(word))
    return lexemes


def detect_akkadian(line: str, structure: LearnedStructure, lexicon_forms: set[str]) -> tuple[bool, dict[str, float]]:
    tokens = lexical_tokens(line)
    if not tokens:
        return False, {"score": 0}

    score = 0
    has_hyphen = any("-" in token for token in tokens)
    if has_hyphen:
        score += 2

    lexicon_hits = sum(token in lexicon_forms for token in tokens)
    coverage = lexicon_hits / len(tokens)
    if coverage >= 0.5:
        score += 2

    length_ok = structure.detection_length_min <= len(tokens) <= structure.detection_length_max
    if length_ok:
        score += 1

    starter_ok = tokens[0] in structure.common_sentence_starters
    if starter_ok:
        score += 1

    return score >= 4, {
        "score": score,
        "coverage": coverage,
        "starter_ok": starter_ok,
        "length_ok": length_ok,
        "has_hyphen": has_hyphen,
    }


def build_reference_assets(published_path: Path) -> tuple[dict[str, list[str]], list[str], dict[str, list[str]]]:
    alias_map: dict[str, list[str]] = defaultdict(list)
    references = []
    reference_index: dict[str, list[str]] = defaultdict(list)

    with published_path.open("r", encoding="utf-8", errors="replace", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            transliteration = clean_text(row.get("transliteration", ""))
            if not transliteration:
                continue

            references.append(transliteration)
            tokens = lexical_tokens(transliteration)
            if tokens:
                reference_index[tokens[0]].append(transliteration)

            alias_str = row.get("aliases", "") or ""
            for alias in alias_str.split("|"):
                alias_clean = clean_text(alias)
                if alias_clean:
                    alias_map[alias_clean].append(transliteration)

    reference_index["__fallback__"] = references[:2000]
    return alias_map, references, reference_index


def collect_translation_candidates(lines: list[str], pivot: int, window: int = 4) -> list[tuple[int, str]]:
    candidates = []
    start = max(0, pivot - window)
    end = min(len(lines), pivot + window + 1)
    for idx in range(start, end):
        if idx == pivot:
            continue
        line = lines[idx]
        if looks_sentence_like(line):
            candidates.append((idx, line))
    return candidates


def extract_try3() -> dict[str, object]:
    start_time = time.time()
    OUTPUT_LOG.write_text("", encoding="utf-8")

    log("TRY 3: Starting linguistically anchored extraction pipeline")
    log("PHASE 1: Learning Akkadian structure from train.csv")
    structure = learn_akkadian_structure(RAW_DATA_DIR / "train.csv")
    log(
        "  learned sentence range="
        f"{structure.length_range['min_length']}..{structure.length_range['max_length']} "
        f"(avg={structure.length_range['avg_length']:.2f}, "
        f"detection_window={structure.detection_length_min}..{structure.detection_length_max})"
    )
    log(f"  common starters tracked={len(structure.common_sentence_starters)}")
    log(f"  frequent tokens tracked={len(structure.frequent_tokens)}")

    log("PHASE 2: Building Akkadian detection assets")
    lexicon_forms = build_lexicon_forms(RAW_DATA_DIR / "OA_Lexicon_eBL.csv")
    dictionary_lexemes = build_dictionary_lexemes(RAW_DATA_DIR / "eBL_Dictionary.csv")
    lexicon_forms.update(dictionary_lexemes)
    lexicon_forms.update(structure.frequent_tokens)
    log(f"  lexicon forms available={len(lexicon_forms)}")

    log("PHASE 3: Building published-text reference anchors")
    alias_map, references, reference_index = build_reference_assets(RAW_DATA_DIR / "published_texts.csv")
    log(f"  reference transliterations={len(references)}")
    log(f"  aliases indexed={len(alias_map)}")

    log("PHASE 4-10: Streaming publications.csv with progress")
    extracted_rows = []
    seen_pairs = set()
    metrics = Counter()
    preferred_forms = lexicon_forms | structure.frequent_tokens
    publications_path = RAW_DATA_DIR / "publications.csv"

    total_rows = 31286
    processed_true_rows = 0
    last_progress = time.time()

    reader = csv.DictReader(cleaned_lines(publications_path))
    for row in reader:
        if str(row.get("has_akkadian", "")).strip().lower() != "true":
            continue

        processed_true_rows += 1
        page_text = str(row.get("page_text", ""))
        page_text_clean = clean_text(page_text)
        lines = [clean_text(line) for line in page_text.split("\n") if clean_text(line)]

        alias_match_found = any(alias in page_text_clean for alias in alias_map.keys())
        if alias_match_found:
            metrics["pages_with_alias"] += 1

        for idx, raw_line in enumerate(lines):
            normalized_line = normalize_candidate_spacing(raw_line, preferred_forms)
            is_akk, detection = detect_akkadian(normalized_line, structure, lexicon_forms)
            if not is_akk:
                continue

            metrics["akkadian_candidates"] += 1
            matched_reference, ref_similarity, best_ref = choose_best_reference(normalized_line, reference_index)
            if not matched_reference:
                continue

            metrics["reference_matches"] += 1
            translation_candidates = collect_translation_candidates(lines, idx, window=4)
            if not translation_candidates:
                continue

            best_translation = None
            best_distance = None
            translated_from = "unknown"
            for t_idx, candidate in translation_candidates:
                candidate_en, language = maybe_translate_to_english(candidate)
                if not candidate_en:
                    continue
                distance = abs(t_idx - idx)
                if distance > 3:
                    continue
                if best_translation is None or distance < best_distance:
                    best_translation = candidate_en
                    best_distance = distance
                    translated_from = language

            if not best_translation:
                continue

            score = 0
            if alias_match_found:
                score += 5
            if matched_reference:
                score += 3
            if best_distance is not None and best_distance <= 3:
                score += 2
            if english_like(best_translation):
                score += 2
            reasonable_length = 3 <= len(lexical_tokens(normalized_line)) <= 40 and 3 <= len(best_translation.split()) <= 50
            if reasonable_length:
                score += 1

            if score < 7:
                continue

            pair = (normalized_line.strip().lower(), best_translation.strip().lower())
            if not pair[0] or not pair[1]:
                continue
            if len(pair[0]) < 5 or len(pair[1]) < 5 or len(pair[0]) > 500 or len(pair[1]) > 500:
                continue
            if pair in seen_pairs:
                continue

            seen_pairs.add(pair)
            extracted_rows.append(
                {
                    "akkadian": pair[0],
                    "english": pair[1],
                    "score": score,
                    "distance": best_distance,
                    "reference_similarity": round(ref_similarity, 4),
                    "reference": best_ref,
                    "translation_language": translated_from,
                    "pdf_name": row.get("\ufeffpdf_name", ""),
                    "page": row.get("page", ""),
                }
            )
            metrics["kept_pairs"] += 1

        now = time.time()
        if processed_true_rows == 1 or processed_true_rows == total_rows or now - last_progress >= 2:
            progress = render_progress(processed_true_rows, total_rows, start_time)
            log(progress)
            last_progress = now

    extracted_rows.sort(key=lambda item: (-item["score"], item["distance"], -item["reference_similarity"]))

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
                "translation_language",
                "pdf_name",
                "page",
            ],
        )
        writer.writeheader()
        writer.writerows(extracted_rows)

    elapsed = time.time() - start_time
    with OUTPUT_MD.open("w", encoding="utf-8") as report:
        report.write("# Try 3 OCR Extraction Log\n\n")
        report.write("## Scope\n")
        report.write("- Attempt: Try 3\n")
        report.write("- Goal: extract new Akkadian-English pairs from `publications.csv` using linguistic validation and reference anchoring\n")
        report.write("- Priority: precision over quantity\n\n")

        report.write("## Phase 1 Learned Structure\n")
        report.write(f"- Common sentence starters tracked: {len(structure.common_sentence_starters)}\n")
        report.write(f"- Top 15 starters: {', '.join(sorted(list(structure.common_sentence_starters))[:15])}\n")
        report.write(
            f"- Length range: min={structure.length_range['min_length']}, "
            f"max={structure.length_range['max_length']}, avg={structure.length_range['avg_length']:.2f}\n"
        )
        report.write(
            f"- Detection window used: {structure.detection_length_min}..{structure.detection_length_max}\n\n"
        )

        report.write("## Phase 2-9 Extraction Metrics\n")
        report.write(f"- Pages scanned with `has_akkadian == true`: {processed_true_rows}\n")
        report.write(f"- Pages with alias hit: {metrics['pages_with_alias']}\n")
        report.write(f"- Akkadian candidates detected: {metrics['akkadian_candidates']}\n")
        report.write(f"- Reference matches retained: {metrics['reference_matches']}\n")
        report.write(f"- Final extracted pairs: {len(extracted_rows)}\n")
        report.write(f"- Runtime: {elapsed:.2f} seconds\n")
        report.write("- Multilingual note: non-English candidates were detected heuristically, but automatic translation was not available in this runtime, so only English-valid lines were retained.\n\n")

        report.write("## Sample Pairs\n")
        for row in extracted_rows[:5]:
            report.write(f"- SRC: {row['akkadian']}\n")
            report.write(f"- TGT: {row['english']}\n")
            report.write(f"  score={row['score']}, distance={row['distance']}, similarity={row['reference_similarity']}\n")

    return {
        "total_pairs": len(extracted_rows),
        "elapsed_seconds": elapsed,
        "samples": extracted_rows[:5],
    }


if __name__ == "__main__":
    summary = extract_try3()
    log(f"TRY 3 COMPLETE: extracted {summary['total_pairs']} pairs in {summary['elapsed_seconds']:.2f}s")
