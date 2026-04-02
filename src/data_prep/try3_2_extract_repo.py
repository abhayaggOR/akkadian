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

OUTPUT_CSV = TRAIN_DIR / "try3_2_extracted.csv"
OUTPUT_MD = TRAIN_DIR / "try3_2_read.md"
OUTPUT_LOG = TRAIN_DIR / "try3_2_process.log"

COMMON_ENGLISH = {
    "the", "and", "to", "of", "in", "for", "that", "with", "is", "was", "on", "as",
    "he", "she", "it", "they", "his", "her", "from", "this", "by", "be", "or", "an",
    "not", "are", "were", "had", "have", "at", "which", "their", "there",
}
ENGLISH_NEARBY_BONUS = {"translation", "translated", "means", "reads", "rendered", "text", "letter"}
SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
SUPERSCRIPT_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
AKKADIAN_SURFACE_RE = re.compile(r"[šṣṭḫāīū]|[a-z]+-[a-z]+", re.IGNORECASE)


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
    text = text.translate(SUBSCRIPT_MAP).translate(SUPERSCRIPT_MAP)
    text = re.sub(r"\s+", " ", text.lower()).strip()
    return text


def strip_editorial_markup(text: str) -> str:
    text = clean_text(text)
    text = re.sub(r"<[^>]*>", " <gap> ", text)
    text = text.replace("{", "").replace("}", "")
    text = text.replace("[", " ").replace("]", " ")
    text = text.replace("(", " ").replace(")", " ")
    text = re.sub(r"\b(fig|table|pl|plate|note|notes|fn|n)\.?\s*\d+\b", " ", text)
    text = re.sub(r"\b\d{1,4}\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_translit(text: str) -> str:
    text = strip_editorial_markup(text)
    text = text.replace("—", "-").replace("–", "-")
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"[,;:!?]", " ", text)
    text = re.sub(r"\.+", " ", text)
    text = re.sub(r"[/|]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_transliteration(text: str) -> list[str]:
    return [normalize_translit(part) for part in re.split(r"[\n\.]", str(text)) if normalize_translit(part)]


def lexical_tokens(text: str) -> list[str]:
    tokens = []
    for token in normalize_translit(text).split():
        token = re.sub(r"^[^\w<]+|[^\w>]+$", "", token)
        if token:
            tokens.append(token)
    return tokens


def translit_token_variants(token: str) -> set[str]:
    token = token.strip()
    if not token:
        return set()
    variants = {token}
    no_hyphen = token.replace("-", "")
    if no_hyphen:
        variants.add(no_hyphen)
    if "-" in token:
        variants.update(part for part in token.split("-") if part)
    return variants


def canonical_string(text: str) -> str:
    return re.sub(r"[^a-z0-9šṣṭḫāīū<>\- ]+", " ", normalize_translit(text))


def english_like(text: str) -> bool:
    words = re.findall(r"[a-z]+", text.lower())
    if len(words) < 3:
        return False
    stopword_hits = sum(word in COMMON_ENGLISH for word in words)
    alpha_chars = [char for char in text if not char.isspace()]
    alpha_ratio = sum(char.isalpha() for char in alpha_chars) / len(alpha_chars) if alpha_chars else 0.0
    return stopword_hits >= 1 and alpha_ratio >= 0.60


def english_score(text: str) -> float:
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return 0.0
    stopword_hits = sum(word in COMMON_ENGLISH for word in words)
    bonus_hits = sum(word in ENGLISH_NEARBY_BONUS for word in words)
    return (stopword_hits / len(words)) + 0.05 * bonus_hits


def looks_sentence_like(text: str) -> bool:
    words = text.split()
    if not (3 <= len(words) <= 60):
        return False
    if english_like(text):
        return True
    if '"' in text or "'" in text or "“" in text or "”" in text:
        return True
    alpha_chars = [char for char in text if not char.isspace()]
    alpha_ratio = sum(char.isalpha() for char in alpha_chars) / len(alpha_chars) if alpha_chars else 0.0
    return alpha_ratio >= 0.68


def render_progress(current: int, total: int, start_time: float) -> str:
    width = 24
    ratio = current / total if total else 1.0
    filled = min(width, int(width * ratio))
    bar = "#" * filled + "-" * (width - filled)
    elapsed = max(time.time() - start_time, 1e-9)
    rate = current / elapsed
    remaining = (total - current) / rate if rate > 0 else 0.0
    return f"[{bar}] {current}/{total} ({ratio * 100:5.1f}%) elapsed={elapsed:6.1f}s eta={remaining:6.1f}s"


@dataclass
class LearnedStructure:
    common_sentence_starters: set[str]
    frequent_tokens: set[str]
    min_length: int
    max_length: int
    avg_length: float
    detect_min: int
    detect_max: int


@dataclass
class ReferenceWindow:
    ref_id: int
    text: str
    canonical_text: str
    tokens: tuple[str, ...]
    token_set: frozenset[str]
    line_count: int


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
        common_sentence_starters={token for token, _ in first_words.most_common(100)},
        frequent_tokens={token for token, _ in token_freq.most_common(1000)},
        min_length=min(lengths),
        max_length=max(lengths),
        avg_length=statistics.mean(lengths),
        detect_min=ordered[int(0.02 * len(ordered))],
        detect_max=ordered[int(0.98 * len(ordered))],
    )


def build_lexicon_forms() -> set[str]:
    forms = set()
    for file_name, column in [("OA_Lexicon_eBL.csv", "form"), ("eBL_Dictionary.csv", "word")]:
        with (RAW_DATA_DIR / file_name).open("r", encoding="utf-8", errors="replace", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                value = normalize_translit(row.get(column, ""))
                if value:
                    forms.add(value)
                    for token in lexical_tokens(value):
                        forms.add(token)
                        forms.update(translit_token_variants(token))
    return forms


def normalize_candidate_spacing(line: str, preferred_forms: set[str]) -> str:
    tokens = normalize_translit(line).split()
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
    return normalize_translit(" ".join(normalized))


def detect_akkadian(line: str, structure: LearnedStructure, lexicon_forms: set[str]) -> tuple[bool, dict[str, float]]:
    tokens = lexical_tokens(line)
    if not tokens:
        return False, {"score": 0.0}

    score = 0
    hyphen_ratio = sum("-" in token for token in tokens) / len(tokens)
    if hyphen_ratio >= 0.20:
        score += 2

    lexicon_hits = 0
    for token in tokens:
        variants = translit_token_variants(token)
        if any(variant in lexicon_forms for variant in variants):
            lexicon_hits += 1
    coverage = lexicon_hits / len(tokens)
    if coverage >= 0.35:
        score += 2
    if coverage >= 0.55:
        score += 1

    if structure.detect_min <= len(tokens) <= min(structure.detect_max, 45):
        score += 1

    if tokens[0] in structure.common_sentence_starters:
        score += 1

    frequent_hits = sum(token in structure.frequent_tokens for token in tokens[:4])
    if frequent_hits >= 1:
        score += 1

    english_penalty = english_score(line)
    if english_penalty > 0.20:
        score -= 2

    return score >= 4, {
        "score": score,
        "coverage": coverage,
        "length": len(tokens),
        "hyphen_ratio": hyphen_ratio,
    }


def build_reference_assets() -> tuple[dict[str, list[int]], dict[str, list[int]], list[ReferenceWindow], dict[str, float], dict[int, str]]:
    alias_map: dict[str, list[int]] = defaultdict(list)
    token_index: dict[str, list[int]] = defaultdict(list)
    windows: list[ReferenceWindow] = []
    doc_freq = Counter()
    ref_doc_names: dict[int, str] = {}

    next_id = 0
    with (RAW_DATA_DIR / "published_texts.csv").open("r", encoding="utf-8", errors="replace", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            transliteration = row.get("transliteration", "")
            aliases = row.get("aliases", "") or ""
            label = clean_text(row.get("label", "") or row.get("publication_catalog", "") or row.get("oare_id", ""))

            split_lines = [line for line in split_transliteration(transliteration) if 2 <= len(lexical_tokens(line)) <= 35]
            if not split_lines:
                continue

            local_ids = []
            for window_size in (1, 2, 3):
                for idx in range(0, len(split_lines) - window_size + 1):
                    window_text = " ".join(split_lines[idx:idx + window_size])
                    tokens = tuple(lexical_tokens(window_text))
                    token_set = frozenset(tokens)
                    if len(token_set) < 2:
                        continue
                    ref = ReferenceWindow(
                        ref_id=next_id,
                        text=window_text,
                        canonical_text=canonical_string(window_text).replace("-", ""),
                        tokens=tokens,
                        token_set=token_set,
                        line_count=window_size,
                    )
                    windows.append(ref)
                    ref_doc_names[next_id] = label
                    local_ids.append(next_id)
                    doc_freq.update(token_set)
                    next_id += 1

            for alias in aliases.split("|"):
                alias_clean = clean_text(alias)
                if alias_clean and local_ids:
                    alias_map[alias_clean].extend(local_ids)

    total_refs = len(windows)
    token_idf = {}
    for token, freq in doc_freq.items():
        token_idf[token] = math.log((1 + total_refs) / (1 + freq)) + 1.0

    for ref in windows:
        for token in ref.token_set:
            token_index[token].append(ref.ref_id)

    return alias_map, token_index, windows, token_idf, ref_doc_names


def reference_candidate_pool(
    candidate_tokens: list[str],
    page_text_clean: str,
    alias_map: dict[str, list[int]],
    token_index: dict[str, list[int]],
    token_idf: dict[str, float],
) -> tuple[bool, list[int]]:
    alias_match = False
    scored_ids = Counter()

    for alias, ref_ids in alias_map.items():
        if alias and alias in page_text_clean:
            alias_match = True
            for ref_id in ref_ids:
                scored_ids[ref_id] += 5

    query_tokens = []
    for token in candidate_tokens:
        variants = translit_token_variants(token)
        for variant in variants:
            if variant in token_index:
                query_tokens.append(variant)

    query_tokens = sorted(set(query_tokens), key=lambda token: token_idf.get(token, 0.0), reverse=True)[:6]
    for token in query_tokens:
        for ref_id in token_index.get(token, [])[:180]:
            scored_ids[ref_id] += token_idf.get(token, 1.0)

    top_ids = [ref_id for ref_id, _ in scored_ids.most_common(40)]
    return alias_match, top_ids


def rerank_reference(
    candidate: str,
    candidate_tokens: list[str],
    pool_ids: list[int],
    windows: list[ReferenceWindow],
    token_idf: dict[str, float],
    alias_match: bool,
) -> tuple[bool, float, ReferenceWindow | None, int, float]:
    candidate_set = set(candidate_tokens)
    candidate_canonical = canonical_string(candidate).replace("-", "")

    coarse_ranked = []
    for ref_id in pool_ids:
        ref = windows[ref_id]
        shared_tokens = candidate_set & ref.token_set
        if not shared_tokens:
            continue
        weighted_overlap = sum(token_idf.get(token, 1.0) for token in shared_tokens)
        union_weight = sum(token_idf.get(token, 1.0) for token in candidate_set | ref.token_set)
        token_overlap = weighted_overlap / union_weight if union_weight else 0.0
        prefix_bonus = 0.05 if candidate_tokens[:2] == list(ref.tokens[:2]) else 0.0
        coarse_score = token_overlap + prefix_bonus + (0.04 if alias_match else 0.0)
        coarse_ranked.append((coarse_score, ref, len(shared_tokens), weighted_overlap))

    coarse_ranked.sort(key=lambda item: item[0], reverse=True)
    shortlist = coarse_ranked[:6]

    best_score = 0.0
    best_ref = None
    best_shared = 0
    best_weight = 0.0

    for token_overlap, ref, shared_len, weighted_overlap in shortlist:
        char_ratio = SequenceMatcher(None, candidate_canonical, ref.canonical_text).ratio()
        score = 0.65 * token_overlap + 0.35 * char_ratio
        if alias_match:
            score += 0.05

        if score > best_score:
            best_score = score
            best_ref = ref
            best_shared = shared_len
            best_weight = weighted_overlap

    matched = False
    if best_ref is not None:
        if best_score >= 0.40 and best_shared >= 3:
            matched = True
        elif best_score >= 0.47 and best_shared >= 2:
            matched = True
        elif alias_match and best_score >= 0.34 and best_shared >= 2:
            matched = True

    return matched, best_score, best_ref, best_shared, best_weight


def collect_translation_candidates(lines: list[str], pivot: int) -> list[tuple[int, str]]:
    start = max(0, pivot - 6)
    end = min(len(lines), pivot + 7)
    candidates = []
    for idx in range(start, end):
        if idx == pivot:
            continue
        line = clean_text(lines[idx])
        if looks_sentence_like(line):
            candidates.append((idx, line))
    return candidates


def choose_translation(lines: list[str], pivot: int) -> tuple[str | None, int | None, float]:
    best_line = None
    best_distance = None
    best_score = 0.0

    for idx, line in collect_translation_candidates(lines, pivot):
        if not english_like(line):
            continue
        distance = abs(idx - pivot)
        score = english_score(line) + 0.04 * max(0, 6 - distance)
        if best_line is None or score > best_score:
            best_line = line
            best_distance = distance
            best_score = score

    return best_line, best_distance, best_score


def extract_try3_2() -> dict[str, object]:
    start_time = time.time()
    OUTPUT_LOG.write_text("", encoding="utf-8")

    log("TRY 3.2: Starting OCR-to-reference alignment pipeline")
    log("PHASE 1: Learning Akkadian structure from train.csv")
    structure = learn_akkadian_structure(RAW_DATA_DIR / "train.csv")
    log(
        f"  learned range={structure.min_length}..{structure.max_length} "
        f"(avg={structure.avg_length:.2f}, detect_window={structure.detect_min}..{structure.detect_max})"
    )

    log("PHASE 2: Building lexicon-backed OCR normalization system")
    lexicon_forms = build_lexicon_forms()
    lexicon_forms.update(structure.frequent_tokens)
    log(f"  lexicon-backed forms/tokens={len(lexicon_forms)}")

    log("PHASE 3: Building reference windows and retrieval index from published_texts.csv")
    alias_map, token_index, windows, token_idf, ref_doc_names = build_reference_assets()
    log(f"  alias keys={len(alias_map)}")
    log(f"  reference windows={len(windows)}")
    log(f"  indexed tokens={len(token_index)}")

    preferred_forms = lexicon_forms | structure.frequent_tokens
    metrics = Counter()
    extracted_rows = []
    seen_pairs = set()
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
        raw_lines = [line for line in page_text.split("\n") if clean_text(line)]

        for idx, raw_line in enumerate(raw_lines):
            if not AKKADIAN_SURFACE_RE.search(raw_line.lower()):
                continue
            normalized = normalize_candidate_spacing(raw_line, preferred_forms)
            is_akkadian, detect_meta = detect_akkadian(normalized, structure, lexicon_forms)
            if not is_akkadian:
                continue

            metrics["akkadian_candidates"] += 1
            candidate_tokens = lexical_tokens(normalized)
            alias_match, pool_ids = reference_candidate_pool(candidate_tokens, page_text_clean, alias_map, token_index, token_idf)
            if not pool_ids:
                continue

            metrics["retrieval_hits"] += 1
            matched, ref_score, best_ref, shared_count, shared_weight = rerank_reference(
                normalized,
                candidate_tokens,
                pool_ids,
                windows,
                token_idf,
                alias_match,
            )
            if not matched or best_ref is None:
                continue

            metrics["reference_matches"] += 1
            best_translation, best_distance, translation_score = choose_translation(raw_lines, idx)
            if not best_translation:
                continue

            metrics["translation_matches"] += 1
            score = 0
            if alias_match:
                score += 5
            if matched:
                score += 3
            if best_distance is not None and best_distance <= 4:
                score += 2
            elif best_distance is not None and best_distance <= 6:
                score += 1
            if english_like(best_translation):
                score += 2
            if shared_count >= 3:
                score += 1
            if detect_meta["coverage"] >= 0.5:
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
                    "reference_score": round(ref_score, 4),
                    "shared_tokens": shared_count,
                    "shared_weight": round(shared_weight, 4),
                    "reference": best_ref.text,
                    "reference_doc": ref_doc_names.get(best_ref.ref_id, ""),
                    "alias_match": alias_match,
                    "translation_score": round(translation_score, 4),
                    "pdf_name": row.get("\ufeffpdf_name", ""),
                    "page": row.get("page", ""),
                }
            )
            metrics["kept_pairs"] += 1

        now = time.time()
        if processed_true_rows == 1 or processed_true_rows == total_rows or now - last_progress >= 2:
            log(render_progress(processed_true_rows, total_rows, start_time))
            last_progress = now

    extracted_rows.sort(
        key=lambda row: (-row["score"], -row["reference_score"], -row["shared_tokens"], row["distance"] if row["distance"] is not None else 99)
    )

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "akkadian",
                "english",
                "score",
                "distance",
                "reference_score",
                "shared_tokens",
                "shared_weight",
                "reference",
                "reference_doc",
                "alias_match",
                "translation_score",
                "pdf_name",
                "page",
            ],
        )
        writer.writeheader()
        writer.writerows(extracted_rows)

    elapsed = time.time() - start_time
    with OUTPUT_MD.open("w", encoding="utf-8") as file:
        file.write("# Try 3.2 OCR Extraction Log\n\n")
        file.write("## Scope\n")
        file.write("- Attempt: Try 3.2\n")
        file.write("- Goal: improve OCR-to-reference alignment using retrieval plus reranking instead of only direct fuzzy matching\n")
        file.write("- Priority: still precision-first, but with broader normalized retrieval before rejection\n\n")

        file.write("## Key Changes From Try 3.1\n")
        file.write("- Added OCR cleanup and transliteration normalization before matching\n")
        file.write("- Built 1-line, 2-line, and 3-line reference windows from `published_texts.csv`\n")
        file.write("- Indexed reference windows by informative transliteration tokens\n")
        file.write("- Retrieved top candidate windows using rare-token overlap before reranking\n")
        file.write("- Reranked with weighted token overlap, canonical-string similarity, and prefix bonus\n")
        file.write("- Deferred nearby English alignment until after a reference window was accepted\n\n")

        file.write("## Metrics\n")
        file.write(f"- Pages scanned with `has_akkadian == true`: {processed_true_rows}\n")
        file.write(f"- Akkadian candidates detected: {metrics['akkadian_candidates']}\n")
        file.write(f"- Retrieval hits: {metrics['retrieval_hits']}\n")
        file.write(f"- Reference matches retained: {metrics['reference_matches']}\n")
        file.write(f"- Nearby English matches retained: {metrics['translation_matches']}\n")
        file.write(f"- Final extracted pairs: {len(extracted_rows)}\n")
        file.write(f"- Runtime: {elapsed:.2f} seconds\n")
        file.write("- Multilingual note: this runtime still keeps only English-valid nearby lines because automatic translation is not available locally.\n\n")

        file.write("## Sample Pairs\n")
        for row in extracted_rows[:5]:
            file.write(f"- SRC: {row['akkadian']}\n")
            file.write(f"- TGT: {row['english']}\n")
            file.write(
                f"  score={row['score']}, ref_score={row['reference_score']}, shared_tokens={row['shared_tokens']}, distance={row['distance']}, alias_match={row['alias_match']}\n"
            )

    return {"total_pairs": len(extracted_rows), "elapsed_seconds": elapsed, "samples": extracted_rows[:5]}


if __name__ == "__main__":
    summary = extract_try3_2()
    log(f"TRY 3.2 COMPLETE: extracted {summary['total_pairs']} pairs in {summary['elapsed_seconds']:.2f}s")
