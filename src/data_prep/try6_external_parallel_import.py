from __future__ import annotations

import csv
import re
import time
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = REPO_ROOT / "train_folder"
HELP_DIR = REPO_ROOT.parent / "help" / "Akkademia" / "NMT_input"

ACTIVE_SRC = TRAIN_DIR / "try4_2_train.src"
ACTIVE_TGT = TRAIN_DIR / "try4_2_train.tgt"

OUTPUT_IMPORTED = TRAIN_DIR / "try6_external_import.csv"
OUTPUT_ADDED = TRAIN_DIR / "try6_added_only.csv"
OUTPUT_SRC = TRAIN_DIR / "try6_train.src"
OUTPUT_TGT = TRAIN_DIR / "try6_train.tgt"
OUTPUT_READ = TRAIN_DIR / "try6_read.md"
OUTPUT_LOG = TRAIN_DIR / "try6_process.log"


def log(message: str) -> None:
    print(message, flush=True)
    with OUTPUT_LOG.open("a", encoding="utf-8") as file:
        file.write(message + "\n")


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def looks_non_linguistic(text: str) -> bool:
    return not bool(re.search(r"[a-zšṣṭḫāīū]", text, re.IGNORECASE))


def acceptable_ratio(src: str, tgt: str) -> bool:
    src_len = max(1, len(src.split()))
    tgt_len = max(1, len(tgt.split()))
    ratio = tgt_len / src_len
    return 0.3 <= ratio <= 6.0


def load_active_pairs() -> list[tuple[str, str]]:
    src_lines = ACTIVE_SRC.read_text(encoding="utf-8").splitlines()
    tgt_lines = ACTIVE_TGT.read_text(encoding="utf-8").splitlines()
    return [(clean_text(src), clean_text(tgt)) for src, tgt in zip(src_lines, tgt_lines) if clean_text(src) and clean_text(tgt)]


def iter_external_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for mode in ["train", "valid", "test"]:
        tr_lines = (HELP_DIR / f"{mode}.tr").read_text(encoding="utf-8").splitlines()
        en_lines = (HELP_DIR / f"{mode}.en").read_text(encoding="utf-8").splitlines()
        log(f"TRY 6 [{mode}] loading {len(tr_lines)} transliteration lines and {len(en_lines)} translation lines")
        for index, (src, tgt) in enumerate(zip(tr_lines, en_lines), start=1):
            rows.append(
                {
                    "mode": mode,
                    "row_id": f"{mode}:{index}",
                    "akkadian": clean_text(src),
                    "english": clean_text(tgt),
                }
            )
    return rows


def filter_external_rows(external_rows: list[dict[str, str]], active_set: set[tuple[str, str]]) -> tuple[list[dict[str, object]], dict[str, int]]:
    deduped: list[dict[str, str]] = []
    seen = set()
    duplicate_inside_external = 0

    for row in external_rows:
        pair = (row["akkadian"], row["english"])
        if not row["akkadian"] or not row["english"]:
            continue
        if pair in seen:
            duplicate_inside_external += 1
            continue
        seen.add(pair)
        deduped.append(row)

    imported_rows: list[dict[str, object]] = []
    already_in_active = 0
    filtered_out = 0
    mode_counts: Counter[str] = Counter()

    for row in deduped:
        pair = (row["akkadian"], row["english"])
        if pair in active_set:
            already_in_active += 1
            continue
        if looks_non_linguistic(row["akkadian"]) or looks_non_linguistic(row["english"]):
            filtered_out += 1
            continue
        if len(row["akkadian"].split()) < 3 or len(row["english"].split()) < 3:
            filtered_out += 1
            continue
        if not acceptable_ratio(row["akkadian"], row["english"]):
            filtered_out += 1
            continue

        src_len = len(row["akkadian"].split())
        tgt_len = len(row["english"].split())
        imported_rows.append(
            {
                "mode": row["mode"],
                "row_id": row["row_id"],
                "akkadian": row["akkadian"],
                "english": row["english"],
                "src_len": src_len,
                "tgt_len": tgt_len,
                "ratio": round(tgt_len / max(1, src_len), 3),
            }
        )
        mode_counts[str(row["mode"])] += 1

    stats = {
        "external_raw": len(external_rows),
        "external_unique": len(deduped),
        "duplicate_inside_external": duplicate_inside_external,
        "already_in_active": already_in_active,
        "filtered_out": filtered_out,
        "imported_new": len(imported_rows),
        "train_kept": mode_counts["train"],
        "valid_kept": mode_counts["valid"],
        "test_kept": mode_counts["test"],
    }
    return imported_rows, stats


def write_outputs(active_pairs: list[tuple[str, str]], imported_rows: list[dict[str, object]], stats: dict[str, int], elapsed: float) -> None:
    with OUTPUT_IMPORTED.open("w", encoding="utf-8", newline="") as csv_file:
        fieldnames = ["mode", "row_id", "akkadian", "english", "src_len", "tgt_len", "ratio"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(imported_rows)

    OUTPUT_ADDED.write_text(OUTPUT_IMPORTED.read_text(encoding="utf-8"), encoding="utf-8")

    merged_pairs = active_pairs + [(str(row["akkadian"]), str(row["english"])) for row in imported_rows]
    OUTPUT_SRC.write_text("\n".join(src for src, _ in merged_pairs) + "\n", encoding="utf-8")
    OUTPUT_TGT.write_text("\n".join(tgt for _, tgt in merged_pairs) + "\n", encoding="utf-8")

    sample_rows = imported_rows[:5]
    with OUTPUT_READ.open("w", encoding="utf-8") as file:
        file.write("# Try 6 External Parallel Import\n\n")
        file.write("## Scope\n")
        file.write("- Attempt: Try 6\n")
        file.write("- Goal: expand the current Try 4.2 working dataset using an externally prepared line-aligned Akkadian-English corpus\n")
        file.write("- Active baseline before import: `try4_2_train.src` / `try4_2_train.tgt`\n\n")
        file.write("## Techniques Used\n")
        file.write("- Imported pre-aligned transliteration and translation files directly by row index\n")
        file.write("- Used the plain-text `train`, `valid`, and `test` splits rather than tokenized variants\n")
        file.write("- Deduplicated the external corpus internally before comparison\n")
        file.write("- Removed rows already present in the active Try 4.2 set\n")
        file.write("- Applied simple safety filters: non-empty rows, minimum token length, and source-target ratio bounds\n")
        file.write("- Stored the imported rows separately before creating the merged corpus\n\n")
        file.write("## Results\n")
        file.write(f"- External raw pairs loaded: {stats['external_raw']}\n")
        file.write(f"- External unique pairs after internal deduplication: {stats['external_unique']}\n")
        file.write(f"- Exact duplicates already present in Try 4.2: {stats['already_in_active']}\n")
        file.write(f"- Additional rows filtered out by safety rules: {stats['filtered_out']}\n")
        file.write(f"- New imported pairs kept: {stats['imported_new']}\n")
        file.write(f"- Kept from train split: {stats['train_kept']}\n")
        file.write(f"- Kept from valid split: {stats['valid_kept']}\n")
        file.write(f"- Kept from test split: {stats['test_kept']}\n")
        file.write(f"- Final merged total (Try 4.2 + Try 6): {len(merged_pairs)}\n")
        file.write(f"- Runtime: {elapsed:.2f} seconds\n\n")
        if sample_rows:
            file.write("## Sample Pairs\n")
            for row in sample_rows:
                file.write(f"- {row['row_id']} ({row['mode']})\n")
                file.write(f"  akkadian={row['akkadian']}\n")
                file.write(f"  english={row['english']}\n")


def main() -> None:
    start = time.time()
    OUTPUT_LOG.write_text("", encoding="utf-8")
    log("TRY 6: Importing external line-aligned corpus")
    active_pairs = load_active_pairs()
    active_set = set(active_pairs)
    external_rows = iter_external_rows()
    imported_rows, stats = filter_external_rows(external_rows, active_set)
    elapsed = time.time() - start
    write_outputs(active_pairs, imported_rows, stats, elapsed)
    log(
        "TRY 6 COMPLETE: "
        f"external_raw={stats['external_raw']}, "
        f"external_unique={stats['external_unique']}, "
        f"imported_new={stats['imported_new']}, "
        f"merged_total={len(active_pairs) + len(imported_rows)}, "
        f"runtime={elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
