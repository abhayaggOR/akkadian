# Akkadian to English NMT: Data Preprocessing Pipeline

This repository contains the dataset processing pipeline and generated training pairs for translating Old Assyrian transliterated texts into English, utilizing the Deep Past Challenge dataset.

## Try 1: High-Precision Baseline Extraction

This represents our **First Try** at extracting a workable dataset from the raw challenge data. The core philosophy driving this extraction was to rigidly prioritize **precision over quantity** ("better few clean pairs than many noisy ones"). By refusing to aggressively mine messy OCR text, we actively immunized the training set from misalignment noise. We executed a rigorous 7-Phase data pipeline to construct the datasets located in `train_folder/`.

### The 7 Phases & Our Analytical Decisions

#### Phase 1: Simple Sentence Splitting (`train.csv`)
We split text inputs into sentences based strictly on structural returns (`\n` and `.`). 
*Decision Logic:* Since `train.csv` provided document-level translations rather than sentence-level, splitting by robust punctuation ensures our sequential Zip alignment actually pairs contextual source statements with their English targets.

#### Phase 2: Cleaning & Length Constraints
We lowercased all pairs, stripped whitespace, removed sequences of pure punctuation/numbers, and dropped sentences with less than 5 characters.
*Decision Logic:* Tiny fragmented relics from broken clay tablets (e.g. `1...`) carry negligible semantic value for sequence-to-sequence modelling. Lowercasing helps restrict vocabulary sparsity so the tokenizers hit robust character patterns faster.

#### Phase 3: Writing Training Files & Preserving Hyphens
We validated the pairs and outputted the first cleanly aligned datasets, explicitly keeping hyphens safely preserved in the Akkadian source text.
*Decision Logic:* In Assyriology and Akkadian transliteration, hyphens act as foundational syllabic sign boundaries (e.g., `ma-nu-ki-a-šur`). Stripping them would haphazardly merge distinct cuneiform readings together, permanently destroying the complex morphology needed to translate accurately.

#### Phase 4: Foundational Tokenizer Corpus Generation (`corpus.src`)
From `published_texts.csv` we generated a continuous unaligned 55,105 line Akkadian string corpus.
*Decision Logic:* Modern Subword Tokenizers (like Byte-Pair Encodings or WordPiece) rely on massive statistical corpuses to correctly segment unknown words. Before deploying transformers, we structurally isolated this unaligned Akkadian text precisely to train the underlying Tokenizer model optimally over millions of subwords.

#### Phase 5: Ultra-Conservative OCR Extraction (`publications.csv`)
We parsed the expansive 580MB scholarly publications OCR dump, programming it to aggressively seek near-perfect ASCII English sentences residing strictly within 3 lines of Akkadian-styled strings (i.e. exhibiting internal hyphens, `š`, `ṣ`, `ṭ`, `ḫ`). 
*Decision Logic:* Because scanning old scholarly PDFs via OCR inherently introduces severe noise (abrupt footnotes, multi-language paragraphs, weird spacings), attempting loose fuzzy mapping injects devastating alignment issues (hallucinations) into neural model training. By enforcing rigid constraints, the matching hit-rate collapsed, and we intentionally extracted **exactly 0 pairs**. This safeguarded the integrity of our dataset. "Try 1" focuses purely on pristine pairings.

#### Phase 6: Dataset Aggregation
We deduplicated pairs across both datasets (originally intending to merge Phase 2 with Phase 5), and shuffled the pairs systematically to yield a randomized distribution.
*Decision Logic:* Since Phase 5 prioritized safety over quantity yielding 0 external pairs, our dataset composition holds exclusively at 100% clean mappings extracted originally from `train.csv`. Removing duplicates actively blocks translation memorization during epochs.

#### Phase 7: Final Baseline Architecture (`final_train.src / tgt`)
We finalized validation byte allocations, resulting in exactly **6,052 pristine sentence-aligned pairs**.
*Decision Logic:* A clean dataset is superior to a large noisy dataset in NMT architecture. `final_train.src` and `final_train.tgt` are structurally standardized and will map directly into the HuggingFace `datasets` framework easily.

---

### Folder Architecture
- `train_folder/`: Contains all mapped parallel `.src` and `.tgt` text distributions, the python execution script (`process_data.py`), and the deeply granular processing metadata log (`read.md`).
- `superficial_read.md`: A high-level processing log detailing the initial steps undertaken.

## Try 2: Multi-lingual OCR Mining (10-Step Anchor Logic)

In this **Second Try**, we attempted an advanced programmatic approach to mine the massive, unstructured OCR data dumped in `publications.csv` (which originally yielded 0 pairs in Try 1 due to our ultra-conservative string-matching).

We implemented a sophisticated extraction pipeline (`try2_extract.py`) using a heuristic point-based Anchor Logic model.

### Execution Logic & Anchor Proximity Scoring
Because the OCR data is highly unstructured (containing scattered scholarly footnotes, language headers, and formatting errors), we calculated dynamic confidence scores line-by-line between potential Akkadian sources and English translations. 

The pipeline assigned points based on the following rules:
1. **Keyword Anchor Match (+5 points):** The page explicitly contained document structure keywords like "translation", "transliteration", "akkadian", or "english".
2. **Close Proximity (+5 points):** The English translation candidate was structurally within 3 positional lines of the Akkadian line.
3. **Language Identifiers (+2 points):** The candidate text was detected explicitly as native English using `langdetect`.
4. **Translated Fallback (+1 point):** We integrated the `deep-translator` multi-lingual fallback. If the candidate was a foreign language (e.g. French, German, Spanish), the pipeline translated it immediately to English on the fly. 

To prevent neural architecture hallucinations, we set an extremely conservative validation boundary: **Only pairs scoring 7 or more points were accepted.** 

### Results & Final Extraction
After completely processing the `publications.csv` document matrix, the algorithm **extracted exactly 0 validating pairs**. 

**Core Diagnostics:**
Our strict threshold validation math structurally eliminated all translated foreign translations that appeared without explicit page headers. 
For instance, if a German translation was perfectly aligned alongside an Akkadian text (+5 distance points), but the actual PDF page lacked an explicit header reading "Translation" (0 anchor points), the German candidate would translate successfully (+1 translation point), reaching a final score of **6 points**. Because 6 < 7, it was algorithmically dropped right at the validation threshold.

Additionally, older scholarly book publications frequently decouple Accadian cuneiform and modern translations entirely across opposite split pages, fracturing our core 3-line structural proximity assumption.

To iterate in future data mining, we will likely lower the translation threshold score to 6, or radically expand the positional line scanning distance.

## Try 3: Linguistically Anchored Reference Matching

In this **Third Try**, we move away from simple OCR proximity heuristics alone and instead combine:

1. learned Akkadian structure from `train.csv`
2. lexicon validation from `OA_Lexicon_eBL.csv`
3. reference anchoring from `published_texts.csv`
4. conservative translation-line validation around each Akkadian candidate

The new repo-native script is located at `src/data_prep/try3_extract_repo.py`.

### What Try 3 Adds

- Learns common sentence starters, token frequencies, and sentence lengths directly from `train.csv`
- Detects Akkadian candidates using hyphen patterns, lexicon coverage, starter frequency, and learned length constraints
- Anchors candidates against `published_texts.csv` using alias lookup plus fuzzy reference similarity
- Searches only nearby OCR lines for translation candidates
- Applies confidence scoring to keep only high-precision pairs
- Writes its outputs into `train_folder/try3_extracted.csv`, `train_folder/try3_read.md`, and `train_folder/try3_process.log`

### Precision Note

Try 3 still prioritizes **precision over quantity**. In the current local runtime, multilingual OCR lines can be detected heuristically, but automatic machine translation may not be available unless optional dependencies are installed. Because of that, the current implementation keeps only translation candidates that validate as English in the runtime environment.

## Try 3.1: Sentence-Level Anchor Relaxation

In **Try 3.1**, we preserve the overall Try 3 structure but specifically relax the anchor stage that previously collapsed the extraction count to zero.

The refined script is located at `src/data_prep/try3_1_extract_repo.py`.

### What Try 3.1 Changes

- Splits `published_texts.csv` transliterations into sentence-level reference lines before matching
- Uses a wider first-token and alias-backed candidate pool instead of only full-text anchor matching
- Lowers the lexicon coverage requirement slightly to admit more OCR-damaged Akkadian candidates
- Allows weaker similarity when the page contains a strong alias anchor
- Expands the nearby translation search window while still selecting only the nearest English-valid line

### Outputs

Try 3.1 writes:

- `train_folder/try3_1_extracted.csv`
- `train_folder/try3_1_read.md`
- `train_folder/try3_1_process.log`

### Current Result

In the current local run, Try 3.1 scanned all `31,286` OCR pages where `has_akkadian == true`, detected additional Akkadian candidates, and completed in roughly `217` seconds, but still extracted **0 final pairs** after the confidence and English-validation stages.

This suggests the next bottleneck is no longer only candidate detection. The remaining failure point is the combination of:

- noisy OCR degradation in Akkadian lines
- weak nearby English alignment on many pages
- lack of multilingual translation support in the current runtime

## Try 3.2: OCR-to-Reference Retrieval and Reranking

In **Try 3.2**, we reworked the alignment stage from a plain fuzzy check into a retrieval problem followed by reranking.

The new script is located at `src/data_prep/try3_2_extract_repo.py`.

### What Try 3.2 Changes

- Applies more aggressive OCR cleanup before matching
- Normalizes transliteration spacing, hyphens, editorial marks, and digit noise
- Builds 1-line, 2-line, and 3-line reference windows from `published_texts.csv`
- Indexes those windows by informative transliteration tokens
- Retrieves candidate reference windows using rare-token overlap first
- Reranks only a small shortlist using weighted token overlap and canonical string similarity
- Delays translation alignment until an Akkadian reference anchor has already been accepted

### Outputs

Try 3.2 writes:

- `train_folder/try3_2_extracted.csv`
- `train_folder/try3_2_read.md`
- `train_folder/try3_2_process.log`

### Current Result

Try 3.2 completed a full local run over all `31,286` pages where `has_akkadian == true` in roughly `781.55` seconds.

Its main improvement is that the anchor stage finally stopped collapsing to zero:

- Akkadian candidates detected: `6,428`
- Reference matches retained: `18`
- Nearby English matches retained: `0`
- Final extracted pairs: `0`

This means the bottleneck has moved again. Try 3.2 can now anchor some OCR Akkadian lines to clean reference windows, but the surrounding OCR pages still do not yield nearby English lines that validate strongly enough under the current runtime constraints.

## Try 4: Train.csv Sentence Expansion

In **Try 4**, we shifted focus away from OCR mining and instead tried to expand the usable parallel corpus directly from `train.csv`.

The script for this attempt is located at `src/data_prep/try4_expand_train_pairs.py`.

### What Try 4 Changes

- Keeps the original `6,052` clean sentence pairs as the baseline backbone
- Normalizes transliteration before the second pass to reduce fragmentation
- Splits English documents into sentence-like units using punctuation anchors
- Allocates Akkadian source spans proportionally according to English sentence length
- Re-filters the new candidate pairs using noise checks, copy checks, and length-ratio checks
- Adds only pairs that are new relative to the original baseline

### How Try 4 Works

The pipeline first recreates the original clean baseline from `train.csv` using the existing newline/period split. It then runs a second sentence-expansion pass:

1. normalize Akkadian transliteration to reduce sparsity and variation
2. split the English side into sentence-like chunks using punctuation anchors
3. allocate Akkadian token spans proportionally based on the relative English sentence lengths
4. attach leftover Akkadian words to the final chunk so tokens are not dropped
5. re-filter every newly proposed pair using stricter sanity checks
6. keep only genuinely new pairs not already present in the `6,052` baseline

### Outputs

Try 4 writes:

- `train_folder/try4_train.src`
- `train_folder/try4_train.tgt`
- `train_folder/try4_added_only.csv`
- `train_folder/try4_read.md`
- `train_folder/try4_process.log`

### Current Result

Try 4 expanded the corpus beyond the original baseline:

- Original clean baseline pairs: `6,052`
- Additional Try 4 pairs: `3,312`
- Final expanded training pairs: `9,364`
- Runtime: about `0.94` seconds

This attempt succeeded at increasing coverage from `train.csv`, but the newly added pairs are more heuristic than the original baseline and should be treated as an expansion set rather than automatically assumed to be equally clean.

## Try 4.1: Cleaner Sentence Expansion

In **Try 4.1**, we kept the same overall goal as Try 4, but tried to make the added sentence pairs safer and less noisy.

### Strategy

- Start from the Try 4 expansion logic rather than from scratch
- Score each newly proposed pair instead of accepting all pairs that pass basic filters
- Use baseline-quality statistics from the original `6,052` clean pairs as a reference range
- Reject pairs with suspicious source-target length ratios, broken punctuation, chopped English fragments, or weak Akkadian chunks
- Keep only the higher-confidence additions, even if the final increase is smaller than `+3,312`

### Outputs

- `train_folder/try4_1_train.src`
- `train_folder/try4_1_train.tgt`
- `train_folder/try4_1_added_only.csv`
- `train_folder/try4_1_read.md`
- `train_folder/try4_1_process.log`

### Current Result

Try 4.1 completed a full local run with the stricter scoring layer:

- Original clean baseline pairs: `6,052`
- Additional cleaner Try 4.1 pairs: `3,603`
- Final expanded training pairs: `9,655`
- Runtime: about `2.04` seconds

Try 4.1 is somewhat more selective than Try 4 in how it scores candidates, but it still produces a fairly large heuristic add-on set. It is best treated as a cleaner expansion attempt, not as guaranteed baseline-quality data.

## Try 4.2: High-Confidence Sentence Expansion

In **Try 4.2**, we pushed the filtering further and aimed for a smaller, higher-confidence add-on set than Try 4.1.

### Strategy

- Reuse the Try 4.1 scoring pipeline as the proposal stage
- Tighten confidence thresholds further
- Penalize suspicious short-source / polished-English combinations more strongly
- Prefer additions whose source/target lengths sit near the core baseline distribution, not just inside the broad range
- Keep the added set intentionally smaller if that improves trustworthiness

### Outputs

- `train_folder/try4_2_train.src`
- `train_folder/try4_2_train.tgt`
- `train_folder/try4_2_added_only.csv`
- `train_folder/try4_2_read.md`
- `train_folder/try4_2_process.log`

### Current Result

Try 4.2 completed a full local run with the tighter high-confidence gate:

- Original clean baseline pairs: `6,052`
- Additional high-confidence Try 4.2 pairs: `3,487`
- Final expanded training pairs: `9,539`
- Runtime: about `2.21` seconds

Try 4.2 is more selective than Try 4.1 in how it defines the core baseline band and high-confidence cutoff, but it still remains a heuristic expansion set rather than guaranteed baseline-quality data.

### Current Working Dataset

For the current training workflow, the selected working dataset is the **Try 4.2 expanded corpus**:

- Active source file: `train_folder/try4_2_train.src`
- Active target file: `train_folder/try4_2_train.tgt`
- Working training-pair count: `9,539`

Later attempts such as Try 5 and Try 5.1 remain useful as separate experiments, but for now the main dataset to move forward with is the Try 4.2 version.

## Try 5: Metadata-Driven Archive Expansion

In **Try 5**, the focus shifts away from blind OCR mining and away from purely heuristic sentence splitting. Instead, the idea is to expand the dataset by linking clean Akkadian transliterations to publication-derived English translations through stable metadata.

### Core Idea

The main principle is:

1. trust `published_texts.csv` for the Akkadian transliteration side
2. extract translations from selected publication material
3. connect both sides using text identifiers and archive metadata
4. build a new supplemental parallel corpus from those matched texts

This is meant to be a more structured and higher-precision route than trying to recover sentence pairs directly from noisy OCR pages.

### Why Try 5 Is Different

Earlier OCR attempts tried to detect Akkadian-like lines and nearby English lines automatically inside `publications.csv`. That approach struggled because:

- OCR noise damaged both Akkadian and translation lines
- nearby English text often turned out to be commentary rather than direct translation
- line-level fuzzy matching was too brittle

Try 5 changes the unit of work from **line-level OCR guessing** to **text-level archive matching**.

### What Try 5 Uses

- `published_texts.csv` as the trusted source of Akkadian transliterations
- `publications.csv` as the source of publication-derived translations or translation-bearing text
- metadata fields such as:
  - `aliases`
  - `excavation_no`
  - `note`
  - publication-specific text identifiers
- `train.csv` as a reference for what is already present and what may still be missing

### Process

1. **Start from one publication or archive at a time**
   Instead of scanning the entire OCR dump at once, isolate one PDF/archive family at a time.

2. **Collect text identifiers from the publication side**
   These may be text labels, excavation numbers, archive IDs, or alias-like references present in the publication material.

3. **Normalize identifiers**
   Lowercase, strip punctuation, normalize spacing, and simplify inconsistent formatting before matching.

4. **Search `published_texts.csv` using metadata**
   Match publication-side identifiers against:
   - `aliases`
   - `excavation_no`
   - `note`
   - related catalog-like fields when useful

5. **Build text-level parallel pairs**
   Once a publication text is matched to a `published_texts.csv` record:
   - take the clean transliteration from `published_texts.csv`
   - pair it with the corresponding translation extracted from the publication side

6. **Keep the supplemental data separate first**
   Save these new matches into a separate dataset rather than immediately mixing them into the original train set.

7. **Only later consider sentence splitting**
   After trustworthy text-level pairs exist, they can be split into smaller units in a second stage if needed.

### Why This May Work Better

- Metadata identifiers are more stable than noisy OCR strings
- `published_texts.csv` already gives the cleaner Akkadian side
- publication-level linking is less brittle than line-level OCR alignment
- even semi-manual or partially curated additions can be high-value in a low-resource setting

### Outputs

- `train_folder/try5_train_plus.csv`
- `train_folder/try5_read.md`
- `train_folder/try5_process.log`

### Current Result

Try 5 produced a first archive-level supplemental corpus:

- Total matched supplemental texts: `18`
- Alias-driven archive matches: `12`
- Excavation-number matches: `6`
- New records relative to `train.csv`: `18`
- Runtime: about `0.18` seconds

This is different from the OCR attempts: the output is a **separate archive-level parallel set**, not yet mixed into the baseline sentence corpus. That makes it easier to inspect and later decide whether to keep it as document-level data or split it into smaller training units in a second stage.

## Try 5.1: Sentence-Level Expansion From Archive Matches

In **Try 5.1**, the archive-level pairs from Try 5 are turned into sentence-level additions. The goal is not to maximize count, but to keep only the safer sentence-sized pairs that can plausibly be merged with the original `train.src` / `train.tgt` baseline.

### Core Idea

The process is:

1. start from the 18 archive-level parallel pairs recovered in Try 5
2. split the English side into sentence-like units
3. assign proportional Akkadian chunks from the full transliteration
4. score each chunk against the quality profile of the original 6,052-pair baseline
5. keep only the higher-confidence sentence pairs as a separate expansion set

### Why Try 5.1 Is Useful

Try 5 gave document-level pairs, which are useful but awkward to mix directly into sentence-level MT training. Try 5.1 is the bridge step:

- it reuses the cleaner metadata-linked documents from Try 5
- it converts them into smaller sentence-sized units
- it applies stricter quality controls than the broader Try 4 family
- it keeps the additions inspectable before they are trusted as part of the main corpus

### What Try 5.1 Uses

- `train_folder/try5_train_plus.csv` as the trusted archive-level input
- `train_folder/train.src` and `train_folder/train.tgt` as the baseline quality reference
- punctuation-based English sentence splitting
- proportional Akkadian chunk allocation
- confidence scoring based on transliteration quality, English quality, and length-ratio fit

### Process

1. **Load the 18 archive-level pairs**
   Use the metadata-linked transliteration and translation pairs produced in Try 5.

2. **Split the English side into sentence-like segments**
   Use punctuation boundaries to create sentence candidates from the translation side.

3. **Allocate Akkadian chunks proportionally**
   Divide the full transliteration across the English sentence segments based on their relative length.

4. **Filter obvious noise**
   Drop pairs that are empty, mostly punctuation, or have extreme source-target imbalance.

5. **Score against the baseline corpus**
   Compare candidate chunk lengths and source-target ratios to the original 6,052 training pairs.

6. **Keep only higher-confidence additions**
   Save accepted sentence pairs separately, then build a merged corpus only after inspection.

### Outputs

- `train_folder/try5_1_train.src`
- `train_folder/try5_1_train.tgt`
- `train_folder/try5_1_added_only.csv`
- `train_folder/try5_1_read.md`
- `train_folder/try5_1_process.log`

### Current Result

Try 5.1 produced a smaller sentence-level expansion set from the 18 archive matches:

- Archive-level input pairs: `18`
- Sentence candidates considered: `130`
- New sentence-level additions kept: `76`
- Final merged total (baseline + Try 5.1): `6,128`
- Runtime: about `0.15` seconds

This attempt is more conservative than the broader train expansion attempts. It only works on the metadata-linked archive pairs from Try 5 and keeps the accepted additions in a separate file for inspection.
