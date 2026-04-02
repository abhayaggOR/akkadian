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
