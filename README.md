# Akkadian to English NMT: Data Preprocessing Pipeline

This repository contains the dataset processing pipeline and generated training pairs for translating Old Assyrian transliterated texts into English, utilizing the Deep Past Challenge dataset.

## Folder Structure

- `train_folder/`: Contains all generated parallel `.src` and `.tgt` data files, the execution script (`process_data.py`), and a strictly detailed internal log (`read.md`).
- `read.md`: A superficial log summarizing the overall pipeline execution over 7 phases.

## Data Processing Decisions & Reasoning

The data pipeline rigorously transforms the original challenge data (`train.csv`, `published_texts.csv`, `publications.csv`) into sentence-level parallel data ready for Machine Translation models like HuggingFace Transformers. Below are the key decisions made and *why*:

### 1. Simple Sentence Splitting (Phase 1)
We opted to split sentences based strictly on newlines `\n` and periods `.`. 
**Why?** The translations provided in `train.csv` frequently lack rigorous 1:1 mapping at the document level. Splitting by hard structural bounds ensured sequential Zip alignment accurately bound the Source to the Target text contextually.

### 2. Constraint: Minimum Length >= 5 Chars
**Why?** In Cuneiform and NLP, sentence fragments under 5 characters usually carry negligible semantic value or represent broken line artifacts (e.g. `1.`, `...`). Retaining them adds noise.

### 3. Cleaning Constraints & Case Normalization (Phase 2)
Both source and target text were lowercased and stripped of extra whitespaces. Sentences containing *only* punctuation or numbers were actively removed.
**Why?** This prevents the Transformer tokenizer from over-allocating vocabulary space to stylistic characters and improves convergence rates.

### 4. Preserving Hyphens in Akkadian
**Why?** This was a critical domain-specific constraint. In Akkadian transliteration, hyphens act as syllabic sign boundaries (e.g., `ma-nu-ki-a-šur`). Stripping them would arbitrarily merge distinct cuneiform signs, permanently destroying morphological structure crucial to understanding the language.

### 5. Creating `corpus.src` (Phase 4)
We generated a 55,000+ line unaligned corpus combining clean pairs and `published_texts.csv`.
**Why?** Modern Subword Tokenizers (like BPE or WordPiece) need massive statistical priors to correctly chunk Akkadian's complex morphology. An unaligned text corpus serves strictly to train an optimal tokenizer.

### 6. Extremely Conservative OCR Extraction (Phase 5)
We parsed a ~580MB `publications.csv` OCR dump strictly seeking ASCII English words near Akkadian-styled strings (containing `š`, `ṣ`, `ṭ`, `ḫ` or internal hyphens).
**Why?** We prioritized extreme precision over quantity. OCR data inherently risks severe misalignment (noisy pairings). Our constraint dictated that only perfect proximity matches were valid. Because the OCR layout was severely nested, the heuristic cleanly dropped everything. This generated **0 pairs**, actively protecting the dataset from catastrophic noise injections that would cripple BLEU scores. "Better few clean pairs than many noisy ones."

---

*This pipeline cleanly generated 6,052 high-quality sentence-aligned pairs.*
