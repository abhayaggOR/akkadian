# Try 3.1 OCR Extraction Log

## Scope
- Attempt: Try 3.1
- Goal: relax the anchor stage from Try 3 without abandoning precision
- Core refinement: compare OCR candidates against sentence-split reference lines instead of full transliterations

## Key Changes From Try 3
- Lowered lexicon coverage gate from ultra-strict to moderate (`>= 40%`)
- Indexed sentence-level references from `published_texts.csv`
- Allowed alias-backed weaker similarity matches when the page anchor is strong
- Expanded translation search window to `+-5` lines, while keeping nearest English-valid line only
- Preserved precision-first final confidence threshold

## Metrics
- Pages scanned with `has_akkadian == true`: 31286
- Akkadian candidates detected: 1164
- Reference matches retained: 0
- Final extracted pairs: 0
- Runtime: 217.36 seconds
- Multilingual note: this runtime still keeps only English-valid nearby lines because automatic translation is not available locally.

## Sample Pairs
