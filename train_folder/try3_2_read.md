# Try 3.2 OCR Extraction Log

## Scope
- Attempt: Try 3.2
- Goal: improve OCR-to-reference alignment using retrieval plus reranking instead of only direct fuzzy matching
- Priority: still precision-first, but with broader normalized retrieval before rejection

## Key Changes From Try 3.1
- Added OCR cleanup and transliteration normalization before matching
- Built 1-line, 2-line, and 3-line reference windows from `published_texts.csv`
- Indexed reference windows by informative transliteration tokens
- Retrieved top candidate windows using rare-token overlap before reranking
- Reranked with weighted token overlap, canonical-string similarity, and prefix bonus
- Deferred nearby English alignment until after a reference window was accepted

## Metrics
- Pages scanned with `has_akkadian == true`: 31286
- Akkadian candidates detected: 6428
- Retrieval hits: 6428
- Reference matches retained: 18
- Nearby English matches retained: 0
- Final extracted pairs: 0
- Runtime: 781.55 seconds
- Multilingual note: this runtime still keeps only English-valid nearby lines because automatic translation is not available locally.

## Sample Pairs
