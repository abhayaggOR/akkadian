# Try 3 OCR Extraction Log

## Scope
- Attempt: Try 3
- Goal: extract new Akkadian-English pairs from `publications.csv` using linguistic validation and reference anchoring
- Priority: precision over quantity

## Phase 1 Learned Structure
- Common sentence starters tracked: 50
- Top 15 starters: 0, 1, 10, 15, 1666, 2, 25, 3, 3333, 4, 5, 6, 6666, 75, 8333
- Length range: min=1, max=120, avg=10.01
- Detection window used: 1..30

## Phase 2-9 Extraction Metrics
- Pages scanned with `has_akkadian == true`: 31286
- Pages with alias hit: 31074
- Akkadian candidates detected: 116
- Reference matches retained: 0
- Final extracted pairs: 0
- Runtime: 326.44 seconds
- Multilingual note: non-English candidates were detected heuristically, but automatic translation was not available in this runtime, so only English-valid lines were retained.

## Sample Pairs
