# Try 4.1 Cleaner Expansion Log

## Scope
- Attempt: Try 4.1
- Goal: produce a smaller but cleaner train.csv expansion set than Try 4
- Strategy: score candidate additions against baseline-quality ranges and keep only high-confidence pairs

## Techniques Used
- Reused the original 6,052 baseline as the untouchable clean backbone
- Reused the English-anchored proportional splitter from Try 4 only as a proposal generator
- Computed baseline token-length and source-target ratio ranges from the original clean corpus
- Rejected additions that fell outside the safer baseline distribution band
- Scored English quality using stopwords, alphabetic ratio, punctuation completeness, and corruption checks
- Scored Akkadian chunk quality using transliteration-like surface patterns, hyphen density, and gap hygiene
- Kept only pairs with high final confidence

## Baseline Reference Range
- Source token band: 3..35
- Target token band: 4..47
- Ratio band: 0.312..8.600
- Average ratio: 2.753

## Results
- Original clean baseline pairs: 6052
- Additional cleaner Try 4.1 pairs: 3603
- Final expanded training pairs: 9655
- Runtime: 2.04 seconds

## Sample Added Pairs
- SRC: i-šu-šu-um šu-ma lá i-da-na-ku-nu-tí sí-ku-šu
- TGT: if he does not give it to you then detain him.
  confidence=18, src_len=5, tgt_len=11, ratio=2.2
- SRC: ú bé-la-am ú-la i-šu a-ša
- TGT: don't be angry because i did not turn up with you.
  confidence=18, src_len=5, tgt_len=11, ratio=2.2
- SRC: né-ri-šu-ma lá i-dí-ni-a-tí ku-lu-ni nu-ta-ni-ih-ma
- TGT: all of us have been toiling, but we are now well.
  confidence=18, src_len=5, tgt_len=11, ratio=2.2
- SRC: mì-ma ṭup-pí a-ni-ú-tim a-na qá-tí
- TGT: i left all these tablets in the hands of the maid.
  confidence=18, src_len=5, tgt_len=11, ratio=2.2
- SRC: ù-lá ub-lam i-na a-lá-ki-šu ší-im-ší-na
- TGT: when he comes i shall send you the proceeds from them.
  confidence=18, src_len=5, tgt_len=11, ratio=2.2
- SRC: nu-sá-ni-iq-ma 47 gú urudu ṣa-lá-mu-um
- TGT: thereof 7 talents of copper was used for our father's grave.
  confidence=18, src_len=5, tgt_len=11, ratio=2.2
- SRC: ṣí-bi-it ni-ga-lim i-da-an ṭup-pá-am ša
- TGT: seize them and make them pay the silver and its interest.
  confidence=18, src_len=5, tgt_len=11, ratio=2.2
- SRC: 21 ma-na 10 gín urudu
- TGT: sell it for silver, seal the silver and let it remain.
  confidence=18, src_len=5, tgt_len=11, ratio=2.2
- SRC: ma-nim lá-aq-bi sà-ah-ra-ku-ma ú-lá a-le-e-ma
- TGT: i am staying here and i am unable to answer for it.
  confidence=18, src_len=5, tgt_len=12, ratio=2.4
- SRC: e-ri-iš-kà lá ta-da-an um-ma a-ta-ma
- TGT: if over there he asks you for silver, do not give it.
  confidence=18, src_len=5, tgt_len=12, ratio=2.4
