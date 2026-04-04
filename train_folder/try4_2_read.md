# Try 4.2 High-Confidence Expansion Log

## Scope
- Attempt: Try 4.2
- Goal: produce a smaller, higher-confidence train.csv expansion set than Try 4.1
- Strategy: keep only additions that are close to the core baseline distribution and score strongly on both source and target quality

## Techniques Used
- Reused the baseline 6,052 clean pairs without modification
- Reused English-anchored proportional splitting only to propose candidate additions
- Tightened the acceptable source/target distribution to the middle band of the baseline corpus
- Penalized suspicious short-source / polished-English combinations more aggressively
- Required stronger transliteration-like surface patterns and stronger English sentence quality
- Kept only high-confidence additions

## Baseline Core Band
- Source token band: 3..20
- Target token band: 6..30
- Ratio band: 0.615..4.800
- Median ratio: 1.667

## Results
- Original clean baseline pairs: 6052
- Additional high-confidence Try 4.2 pairs: 3487
- Final expanded training pairs: 9539
- Runtime: 2.21 seconds

## Sample Added Pairs
- SRC: me-et 17 gú urudu sig₅ ha-bu-lam
- TGT: he owes me 17 talents of copper of good quality.
  confidence=22, src_len=6, tgt_len=10, ratio=1.667
- SRC: na-pá-al-tù-šu šu-up-ra-nim ù 5 gú urudu
- TGT: he must also pay 5 talents of copper of šu-ištar.
  confidence=22, src_len=6, tgt_len=10, ratio=1.667
- SRC: ša sà-li-bu ha-bu-lu-ni a-dí-ni ú-lá ni-im-hu-ur
- TGT: the silver that salibu owes we have not yet received.
  confidence=22, src_len=6, tgt_len=10, ratio=1.667
- SRC: ku-<gap> lu-ub-lam <gap>-nu-<gap> i ha-hi-im ra
- TGT: the servant of the s. of kasiya should bring it.
  confidence=22, src_len=6, tgt_len=10, ratio=1.667
- SRC: urudu a-na ha-nu a ší-a-ma-tim a-dí-in
- TGT: i gave 10 minas of copper to hanu for purchases.
  confidence=22, src_len=6, tgt_len=10, ratio=1.667
- SRC: a-dí-šu-um té-er-ta-kà li-li-kam-ma a-šar pá-nu-im pá-ni-ší
- TGT: send me a word and dispose of it wherever possible.
  confidence=22, src_len=6, tgt_len=10, ratio=1.667
- SRC: 5 gú urudu iš-tí ku-ur-zi-a ù
- TGT: 5 talents of copper is owed by kurziya and šu-nisaba.
  confidence=22, src_len=6, tgt_len=10, ratio=1.667
- SRC: ub-lam urudu nu-sá-ni-iq-ma 1 gú-tum 1
- TGT: today we checked your seal, and 8 shekels were missing.
  confidence=22, src_len=6, tgt_len=10, ratio=1.667
- SRC: ší-sí-a ù am-ma ša ki-ma i-a-tí
- TGT: let my brother šalim-aššur come, and then settle with him.
  confidence=22, src_len=6, tgt_len=10, ratio=1.667
- SRC: ma-ah-ri-kà-ma iṣ-bu-tù a-ma-kam ṣa-ba-sú-nu-ma urudu li-dí-nu-ni-ku-um
- TGT: seize them there and have them give you the copper.
  confidence=22, src_len=6, tgt_len=10, ratio=1.667
