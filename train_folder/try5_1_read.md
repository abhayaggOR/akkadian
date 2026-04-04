# Try 5.1 Sentence Expansion

## Scope
- Attempt: Try 5.1
- Input: archive-level metadata matches from `try5_train_plus.csv`
- Goal: convert the 18 document-level additions into cleaner sentence-level training pairs

## Techniques Used
- Split the English side into sentence-like units using punctuation boundaries
- Allocated Akkadian chunks proportionally by English sentence length
- Scored each candidate chunk against the original 6,052-pair baseline distribution
- Kept only chunks that passed transliteration quality, English quality, and length-ratio checks
- Stored the additions separately before mixing them into the merged corpus

## Results
- Archive-level input pairs: 18
- Sentence candidates considered: 130
- New sentence-level additions kept: 76
- Final merged total (baseline + Try 5.1): 6128
- Runtime: 0.15 seconds

## Sample Pairs
- KEY: Cct 4, 1b (confidence=20)
  akkadian=iš-tí tap-pá-e-a lá-pì-it ì-lí-šu-gan dumu na-ni ṣa-ba-at-ma pá-šu-ra-am pá-ni-a-ma ša-dí-in
  english=seize ili-sukkal son of nanni and make him hand over the good quality retail goods at once.
- KEY: POAT 5 | L29-558 (confidence=20)
  akkadian=pá-ni-im-ma šé-bi4-lam lu-qú-tám ša šé-pí-kà za-ki-a-ma tí-ib-a-ma
  english=clear the goods from your own caravan, set out and come here.
- KEY: POAT 5 | L29-558 (confidence=20)
  akkadian=ú-ṣú-ur-ša-a-šùr ub-lá-ku-nu-tí ig-ri sà-ri-dim ša-bu mì-ma lá ta-da-na-šu-um
  english=he has been paid the wages for a donkeydriver; do not give him anything.
- KEY: Cct 5, 6a (confidence=20)
  akkadian=a-i-a-tum ma-áš-kà-na-tù-a e-la ma-la ma-áš-kà-tám
  english=should i make another deposit on top of that?
- KEY: Cct 4, 1a (confidence=20)
  akkadian=e-ra-bi-kà-ma ki-ma šu-ul-mì-kà šé-bi4-lá-am šu-ma a-ḫi
  english=if you are truly my brother, do not make me angry.
