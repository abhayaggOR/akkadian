# Try 4 Train Expansion Log

## Scope
- Attempt: Try 4
- Goal: expand sentence pairs from `train.csv` beyond the original 6,052 clean pairs
- Strategy: keep the original baseline intact, then add only new high-confidence pairs from a second sentence-splitting pass

## Techniques Used
- Preserved the original newline/period baseline split as the backbone corpus
- Applied transliteration normalization before the second pass (`ḫ/h`, `sz/š`, gap cleanup)
- Split English documents into sentence-like units using punctuation anchors
- Allocated Akkadian source spans proportionally based on English sentence lengths
- Appended leftover Akkadian tokens to the last chunk to avoid truncation
- Re-filtered new pairs using minimum token thresholds, source-target length ratio, punctuation/numeric noise checks, and copy-noise detection
- Added only pairs that were genuinely new compared with the original 6,052 baseline

## Results
- Original clean baseline pairs: 6052
- Additional high-confidence Try 4 pairs: 3312
- Final expanded training pairs: 9364
- Runtime: 0.94 seconds

## Sample Added Pairs
- SRC: gín kù.babbar sig₅ i-ṣé-er puzur₄-a-šur dumu a-ta-a a-lá-hu-um i-šu
- TGT: puzur-aššur son of ataya owes 22 shekels of good silver to ali-ahum.
- SRC: iš-tù ha-muš-tim ša ì-lí-dan itu.kam ša ke-na-tim li-mu-um e-na-sú-in a-na itu 14 ha-am-ša-tim i-ša-qal
- TGT: reckoned from the week of ilī-dan, month of ša-kēnātim, in the eponymy of enna-suen, he will pay in 14 weeks.
- SRC: túg u-la i-dí-na-ku-um
- TGT: <gap> he did not give you a textile.
- SRC: i-tù-ra-ma 9 gín kù.babbar
- TGT: he returned and 9 shekels of silver <gap>
- SRC: kišib šu-{d}en.líl dumu šu-ku-bi-im kišib ṣí-lu-lu dumu
- TGT: seal of šu-illil son of šu-kūbum, seal of ṣilūlu son of uku.
- SRC: ú-ku i-nu-mì i-dí-a-bu-um a-wa-sú iq-bi-ú 10 ma-na kù.babbar a-na
- TGT: when iddin-abum spoke his will, he gave 10 minas ofדsilver to šalim-aššur.
- SRC: ša-lim-a-šùr i-dí-in um-ma šu-ut-ma i-ṣí-ba-at kù-pì-a li-il₅-qé
- TGT: he said: he may take it from the interest on my silver.""
- SRC: <gap> ša lá ta-ha-dì-ri a-na ištar-lá-ma-sí qí-bi₄-ma šu-ma a-ha-tí
- TGT: to ištar-lamassī: if you are truly my sister, then encourage her.
- SRC: a-ta li-ba-am
- TGT: do not fear.
- SRC: dì-ni-ší-im lá ta-ha-da-ar a-na ni-ta-ah-šu-šar qí-bi₄-ma
- TGT: to nitahšušar: air the -textiles that i left.
