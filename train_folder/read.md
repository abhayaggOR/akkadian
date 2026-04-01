# Detailed Processing Log

## Phase 1: Convert train.csv into sentence-level parallel data
- Num pairs generated: 6092
- Sample pairs:
  - SRC: KIŠIB ma-nu-ba-lúm-a-šur DUMU ṣí-lá-{d}IM KIŠIB šu-{d}EN
  - TGT: Seal of Mannum-balum-Aššur son of Ṣilli-Adad, seal of Šu-Illil son of Mannum-kī-Aššur, seal of Puzur-Aššur son of Ataya
  - SRC: LÍL DUMU ma-nu-ki-a-šur KIŠIB MAN-a-šur DUMU a-ta-a 0
  - TGT: Puzur-Aššur son of Ataya owes 22 shekels of good silver to Ali-ahum
  - SRC: 3333 ma-na 2 GÍN KÙ
  - TGT: Reckoned from the week of Ilī-dan, month of Ša-kēnātim, in the eponymy of Enna-Suen, he will pay in 14 weeks
  - SRC: BABBAR SIG₅ i-ṣé-er PUZUR₄-a-šur DUMU a-ta-a a-lá-ḫu-um i-šu iš-tù ḫa-muš-tim ša ì-lí-dan ITU
  - TGT: If he has not paid in time, he will add interest at the rate 1
  - SRC: KAM ša ke-na-tim li-mu-um e-na-sú-in a-na ITU 14 ḫa-am-ša-tim i-ša-qal šu-ma lá iš-qú-ul 1
  - TGT: 5 shekel per mina per month

## Phase 2: Clean sentence pairs
- Pairs after cleaning: 6052
- 5 Sample pairs:
  - SRC: kišib ma-nu-ba-lúm-a-šur dumu ṣí-lá-{d}im kišib šu-{d}en
  - TGT: seal of mannum-balum-aššur son of ṣilli-adad, seal of šu-illil son of mannum-kī-aššur, seal of puzur-aššur son of ataya
  - SRC: líl dumu ma-nu-ki-a-šur kišib man-a-šur dumu a-ta-a 0
  - TGT: puzur-aššur son of ataya owes 22 shekels of good silver to ali-ahum
  - SRC: 3333 ma-na 2 gín kù
  - TGT: reckoned from the week of ilī-dan, month of ša-kēnātim, in the eponymy of enna-suen, he will pay in 14 weeks
  - SRC: babbar sig₅ i-ṣé-er puzur₄-a-šur dumu a-ta-a a-lá-ḫu-um i-šu iš-tù ḫa-muš-tim ša ì-lí-dan itu
  - TGT: if he has not paid in time, he will add interest at the rate 1
  - SRC: kam ša ke-na-tim li-mu-um e-na-sú-in a-na itu 14 ḫa-am-ša-tim i-ša-qal šu-ma lá iš-qú-ul 1
  - TGT: 5 shekel per mina per month

## Phase 3: Create training files (train.src, train.tgt)
- Lines written: 6052
- First 5 lines (train.src):
  - kišib ma-nu-ba-lúm-a-šur dumu ṣí-lá-{d}im kišib šu-{d}en
  - líl dumu ma-nu-ki-a-šur kišib man-a-šur dumu a-ta-a 0
  - 3333 ma-na 2 gín kù
  - babbar sig₅ i-ṣé-er puzur₄-a-šur dumu a-ta-a a-lá-ḫu-um i-šu iš-tù ḫa-muš-tim ša ì-lí-dan itu
  - kam ša ke-na-tim li-mu-um e-na-sú-in a-na itu 14 ḫa-am-ša-tim i-ša-qal šu-ma lá iš-qú-ul 1
- First 5 lines (train.tgt):
  - seal of mannum-balum-aššur son of ṣilli-adad, seal of šu-illil son of mannum-kī-aššur, seal of puzur-aššur son of ataya
  - puzur-aššur son of ataya owes 22 shekels of good silver to ali-ahum
  - reckoned from the week of ilī-dan, month of ša-kēnātim, in the eponymy of enna-suen, he will pay in 14 weeks
  - if he has not paid in time, he will add interest at the rate 1
  - 5 shekel per mina per month

## Phase 4: Create single corpus file for tokenizer from published_texts.csv
- Total lines in corpus: 55105
- Sample 5 lines:
  - babbar ù ṣí-ba-sú šé-bi-lá-nim ṣí-ib-tum lá i-ma-i-dam li-ba-ku-nu lá i-ma-ra-aṣ <gap>
  - 5 ma-na 3 gín kù
  - babbar ší-im é ša i-ku-nim um-mì-na-ra dam en-nam-a-šur kù
  - babbar a-na šu-iš-kà-na ta-dí-na é-tù ša šu-iš-ku-na ta-áš-ú-mu ša um-mì-na-ra la ta-qá-bi šu-iš-ku-na um-ma ší-it-ma é-tù-a a-na wa-ša-bu-tim ki-a-ma wa-áš-ba-at é-tù ša um-mì-na-ra kù
  - babbar ta-áš-qú-ul igi áb-ša-lim igi šu-pí-a-aḫ-šu igi ší-zu-ur

## Phase 5: Extract Akkadian-English pairs from publications.csv
- Number of extracted pairs: 0
- 5 Examples:

## Phase 6: Combine datasets
- Final number of sentence pairs: 6052
- 5 Sample pairs:
  - SRC: 8333 ma-na a qá-tí-a ta-dí šu
  - TGT: 8333 mina you deposited as my share
  - SRC: babbar a-na pá-ni-a ša a-ḫu-zu-ma tù-šé-bi₄-lá-ni 0
  - TGT: 5 mina of silver under my seal ennānum son of kuziya brought to you
  - SRC: babbar šu-{d}en
  - TGT: when we asked šu-illil son of mannum-kī-aššur for the 3 minas of silver he said: i took it in accordance with my certified tablet
  - SRC: a-na šu-ta-mu-zi e-lá-a a-ba-ba ù lá-ma-sí qí-bi-ma um-ma a-dí-da ù a-lá-ḫu-um-ma kišib ma-ma-ḫi-ir dumu a-mur-ištar kišib en-na-sú-in dumu i-dí-a-bi₄-im 18
  - TGT: to šu-tammuzi, elaya, abada and lamassī from adida and ali-ahum: under the seal of man-mahir son of amur-ištar (and) enna-suen son of iddin-abum itūr-ilī gave 18
  - SRC: gàr-ri-im a-sí-ma i-na igi me-ra um-me-a-nim kù
  - TGT: my dear son, the tablet concerning ṣill-adad's debt as well as one or two other tablets, and the tablet concerning bēlum-bāni are available - entrust those tablets there with uṣurānum as witness to puzurkunu and aluwa so they may bring them
- % from original: 100.00%
- % from extracted: 0.00%

## Phase 7: Prepare final dataset
- Total training pairs: 6052
- Average sentence length (source): 83.56 chars
- Average sentence length (target): 98.80 chars
