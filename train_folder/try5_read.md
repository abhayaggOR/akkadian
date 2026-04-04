# Try 5 Archive Expansion Log

## Scope
- Attempt: Try 5
- Goal: build a supplemental parallel corpus using metadata-driven archive matching rather than blind OCR line mining
- Strategy: pair trusted transliterations from `published_texts.csv` with publication-derived translations through aliases and excavation identifiers

## Techniques Used
- Worked archive-by-archive instead of scanning the full OCR dump indiscriminately
- Matched clean transliterations through metadata keys rather than fuzzy transliteration-only alignment
- Normalized aliases before exact matching
- Used excavation numbers as a second stable identifier source
- Cleaned the translation side by removing line numbers, running headers like `lo.e.`/`rev.`/`u.e.`/`l.e.`, and broken line wraps
- Kept the resulting corpus separate as a supplemental archive-level dataset

## Results
- Total matched supplemental texts: 18
- Alias-driven archive matches: 12
- Excavation-number matches: 6
- New records relative to `train.csv`: 18
- Runtime: 0.18 seconds

## Sample Records
- KEY: Cct 5, 6a
  oare_id=7a563f13-abf1-0749-ced9-41ed3f3fb851, new_vs_train=true
  transliteration=um-ma a-šùr-i-dí-ma a-na a-la-ḫi-im e-la-ma pu-šu-ke-en6 ù a-šur-ta-ak-la-ku qí-bi-ma a-ḫu-a a-tù-nu iš-tù MU.ŠÈ 30 i-na a-lim{ki} wa-áš-ba-ku-ma ù ni-kà-sí ù-ša-qal kà-ra-am mu-ùḫ-ra-ma iš-tí kà-ri-im e-na-na-tim er-ša-
  translation=From Aššur-idi to Alâhum, Elamma, Pnsu-kén and Aššur-taldâku: My dear brothers, for 30 years I have been living in the City and I always pay the accounts. Appeal to the colony authorities and ask for clemency for me from
- KEY: POAT 5 | L29-558
  oare_id=d546d6e3-fd34-4c32-a586-78989a79ecbe, new_vs_train=true
  transliteration=um-ma a-šùr-i-dí-ma a-na a-mur-IŠTAR a-la-ḫi-im ì-lí-a-lim ù a-šùr-ta-ak-lá-ku qí-bi4-ma 2 GÚ 1 ma-na AN.NA-ki ku-nu-ku 38 TÚG.ḪI.A ú-ṣú-ur-ša-a-šùr ub-lá-ku-nu-tí ig-ri sà-ri-dim ša-bu mì-ma lá ta-da-na-šu-um i-na pá-ni
  translation=From Aššur-idi to Amur-Istar, Ala.hum, Ili-alum and Aššur-taklaku: talents 10 minas of tin under seal (and) 38 textiles Usur-sa-Aššur has brought to you. He has been paid the wages for a donkeydriver; do not give him any
- KEY: RC 1749B
  oare_id=a14e6a2d-0a8c-417c-829e-40d0fa91e98d, new_vs_train=true
  transliteration=um-ma a-šur-i-dí-ma a-na a-lá-ḫi-im a-šur-na-da ì-lí-a-lim ù a-šur-ta-ak-lá-ku qí-bi-ma 8 {TÚG}ku-ta-nu i-a-ú-tum 1.5 ma-na AN.NA a qá-tí-šu a-dí-in ḫa-mu-uš-tí ANŠE mì-ma a-nim i-a-um 1 {TÚG}ku-ta-nu-um ša ṭá-áb-ṣí-li-a
  translation=From Aššur-idi to Alâhum, Assur-nâdâ, Ili-âlum and Aššur-taklâku: The 8 textiles are mine; I gave him 1 mina of tin for expenses; one-fifth of a donkey—all this belongs to me. 1 kutânu-textile belongs to Tab-sill-Aššur; 
- KEY: Cct 4, 1b
  oare_id=17cf8012-b381-6fc7-d72e-39bd1d3964aa, new_vs_train=true
  transliteration=um-ma a-šùr-i-dí-ma a-na ì-lí-SUKKAL ù a-šùr-ta-ak-lá-ku qí-bi4-ma a-na ì-lí-ŠU-GAN qí-bi4-ma ma-áš-ki ša-pá-tim a-dì-na-kum um-ma a-ta-ma a-na ru-ba-im a-qí-áš ù 2 ma-na ḫu-ša-e uṣ-ba-ku-um um-ma a-ta-ma pá-šu-ra-am SIG
  translation=From Aššur-idi to Ili-sukkal and Aššur-taklâku: (specifically) to Ill-sukkal: I gave you some woollen fleeces. You said: "I shall give them as a gift to the king." I furthermore added for you 2 minas of scrap metal. You 
- KEY: TC 3, 95
  oare_id=2cb3d8ef-f49f-8419-6238-684ea925da30, new_vs_train=true
  transliteration=um-ma a-šùr-i-dí-ma a-na PUZUR4.IŠTAR ù ás-qú-dim qí-bi4-ma 1 GÚ 5 ma-na AN.NA ku-nu-ki ša ás-qú-dum na-áš-ú a-na-ma tí-me-el-ki-a ta-kà-ša-dá-ni um-ma a-tù-nu-ma a-na šu-IŠTAR DUMU a-zu-da qí-bi4-a-ma té-er-tí a-šùr-i-d
  translation=From Aššur-idi to Puzur-Ištar cpd Asgndum: As to the 1 talent 5 minas of tin under seal which Asqùdum is bringing, as soon as you reach Timelkia you must say as follows to Sû-Ištar, the son of Azuda: "Aššur-idi's orders 
