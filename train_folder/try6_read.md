# Try 6 External Parallel Import

## Scope
- Attempt: Try 6
- Goal: expand the current Try 4.2 working dataset using an externally prepared line-aligned Akkadian-English corpus
- Active baseline before import: `try4_2_train.src` / `try4_2_train.tgt`

## Techniques Used
- Imported pre-aligned transliteration and translation files directly by row index
- Used the plain-text `train`, `valid`, and `test` splits rather than tokenized variants
- Deduplicated the external corpus internally before comparison
- Removed rows already present in the active Try 4.2 set
- Applied simple safety filters: non-empty rows, minimum token length, and source-target ratio bounds
- Stored the imported rows separately before creating the merged corpus

## Results
- External raw pairs loaded: 56160
- External unique pairs after internal deduplication: 51699
- Exact duplicates already present in Try 4.2: 0
- Additional rows filtered out by safety rules: 15117
- New imported pairs kept: 36582
- Kept from train split: 32959
- Kept from valid split: 1782
- Kept from test split: 1841
- Final merged total (Try 4.2 + Try 6): 46121
- Runtime: 1.76 seconds

## Sample Pairs
- train:1 (train)
  akkadian=nunuz bal-til {ki} šu-qu-ru na-ram {d}-... {d}-še-ru-a ab ba ... pi-tiq {d}-nin-men-na ša₂ a-na be-lut kur.kur ... an ti al šid ... ir-bu-u₂ a-na lugal-u₂-ti gir₃.nita₂ ... ma mu-ṣib ša₃.igi.guru₆.meš a-na ... šu-ri-in-ni zi-ka-ru dan-nu nu-ur kiš-šat un.meš-šu e-tel ... kal mal-ki ... ti da-i-pu ga-re-e-šu guruš qar-du sa-pi-nu ... na-ki-ri ša₂ hur-sa-a-ni et-gu-ru-ti ki-ma qe₂-e u₂-sal-li-tu-ma u₂-... u₂ ...
  english=precious scion of baltil (aššur), beloved of the god(dess) (dn and) šērūa, ... , creation of the goddess ninmena, who (... ) ... for the dominion of the lands, (... ) who grew up to be king, ... (... ) governor, (... ) ... , the one who increases voluntary offerings for ... , ... (... ) of emblems, (5) powerful male, light of all of his people, lord of (... ) all rulers ... , the one who overwhelms his foes, valiant man, the one who destroys (... ) enemies, who cuts (straight) through interlocking mountains like a (taut) string and ...
- train:2 (train)
  akkadian=qu-ra-du ... u₂-šak-ni-šu₂ še-pu-uš-šu ... ina {giš}-tukul u₂-šam-qi-tu ... mu-ut-tal-ku ...
  english=warrior ... who made ... bow down at his feet ... , who put ... to the sword (lit. “weapon”), ... circumspect ... ,
- train:3 (train)
  akkadian=... u₂-ša₂-aš₂-ši-qa gir₃.ii-šu ... ša₂-de-e ... qab-li ... ugu gi-mir lugal.meš a-šib bara₂.meš ... mu-ut-tal-ki ... u₂-šum₂-gal-lum ṣi-i-ru ... da-ad₂-me
  english=... he made ... kiss his feet ... mountains ... in/of battle ... he (a god) made my weapon/rule greater than all of those/the kings who sit on (royal) daises, (5) ... circumspect ... , ... exalted lion-dragon, ... inhabited world.
- train:4 (train)
  akkadian=u₂-za-ʾi-in-šu₂-nu-ti-ma a-na kur-šu-nu il-li-ku uru.meš šu-a-tu-nu a-na eš-šu-ti du₃-uš i-na ugu du₆ kam-ri ša₂ {uru}-hu-mut i-qab-bu-šu-ni uru du₃-uš ul-tu uš-še-šu a-di gaba-dib-be₂-e-šu ar-ṣi-ip u₂-šak-lil e₂.gal mu-šab lugal-ti-ia i-na lib₃-bi ad-di {uru}-kar-aš-šur mu-šu ab-bi {giš}-tukul aš-šur en-ia i-na lib₃-bi ar-me un.meš kur.kur ki-šit-ti šu.ii-ia i-na lib₃-bi u₂-še-šib gun ma-da-at-tu u₂-kin-šu-nu-ti it-ti un.meš kur aš-šur am-nu-šu₂-nu-ti
  english=i adorned them (statues of the gods) and they (the gods) went (back) to their land. i rebuilt those cities. i built a city on top of a tell (lit. “a heaped-up ruin mound”) called ḫumut. i built (and) completed (it) from its foundations to its parapets. inside (it), i founded a palace for my royal residence. i named it kār-aššur, set up the weapon of (the god) aššur, my lord, therein, (and) settled the people of (foreign) lands conquered by me therein. i imposed upon them tax (and) tribute, (and) considered them as inhabitants of assyria.
- train:5 (train)
  akkadian={id₂}-pa-at-ti-{d}-en.lil₂ ša ul-tu ud.meš ru-qu-u₂-ti na-da-at-ma
  english=i dug out the patti-enlil canal, which had lain abandoned for a very long time and
