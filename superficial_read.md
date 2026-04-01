# Deep Past Challenge Processing Log (Superficial)

- **Phase 1-3**: Extracted and cleaned Akkadian and English pairs from `train.csv`. Wrote into `train_folder/train.src` and `train.tgt`. 
- **Phase 4**: Supplemented the corpus using `published_texts.csv` (only transliterated sentences). Wrote `train_folder/corpus.src` for tokenizer training.
- **Phase 5-7**: Parsed OCR output from `publications.csv` dynamically to map English blocks close to Akkadian text segments. Extracted new parallel alignments and combined everything into `train_folder/final_train.src` and `final_train.tgt`.

All in-depth logs, sample sentences, lengths, and metrics have been stored within `train_folder/read.md`. 
The Python execution script is located at `train_folder/process_data.py`.
