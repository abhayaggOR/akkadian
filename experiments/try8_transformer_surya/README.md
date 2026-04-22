## Try 8 Transformer Experiment

Standalone Transformer training run migrated from the `NLP` workspace into this repo's experiment layout.

Artifacts:

- `best_model.pt`: best checkpoint saved by validation loss
- `train_history.csv`: per-epoch train/validation metrics
- `try8_run.log`: full detached training log
- `test_metrics.json`: final test-set metrics parsed from the completed run

Summary:

- training script: `src/training/train_try8_transformer.py`
- stopped by early stopping after epoch 160
- best validation loss: epoch 150
- best validation BLEU: epoch 159
- best validation chrF++: epoch 160

Final test results:

- greedy BLEU: 20.60
- greedy chrF++: 47.80
- beam-4 BLEU: 25.95
- beam-4 chrF++: 49.93
