from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = REPO_ROOT / "train_folder"
EXPERIMENT_DIR = REPO_ROOT / "experiments" / "try7_lstm_baseline"
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_PATH = TRAIN_DIR / "try6_train.src"
TARGET_PATH = TRAIN_DIR / "try6_train.tgt"

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


def log(message: str, log_path: Path | None = None) -> None:
    print(message, flush=True)
    if log_path is not None:
        with log_path.open("a", encoding="utf-8") as file:
            file.write(message + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def choose_device(preferred: str) -> torch.device:
    if preferred == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preferred)


def load_pairs(source_path: Path, target_path: Path, max_pairs: int | None = None) -> list[tuple[str, str]]:
    src_lines = source_path.read_text(encoding="utf-8").splitlines()
    tgt_lines = target_path.read_text(encoding="utf-8").splitlines()
    pairs: list[tuple[str, str]] = []
    for src, tgt in zip(src_lines, tgt_lines):
        src_clean = clean_text(src)
        tgt_clean = clean_text(tgt)
        if not src_clean or not tgt_clean:
            continue
        pairs.append((src_clean, tgt_clean))
        if max_pairs is not None and len(pairs) >= max_pairs:
            break
    return pairs


def split_pairs(
    pairs: list[tuple[str, str]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    shuffled = pairs[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    train_end = int(len(shuffled) * train_ratio)
    val_end = train_end + int(len(shuffled) * val_ratio)
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


class Vocab:
    def __init__(self, stoi: dict[str, int], itos: list[str]) -> None:
        self.stoi = stoi
        self.itos = itos

    @classmethod
    def build(cls, texts: list[str], max_size: int, min_freq: int) -> "Vocab":
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(text.split())

        stoi: dict[str, int] = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
        itos = SPECIAL_TOKENS[:]
        for token, freq in counter.most_common():
            if freq < min_freq:
                continue
            if token in stoi:
                continue
            if len(itos) >= max_size:
                break
            stoi[token] = len(itos)
            itos.append(token)
        return cls(stoi=stoi, itos=itos)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.stoi[BOS_TOKEN])
        ids.extend(self.stoi.get(token, self.stoi[UNK_TOKEN]) for token in text.split())
        if add_eos:
            ids.append(self.stoi[EOS_TOKEN])
        return ids

    def decode(self, ids: list[int], stop_at_eos: bool = True) -> str:
        tokens: list[str] = []
        for idx in ids:
            token = self.itos[idx]
            if token == EOS_TOKEN and stop_at_eos:
                break
            if token in {PAD_TOKEN, BOS_TOKEN}:
                continue
            tokens.append(token)
        return " ".join(tokens)

    def to_json(self) -> dict[str, object]:
        return {"stoi": self.stoi, "itos": self.itos}


class ParallelDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[str, str]],
        src_vocab: Vocab,
        tgt_vocab: Vocab,
        max_src_len: int,
        max_tgt_len: int,
    ) -> None:
        self.examples: list[tuple[torch.Tensor, torch.Tensor]] = []
        for src, tgt in pairs:
            src_ids = src_vocab.encode(src, add_eos=True)[:max_src_len]
            tgt_ids = tgt_vocab.encode(tgt, add_bos=True, add_eos=True)[:max_tgt_len]
            if len(src_ids) < 2 or len(tgt_ids) < 3:
                continue
            self.examples.append(
                (
                    torch.tensor(src_ids, dtype=torch.long),
                    torch.tensor(tgt_ids, dtype=torch.long),
                )
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.examples[index]


@dataclass
class Batch:
    src: torch.Tensor
    src_lengths: torch.Tensor
    tgt: torch.Tensor


def collate_batch(examples: list[tuple[torch.Tensor, torch.Tensor]]) -> Batch:
    src_seqs, tgt_seqs = zip(*examples)
    src_lengths = torch.tensor([len(seq) for seq in src_seqs], dtype=torch.long)
    padded_src = pad_sequence(src_seqs, batch_first=True, padding_value=0)
    padded_tgt = pad_sequence(tgt_seqs, batch_first=True, padding_value=0)
    return Batch(src=padded_src, src_lengths=src_lengths, tgt=padded_tgt)


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        embedded = self.embedding(src)
        outputs, state = self.lstm(embedded)
        return outputs, state


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        embedded = self.embedding(inputs)
        outputs, state = self.lstm(embedded, state)
        logits = self.output(outputs)
        return logits, state


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src: torch.Tensor, tgt_inputs: torch.Tensor) -> torch.Tensor:
        _, state = self.encoder(src)
        logits, _ = self.decoder(tgt_inputs, state)
        return logits

    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor, max_len: int, bos_idx: int, eos_idx: int) -> list[list[int]]:
        _, state = self.encoder(src)
        batch_size = src.size(0)
        current = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=src.device)
        generated: list[list[int]] = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)

        for _ in range(max_len):
            logits, state = self.decoder(current, state)
            next_token = logits[:, -1, :].argmax(dim=-1)
            current = next_token.unsqueeze(1)
            for index, token in enumerate(next_token.tolist()):
                if not finished[index]:
                    generated[index].append(token)
                    if token == eos_idx:
                        finished[index] = True
            if finished.all():
                break
        return generated


def simple_corpus_bleu(predictions: list[list[str]], references: list[list[str]], max_n: int = 4) -> float:
    precisions: list[float] = []
    pred_length = 0
    ref_length = 0

    for n in range(1, max_n + 1):
        overlap = 0
        total = 0
        for pred_tokens, ref_tokens in zip(predictions, references):
            pred_ngrams = Counter(tuple(pred_tokens[i:i + n]) for i in range(max(0, len(pred_tokens) - n + 1)))
            ref_ngrams = Counter(tuple(ref_tokens[i:i + n]) for i in range(max(0, len(ref_tokens) - n + 1)))
            overlap += sum(min(count, ref_ngrams[gram]) for gram, count in pred_ngrams.items())
            total += sum(pred_ngrams.values())
        precisions.append((overlap + 1.0) / (total + 1.0))

    for pred_tokens, ref_tokens in zip(predictions, references):
        pred_length += len(pred_tokens)
        ref_length += len(ref_tokens)

    if pred_length == 0:
        return 0.0

    if pred_length > ref_length:
        brevity_penalty = 1.0
    else:
        brevity_penalty = math.exp(1.0 - (ref_length / max(1, pred_length)))

    bleu = brevity_penalty * math.exp(sum(math.log(p) for p in precisions) / max_n)
    return bleu * 100.0


def run_epoch(
    model: Seq2Seq,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    train_mode: bool,
    epoch_index: int,
    total_epochs: int,
) -> float:
    model.train(train_mode)
    total_loss = 0.0
    total_tokens = 0
    desc = f"{'Train' if train_mode else 'Val'} {epoch_index}/{total_epochs}"
    progress = tqdm(loader, desc=desc, leave=False)

    for batch in progress:
        src = batch.src.to(device)
        tgt = batch.tgt.to(device)
        decoder_inputs = tgt[:, :-1]
        targets = tgt[:, 1:]

        if train_mode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train_mode):
            logits = model(src, decoder_inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            if train_mode:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        non_pad = (targets != 0).sum().item()
        total_loss += loss.item() * non_pad
        total_tokens += non_pad
        avg_loss = total_loss / max(1, total_tokens)
        progress.set_postfix(loss=f"{avg_loss:.4f}")

    return total_loss / max(1, total_tokens)


@torch.no_grad()
def evaluate_predictions(
    model: Seq2Seq,
    loader: DataLoader,
    tgt_vocab: Vocab,
    criterion: nn.Module,
    device: torch.device,
    max_decode_len: int,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    exact_matches = 0
    sequence_count = 0
    token_matches = 0
    token_total = 0
    references: list[list[str]] = []
    predictions: list[list[str]] = []
    sample_rows: list[dict[str, str]] = []

    progress = tqdm(loader, desc="Test", leave=False)
    for batch in progress:
        src = batch.src.to(device)
        tgt = batch.tgt.to(device)
        decoder_inputs = tgt[:, :-1]
        targets = tgt[:, 1:]

        logits = model(src, decoder_inputs)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        non_pad = (targets != 0).sum().item()
        total_loss += loss.item() * non_pad
        total_tokens += non_pad

        generated_ids = model.greedy_decode(
            src=src,
            max_len=max_decode_len,
            bos_idx=tgt_vocab.stoi[BOS_TOKEN],
            eos_idx=tgt_vocab.stoi[EOS_TOKEN],
        )
        for row_index, pred_ids in enumerate(generated_ids):
            reference_ids = targets[row_index].tolist()
            reference_text = tgt_vocab.decode(reference_ids)
            prediction_text = tgt_vocab.decode(pred_ids)
            reference_tokens = reference_text.split()
            prediction_tokens = prediction_text.split()

            references.append(reference_tokens)
            predictions.append(prediction_tokens)
            sequence_count += 1
            if prediction_text == reference_text:
                exact_matches += 1

            for pred_token, ref_token in zip(prediction_tokens, reference_tokens):
                if pred_token == ref_token:
                    token_matches += 1
            token_total += len(reference_tokens)

            if len(sample_rows) < 10:
                source_text = " ".join(
                    token for token in [tgt_vocab.itos[0]] if False
                )
                sample_rows.append(
                    {
                        "reference": reference_text,
                        "prediction": prediction_text,
                    }
                )

        progress.set_postfix(loss=f"{(total_loss / max(1, total_tokens)):.4f}")

    bleu = simple_corpus_bleu(predictions, references)
    return {
        "loss": total_loss / max(1, total_tokens),
        "perplexity": math.exp(min(20.0, total_loss / max(1, total_tokens))),
        "exact_match": exact_matches / max(1, sequence_count),
        "token_accuracy": token_matches / max(1, token_total),
        "bleu": bleu,
        "samples": sample_rows,
    }


def save_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a simple LSTM baseline on the merged Akkadian-English corpus.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-src-vocab", type=int, default=20000)
    parser.add_argument("--max-tgt-vocab", type=int, default=20000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-src-len", type=int, default=80)
    parser.add_argument("--max-tgt-len", type=int, default=80)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-pairs", type=int, default=None)
    args = parser.parse_args()

    log_path = EXPERIMENT_DIR / "try7_run.log"
    log_path.write_text("", encoding="utf-8")
    set_seed(args.seed)
    start_time = time.time()

    device = choose_device(args.device)
    log(f"TRY 7: starting LSTM baseline training on device={device}", log_path)
    log(f"TRY 7: loading merged corpus from {SOURCE_PATH.name} / {TARGET_PATH.name}", log_path)

    all_pairs = load_pairs(SOURCE_PATH, TARGET_PATH, max_pairs=args.max_pairs)
    train_pairs, val_pairs, test_pairs = split_pairs(all_pairs, train_ratio=0.70, val_ratio=0.15, seed=args.seed)
    log(
        f"TRY 7: split sizes train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}",
        log_path,
    )

    src_vocab = Vocab.build([src for src, _ in train_pairs], max_size=args.max_src_vocab, min_freq=args.min_freq)
    tgt_vocab = Vocab.build([tgt for _, tgt in train_pairs], max_size=args.max_tgt_vocab, min_freq=args.min_freq)
    log(
        f"TRY 7: vocab sizes src={len(src_vocab.itos)} tgt={len(tgt_vocab.itos)}",
        log_path,
    )

    train_dataset = ParallelDataset(train_pairs, src_vocab, tgt_vocab, args.max_src_len, args.max_tgt_len)
    val_dataset = ParallelDataset(val_pairs, src_vocab, tgt_vocab, args.max_src_len, args.max_tgt_len)
    test_dataset = ParallelDataset(test_pairs, src_vocab, tgt_vocab, args.max_src_len, args.max_tgt_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    encoder = Encoder(len(src_vocab.itos), args.embed_dim, args.hidden_dim, args.num_layers, args.dropout)
    decoder = Decoder(len(tgt_vocab.itos), args.embed_dim, args.hidden_dim, args.num_layers, args.dropout)
    model = Seq2Seq(encoder, decoder).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    save_json(
        EXPERIMENT_DIR / "split_and_config.json",
        {
            "dataset": SOURCE_PATH.name,
            "split": {"train": len(train_pairs), "val": len(val_pairs), "test": len(test_pairs)},
            "args": vars(args),
            "device": str(device),
            "vocab": {"src": len(src_vocab.itos), "tgt": len(tgt_vocab.itos)},
        },
    )
    save_json(EXPERIMENT_DIR / "src_vocab.json", src_vocab.to_json())
    save_json(EXPERIMENT_DIR / "tgt_vocab.json", tgt_vocab.to_json())

    history_rows: list[dict[str, object]] = []
    best_val_loss = float("inf")
    best_model_path = EXPERIMENT_DIR / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device, True, epoch, args.epochs)
        val_loss = run_epoch(model, val_loader, optimizer, criterion, device, False, epoch, args.epochs)
        elapsed = time.time() - epoch_start

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "train_ppl": round(math.exp(min(20.0, train_loss)), 4),
            "val_ppl": round(math.exp(min(20.0, val_loss)), 4),
            "epoch_seconds": round(elapsed, 2),
        }
        history_rows.append(row)
        log(
            f"TRY 7: epoch={epoch}/{args.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} epoch_time={elapsed:.2f}s",
            log_path,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "src_vocab": src_vocab.to_json(),
                    "tgt_vocab": tgt_vocab.to_json(),
                },
                best_model_path,
            )
            log("TRY 7: saved new best checkpoint", log_path)

    history_path = EXPERIMENT_DIR / "train_history.csv"
    with history_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(history_rows[0].keys()))
        writer.writeheader()
        writer.writerows(history_rows)

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate_predictions(
        model=model,
        loader=test_loader,
        tgt_vocab=tgt_vocab,
        criterion=criterion,
        device=device,
        max_decode_len=args.max_tgt_len,
    )

    test_metrics_path = EXPERIMENT_DIR / "test_metrics.json"
    save_json(
        test_metrics_path,
        {
            "loss": round(float(test_metrics["loss"]), 6),
            "perplexity": round(float(test_metrics["perplexity"]), 4),
            "exact_match": round(float(test_metrics["exact_match"]), 6),
            "token_accuracy": round(float(test_metrics["token_accuracy"]), 6),
            "bleu": round(float(test_metrics["bleu"]), 4),
        },
    )

    samples_path = EXPERIMENT_DIR / "test_samples.csv"
    with samples_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["reference", "prediction"])
        writer.writeheader()
        writer.writerows(test_metrics["samples"])

    total_elapsed = time.time() - start_time
    log(
        "TRY 7 COMPLETE: "
        f"best_val_loss={best_val_loss:.4f} "
        f"test_loss={float(test_metrics['loss']):.4f} "
        f"test_bleu={float(test_metrics['bleu']):.2f} "
        f"test_exact_match={float(test_metrics['exact_match']):.4f} "
        f"runtime={total_elapsed:.2f}s",
        log_path,
    )


if __name__ == "__main__":
    main()
