from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from cache_utils import load_or_build, load_pickle, save_pickle, split_val_test
from step1_metrics_examples import (
    category_diversity_at_k,
    filter_seen,
    load_events,
    novelty_at_k,
    ranking_metrics,
    sample_eval_users,
    temporal_leave_one_out,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRU4Rec baseline — H3")
    parser.add_argument("--project-dir", type=Path,
                        default=Path(__file__).resolve().parents[2])
    parser.add_argument("--max-eval-users", type=int, default=50_000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    # Model hyperparameters
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-seq-len", type=int, default=20,
                        help="Truncate sequences longer than this (keeps most recent).")
    parser.add_argument("--max-train-users", type=int, default=0,
                        help="Subsample training users for speed. 0 = use all.")
    # Evaluation
    parser.add_argument("--top-n-items", type=int, default=50_000,
                        help="Score only the top-N most popular items to keep eval fast.")
    parser.add_argument("--skip-training", action="store_true",
                        help="Load a previously saved model and skip training.")
    parser.add_argument("--device", type=str, default="auto",
                        help="'auto', 'cuda', or 'cpu'.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading — reuses existing cache exactly like step1/2/3
# ---------------------------------------------------------------------------

def load_core(project_dir: Path, max_eval_users: int, seed: int) -> dict:
    data_dir = project_dir / "data"
    cache_dir = project_dir / "H3" / "cache"

    events = load_or_build(cache_dir / "events_preprocessed.pkl",
                           lambda: load_events(data_dir))
    train, test = load_or_build(
        cache_dir / "temporal_leave_one_out_split.pkl",
        lambda: temporal_leave_one_out(events),
    )
    # Same split as step1/2/3: sample → split 50/50 → use second half as test
    sampled = sample_eval_users(test, max_eval_users, seed)
    _, eval_users = split_val_test(sampled, seed)

    return {
        "train":                train,
        "test":                 test,
        "eval_users":           eval_users,
        "seen_items":           load_pickle(cache_dir / "seen_items.pkl"),
        "train_sequences":      load_pickle(cache_dir / "train_sequences.pkl"),
        "train_history_lengths":load_pickle(cache_dir / "train_history_lengths.pkl"),
        "fallback_items":       load_pickle(cache_dir / "global_popularity.pkl")[0],
        "item_prob":            load_pickle(cache_dir / "global_popularity.pkl")[1],
        "catalog_size":         len(load_pickle(cache_dir / "catalog_items.pkl")),
        "item_to_category":     load_pickle(cache_dir / "item_to_category.pkl"),
        "cache_dir":            cache_dir,
    }


# ---------------------------------------------------------------------------
# Item index — maps raw item IDs to contiguous integers [0, N)
# ---------------------------------------------------------------------------

def build_item_index(train_sequences: dict, top_n: int) -> tuple[dict, dict, list]:
    """
    Build item→idx and idx→item mappings.
    We restrict to the top_n most popular items to keep the output layer
    manageable. Items outside this set fall back to global popularity at
    recommendation time.
    """
    counts: dict[int, int] = defaultdict(int)
    for seq in train_sequences.values():
        for item in seq:
            counts[int(item)] += 1

    sorted_items = sorted(counts, key=lambda x: -counts[x])
    top_items = sorted_items[:top_n]

    item_to_idx = {item: idx + 1 for idx, item in enumerate(top_items)}  # 0 = padding
    idx_to_item = {idx + 1: item for idx, item in enumerate(top_items)}
    return item_to_idx, idx_to_item, top_items


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SessionDataset(Dataset):
    """
    Yields (input_seq, target) pairs using the standard GRU4Rec formulation:
    for a sequence [i1, i2, i3, i4], produces:
        input=[i1, i2, i3]  target=i4
        input=[i1, i2]      target=i3
        input=[i1]          target=i2
    Only items in item_to_idx are used; sequences shorter than 2 are skipped.
    """

    def __init__(self, train_sequences: dict, item_to_idx: dict,
                 max_seq_len: int, max_users: int = 0, seed: int = 42):
        self.samples: list[tuple[list[int], int]] = []
        rng = np.random.default_rng(seed)
        users = list(train_sequences.keys())
        if max_users and len(users) > max_users:
            users = rng.choice(users, size=max_users, replace=False).tolist()

        for user in users:
            seq = [item_to_idx[int(i)] for i in train_sequences[user]
                   if int(i) in item_to_idx]
            if len(seq) < 2:
                continue
            seq = seq[-max_seq_len:]          # keep most recent
            for end in range(1, len(seq)):    # sliding window
                inp = seq[:end]
                tgt = seq[end]
                self.samples.append((inp, tgt))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[list[int], int]:
        return self.samples[idx]


def collate_fn(batch: list[tuple[list[int], int]]):
    """Pad sequences to the same length within a batch (left-pad with 0)."""
    seqs, targets = zip(*batch)
    max_len = max(len(s) for s in seqs)
    padded = torch.zeros(len(seqs), max_len, dtype=torch.long)
    lengths = torch.zeros(len(seqs), dtype=torch.long)
    for i, seq in enumerate(seqs):
        padded[i, -len(seq):] = torch.tensor(seq, dtype=torch.long)
        lengths[i] = len(seq)
    targets = torch.tensor(targets, dtype=torch.long)
    return padded, lengths, targets


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GRU4Rec(nn.Module):
    def __init__(self, num_items: int, embedding_dim: int, hidden_dim: int,
                 num_layers: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, num_items)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        x:       (batch, seq_len)  — padded item indices
        lengths: (batch,)          — actual sequence lengths
        Returns: (batch, num_items) logits
        """
        emb = self.dropout(self.embedding(x))          # (B, L, E)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)                   # hidden: (layers, B, H)
        last_hidden = self.dropout(hidden[-1])          # (B, H)
        return self.output(last_hidden)                 # (B, num_items)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model: GRU4Rec, dataset: SessionDataset, epochs: int,
                batch_size: int, lr: float, device: torch.device) -> None:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_fn, num_workers=0, pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        total_loss = 0.0
        for seqs, lengths, targets in loader:
            seqs = seqs.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(seqs, lengths)          # (B, num_items)
            loss = criterion(logits, targets - 1)  # targets are 1-indexed
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        elapsed = time.time() - t0
        print(f"  epoch {epoch}/{epochs}  loss={avg_loss:.4f}  time={elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_user_scores(model: GRU4Rec, sequence: list[int],
                    item_to_idx: dict, max_seq_len: int,
                    device: torch.device) -> np.ndarray | None:
    """
    Returns a score array of shape (num_items,) for the given raw-id sequence.
    Returns None if the sequence has no known items.
    """
    idx_seq = [item_to_idx[int(i)] for i in sequence if int(i) in item_to_idx]
    if not idx_seq:
        return None
    idx_seq = idx_seq[-max_seq_len:]
    t = torch.tensor([idx_seq], dtype=torch.long, device=device)
    lengths = torch.tensor([len(idx_seq)], dtype=torch.long, device=device)
    model.eval()
    logits = model(t, lengths)                    # (1, num_items)
    return logits[0].cpu().numpy()


def recommend_gru4rec(user: int, core: dict, model: GRU4Rec,
                      item_to_idx: dict, idx_to_item: dict,
                      top_items: list[int], max_seq_len: int,
                      device: torch.device, k: int) -> list[int]:
    sequence = core["train_sequences"].get(user, [])
    seen = core["seen_items"].get(user, set())
    scores = get_user_scores(model, sequence, item_to_idx, max_seq_len, device)

    if scores is not None:
        # scores[i] corresponds to idx_to_item[i+1]
        ranked_idxs = np.argsort(-scores)         # descending
        candidates = [idx_to_item[idx + 1] for idx in ranked_idxs
                      if (idx + 1) in idx_to_item]
    else:
        candidates = []

    # Fallback: append global popularity for items not in the model's vocab
    candidates.extend(core["fallback_items"])
    return filter_seen(candidates, seen, k)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(core: dict, model: GRU4Rec, item_to_idx: dict,
             idx_to_item: dict, top_items: list[int],
             max_seq_len: int, device: torch.device, k: int) -> pd.DataFrame:
    test_item = core["test"].set_index("visitorid")["itemid"].astype(int).to_dict()
    rows = []
    coverage: set[int] = set()

    for idx, user in enumerate(core["eval_users"], start=1):
        recs = recommend_gru4rec(user, core, model, item_to_idx, idx_to_item,
                                 top_items, max_seq_len, device, k)
        coverage.update(recs)
        true_item = int(test_item[user])
        row = {
            "model":          "GRU4Rec",
            "visitorid":      user,
            "true_item":      true_item,
            "history_length": int(core["train_history_lengths"].get(user, 0)),
        }
        row.update(ranking_metrics(recs, true_item, k))
        row[f"novelty@{k}"]           = novelty_at_k(recs, core["item_prob"],
                                                       core["catalog_size"], k)
        row[f"category_diversity@{k}"] = category_diversity_at_k(
            recs, core["item_to_category"], k)
        rows.append(row)

        if idx % 5_000 == 0:
            print(f"  evaluated {idx:,} / {len(core['eval_users']):,} users")

    detailed = pd.DataFrame(rows)
    metric_cols = [f"precision@{k}", f"recall@{k}", f"ndcg@{k}",
                   f"novelty@{k}", f"category_diversity@{k}"]
    summary = detailed.groupby("model")[metric_cols].mean().reset_index()
    summary[f"catalog_coverage@{k}"] = len(coverage) / core["catalog_size"]
    short = (detailed[detailed["history_length"] <= 2]
             .groupby("model")[metric_cols].mean().reset_index())
    return detailed, summary, short


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    output_dir = args.project_dir / "H3" / "outputs"
    cache_dir = args.project_dir / "H3" / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading core data…")
    core = load_core(args.project_dir, args.max_eval_users, args.seed)
    suffix = f"sample_{len(core['eval_users'])}_test"
    print(f"eval_users={len(core['eval_users']):,}  suffix={suffix}")

    # ── Item index ─────────────────────────────────────────────────────────
    idx_path = cache_dir / f"gru4rec_item_index_top{args.top_n_items}.pkl"
    item_to_idx, idx_to_item, top_items = load_or_build(
        idx_path,
        lambda: build_item_index(core["train_sequences"], args.top_n_items),
    )
    num_items = len(item_to_idx)
    print(f"Vocabulary size: {num_items:,} items (top-{args.top_n_items})")

    # ── Model ──────────────────────────────────────────────────────────────
    model = GRU4Rec(
        num_items=num_items,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    model_path = cache_dir / (
        f"gru4rec_model_emb{args.embedding_dim}_hid{args.hidden_dim}"
        f"_ep{args.epochs}_top{args.top_n_items}.pt"
    )

    if args.skip_training and model_path.exists():
        print(f"Loading saved model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        # ── Dataset ────────────────────────────────────────────────────────
        print("Building training dataset…")
        dataset = SessionDataset(
            core["train_sequences"], item_to_idx,
            max_seq_len=args.max_seq_len,
            max_users=args.max_train_users,
            seed=args.seed,
        )
        print(f"Training samples: {len(dataset):,}")

        # ── Train ──────────────────────────────────────────────────────────
        print(f"Training GRU4Rec for {args.epochs} epochs…")
        train_model(model, dataset, args.epochs, args.batch_size, args.lr, device)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    # ── Evaluate ───────────────────────────────────────────────────────────
    print("Evaluating on test users…")
    detailed, summary, short = evaluate(
        core, model, item_to_idx, idx_to_item, top_items,
        args.max_seq_len, device, args.k,
    )

    detailed.to_csv(output_dir / f"step8_gru4rec_detailed_{suffix}.csv", index=False)
    summary.to_csv(output_dir / f"step8_gru4rec_summary_{suffix}.csv", index=False)
    short.to_csv(output_dir / f"step8_gru4rec_short_history_{suffix}.csv", index=False)

    print("\n── Global results ──")
    print(summary.to_string(index=False))
    print("\n── Short-history results (≤2 interactions) ──")
    print(short.to_string(index=False))
    print(f"\nOutputs written to {output_dir}")


if __name__ == "__main__":
    main()
