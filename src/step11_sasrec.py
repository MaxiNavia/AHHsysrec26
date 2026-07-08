"""
step11_sasrec.py — SASRec baseline for H3

Implements SASRec (Kang & McAuley, 2018): a self-attentive sequential
recommender that uses multi-head self-attention over item sequences.

Reuses the existing cache (train_sequences, seen_items, etc.) and the
same val/test split used by step1, step2, step3 and step8.

Usage:
    # Full run (train + evaluate):
    python3 src/step11_sasrec.py --max-eval-users 50000

    # Quick smoke test (2 epochs, 50k training users):
    python3 src/step11_sasrec.py --max-eval-users 50000 --epochs 2 --max-train-users 50000

    # Re-evaluate without retraining (if .pt exists in cache):
    python3 src/step11_sasrec.py --max-eval-users 50000 --skip-training
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from cache_utils import load_or_build, load_pickle, split_val_test
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
    parser = argparse.ArgumentParser(description="SASRec baseline — H3")
    parser.add_argument("--project-dir", type=Path,
                        default=Path(__file__).resolve().parents[2])
    parser.add_argument("--max-eval-users", type=int, default=50_000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    # Model hyperparameters
    parser.add_argument("--embedding-dim", type=int, default=128,
                        help="Item embedding and hidden dimension.")
    parser.add_argument("--num-heads", type=int, default=2,
                        help="Number of attention heads.")
    parser.add_argument("--num-blocks", type=int, default=2,
                        help="Number of Transformer blocks.")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-seq-len", type=int, default=20,
                        help="Maximum sequence length (truncates oldest).")
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-train-users", type=int, default=0,
                        help="Subsample training users. 0 = use all.")
    # Evaluation
    parser.add_argument("--top-n-items", type=int, default=50_000,
                        help="Restrict scoring to top-N popular items.")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading — identical pattern to step8_gru4rec.py
# ---------------------------------------------------------------------------

def load_core(project_dir: Path, max_eval_users: int, seed: int) -> dict:
    data_dir  = project_dir / "data"
    cache_dir = project_dir / "H3" / "cache"

    events = load_or_build(cache_dir / "events_preprocessed.pkl",
                           lambda: load_events(data_dir))
    train, test = load_or_build(
        cache_dir / "temporal_leave_one_out_split.pkl",
        lambda: temporal_leave_one_out(events),
    )
    sampled = sample_eval_users(test, max_eval_users, seed)
    _, eval_users = split_val_test(sampled, seed)

    return {
        "train":                 train,
        "test":                  test,
        "eval_users":            eval_users,
        "seen_items":            load_pickle(cache_dir / "seen_items.pkl"),
        "train_sequences":       load_pickle(cache_dir / "train_sequences.pkl"),
        "train_history_lengths": load_pickle(cache_dir / "train_history_lengths.pkl"),
        "fallback_items":        load_pickle(cache_dir / "global_popularity.pkl")[0],
        "item_prob":             load_pickle(cache_dir / "global_popularity.pkl")[1],
        "catalog_size":          len(load_pickle(cache_dir / "catalog_items.pkl")),
        "item_to_category":      load_pickle(cache_dir / "item_to_category.pkl"),
        "cache_dir":             cache_dir,
    }


# ---------------------------------------------------------------------------
# Item index — same logic as GRU4Rec, reuses cache if available
# ---------------------------------------------------------------------------

def build_item_index(train_sequences: dict,
                     top_n: int) -> tuple[dict, dict, list]:
    from collections import defaultdict
    counts: dict[int, int] = defaultdict(int)
    for seq in train_sequences.values():
        for item in seq:
            counts[int(item)] += 1
    top_items   = sorted(counts, key=lambda x: -counts[x])[:top_n]
    item_to_idx = {item: idx + 1 for idx, item in enumerate(top_items)}
    idx_to_item = {idx + 1: item for idx, item in enumerate(top_items)}
    return item_to_idx, idx_to_item, top_items


# ---------------------------------------------------------------------------
# Dataset — sliding window, same as GRU4Rec
# ---------------------------------------------------------------------------

class SeqDataset(Dataset):
    def __init__(self, train_sequences: dict, item_to_idx: dict,
                 max_seq_len: int, max_users: int = 0, seed: int = 42):
        self.max_seq_len = max_seq_len
        self.samples: list[tuple[list[int], int]] = []
        rng   = np.random.default_rng(seed)
        users = list(train_sequences.keys())
        if max_users and len(users) > max_users:
            users = rng.choice(users, size=max_users, replace=False).tolist()

        for user in users:
            seq = [item_to_idx[int(i)] for i in train_sequences[user]
                   if int(i) in item_to_idx]
            if len(seq) < 2:
                continue
            seq = seq[-max_seq_len:]
            for end in range(1, len(seq)):
                inp = seq[:end]
                tgt = seq[end]
                self.samples.append((inp, tgt))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def collate_fn(batch):
    """Left-pad sequences to the same length within a batch."""
    seqs, targets = zip(*batch)
    max_len = max(len(s) for s in seqs)
    padded  = torch.zeros(len(seqs), max_len, dtype=torch.long)
    for i, seq in enumerate(seqs):
        padded[i, -len(seq):] = torch.tensor(seq, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return padded, targets


# ---------------------------------------------------------------------------
# SASRec model
# ---------------------------------------------------------------------------

class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SASRecBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn   = nn.MultiheadAttention(hidden_dim, num_heads,
                                             dropout=dropout, batch_first=True)
        self.ffn    = PointWiseFeedForward(hidden_dim, dropout)
        self.norm1  = nn.LayerNorm(hidden_dim)
        self.norm2  = nn.LayerNorm(hidden_dim)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Causal self-attention (pre-norm)
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask,
                                 need_weights=False)
        x = residual + self.drop(attn_out)
        # Feed-forward (pre-norm)
        residual = x
        x = residual + self.ffn(self.norm2(x))
        return x


class SASRec(nn.Module):
    def __init__(self, num_items: int, hidden_dim: int,
                 num_heads: int, num_blocks: int,
                 max_seq_len: int, dropout: float):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.max_seq_len = max_seq_len

        self.item_emb = nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)
        self.pos_emb  = nn.Embedding(max_seq_len + 1, hidden_dim)
        self.emb_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            SASRecBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

        # Output projection: hidden → num_items logits
        self.output = nn.Linear(hidden_dim, num_items)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight,  std=0.02)
        nn.init.zeros_(self.item_emb.weight[0])   # padding idx = 0

    def _causal_mask(self, seq_len: int,
                     device: torch.device) -> torch.Tensor:
        """Upper-triangular mask to prevent attending to future positions."""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len) — padded item indices (0 = padding)
        Returns: (batch, num_items) logits based on the last non-padding position
        """
        B, L = x.shape

        # Positions: 1-indexed, 0 for padding
        positions = torch.arange(1, L + 1, device=x.device).unsqueeze(0)
        pad_mask  = (x == 0)
        positions = positions.masked_fill(pad_mask, 0)

        h = self.emb_drop(self.item_emb(x) + self.pos_emb(positions))

        causal = self._causal_mask(L, x.device)
        for block in self.blocks:
            h = block(h, attn_mask=causal)
        h = self.norm(h)                        # (B, L, D)

        # Use the last non-padding position for each sequence
        lengths    = (x != 0).sum(dim=1) - 1   # 0-indexed last pos
        lengths    = lengths.clamp(min=0)
        idx        = lengths.view(B, 1, 1).expand(B, 1, self.hidden_dim)
        last_h     = h.gather(1, idx).squeeze(1)  # (B, D)

        return self.output(last_h)              # (B, num_items)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model: SASRec, dataset: SeqDataset,
                epochs: int, batch_size: int, lr: float,
                device: torch.device) -> None:
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                           collate_fn=collate_fn, num_workers=0,
                           pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(1, epochs + 1):
        t0         = time.time()
        total_loss = 0.0

        for seqs, targets in loader:
            seqs    = seqs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(seqs)                   # (B, num_items)
            loss   = criterion(logits, targets - 1) # targets are 1-indexed
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        elapsed  = time.time() - t0
        print(f"  epoch {epoch}/{epochs}  loss={avg_loss:.4f}  time={elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_scores(model: SASRec, sequence: list[int],
               item_to_idx: dict, max_seq_len: int,
               device: torch.device) -> np.ndarray | None:
    idx_seq = [item_to_idx[int(i)] for i in sequence
               if int(i) in item_to_idx]
    if not idx_seq:
        return None
    idx_seq = idx_seq[-max_seq_len:]
    # Left-pad to max_seq_len
    padded  = [0] * (max_seq_len - len(idx_seq)) + idx_seq
    t       = torch.tensor([padded], dtype=torch.long, device=device)
    model.eval()
    logits  = model(t)          # (1, num_items)
    return logits[0].cpu().numpy()


def recommend_sasrec(user: int, core: dict, model: SASRec,
                     item_to_idx: dict, idx_to_item: dict,
                     max_seq_len: int, device: torch.device,
                     k: int) -> list[int]:
    sequence = core["train_sequences"].get(user, [])
    seen     = core["seen_items"].get(user, set())
    scores   = get_scores(model, sequence, item_to_idx, max_seq_len, device)

    if scores is not None:
        ranked_idxs = np.argsort(-scores)
        candidates  = [idx_to_item[idx + 1] for idx in ranked_idxs
                       if (idx + 1) in idx_to_item]
    else:
        candidates = []

    candidates.extend(core["fallback_items"])
    return filter_seen(candidates, seen, k)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(core: dict, model: SASRec,
             item_to_idx: dict, idx_to_item: dict,
             max_seq_len: int, device: torch.device,
             k: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test_item = core["test"].set_index("visitorid")["itemid"].astype(int).to_dict()
    rows: list[dict] = []
    coverage: set[int] = set()

    for idx, user in enumerate(core["eval_users"], start=1):
        recs      = recommend_sasrec(user, core, model, item_to_idx,
                                     idx_to_item, max_seq_len, device, k)
        coverage.update(recs)
        true_item = int(test_item[user])
        row = {
            "model":           "SASRec",
            "visitorid":       user,
            "true_item":       true_item,
            "history_length":  int(core["train_history_lengths"].get(user, 0)),
        }
        row.update(ranking_metrics(recs, true_item, k))
        row[f"novelty@{k}"]            = novelty_at_k(
            recs, core["item_prob"], core["catalog_size"], k)
        row[f"category_diversity@{k}"] = category_diversity_at_k(
            recs, core["item_to_category"], k)
        rows.append(row)

        if idx % 5_000 == 0:
            print(f"  evaluated {idx:,} / {len(core['eval_users']):,} users")

    detailed    = pd.DataFrame(rows)
    metric_cols = [f"precision@{k}", f"recall@{k}", f"ndcg@{k}",
                   f"novelty@{k}", f"category_diversity@{k}"]
    summary     = detailed.groupby("model")[metric_cols].mean().reset_index()
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

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    output_dir = args.project_dir / "H3" / "outputs"
    cache_dir  = args.project_dir / "H3" / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    print("Loading core data...")
    core   = load_core(args.project_dir, args.max_eval_users, args.seed)
    suffix = f"sample_{len(core['eval_users'])}_test"
    print(f"eval_users={len(core['eval_users']):,}  suffix={suffix}")

    # ── Item index (shared with GRU4Rec if same top_n) ────────────────────
    idx_path = cache_dir / f"gru4rec_item_index_top{args.top_n_items}.pkl"
    item_to_idx, idx_to_item, top_items = load_or_build(
        idx_path,
        lambda: build_item_index(core["train_sequences"], args.top_n_items),
    )
    num_items = len(item_to_idx)
    print(f"Vocabulary size: {num_items:,} items (top-{args.top_n_items})")

    # ── Model ──────────────────────────────────────────────────────────────
    model = SASRec(
        num_items   = num_items,
        hidden_dim  = args.embedding_dim,
        num_heads   = args.num_heads,
        num_blocks  = args.num_blocks,
        max_seq_len = args.max_seq_len,
        dropout     = args.dropout,
    ).to(device)

    model_path = cache_dir / (
        f"sasrec_model_emb{args.embedding_dim}_h{args.num_heads}"
        f"_b{args.num_blocks}_ep{args.epochs}_top{args.top_n_items}.pt"
    )

    if args.skip_training and model_path.exists():
        print(f"Loading saved model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        # ── Dataset ────────────────────────────────────────────────────────
        print("Building training dataset...")
        dataset = SeqDataset(
            core["train_sequences"], item_to_idx,
            max_seq_len  = args.max_seq_len,
            max_users    = args.max_train_users,
            seed         = args.seed,
        )
        print(f"Training samples: {len(dataset):,}")

        # ── Train ──────────────────────────────────────────────────────────
        print(f"Training SASRec for {args.epochs} epochs...")
        train_model(model, dataset, args.epochs,
                    args.batch_size, args.lr, device)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    # ── Evaluate ───────────────────────────────────────────────────────────
    print("Evaluating on test users...")
    detailed, summary, short = evaluate(
        core, model, item_to_idx, idx_to_item,
        args.max_seq_len, device, args.k,
    )

    detailed.to_csv(output_dir / f"step11_sasrec_detailed_{suffix}.csv",
                    index=False)
    summary.to_csv(output_dir  / f"step11_sasrec_summary_{suffix}.csv",
                   index=False)
    short.to_csv(output_dir    / f"step11_sasrec_short_history_{suffix}.csv",
                 index=False)

    print("\n── Global results ──")
    print(summary.to_string(index=False))
    print("\n── Short-history results (≤2 interactions) ──")
    print(short.to_string(index=False))
    print(f"\nOutputs written to {output_dir}")


if __name__ == "__main__":
    main()
