from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from cache_utils import load_or_build, load_pickle
from step1_metrics_examples import (
    category_diversity_at_k,
    filter_seen,
    load_events,
    novelty_at_k,
    ranking_metrics,
    sample_eval_users,
    temporal_leave_one_out,
)
from cache_utils import split_val_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--max-eval-users", type=int, default=50_000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-user-items", type=int, default=50)
    parser.add_argument("--max-history-items", type=int, default=20)
    parser.add_argument("--neighbors", type=int, default=200)
    return parser.parse_args()


def load_core(project_dir: Path, max_eval_users: int, seed: int):
    data_dir = project_dir / "data"
    cache_dir = project_dir / "H3" / "cache"
    events = load_or_build(cache_dir / "events_preprocessed.pkl", lambda: load_events(data_dir))
    train, test = load_or_build(
        cache_dir / "temporal_leave_one_out_split.pkl",
        lambda: temporal_leave_one_out(events),
    )
    eval_users = sample_eval_users(test, max_eval_users, seed)
    _, eval_users = split_val_test(eval_users, seed)
    return {
        "train": train,
        "test": test,
        "eval_users": eval_users,
        "seen_items": load_pickle(cache_dir / "seen_items.pkl"),
        "train_sequences": load_pickle(cache_dir / "train_sequences.pkl"),
        "train_history_lengths": load_pickle(cache_dir / "train_history_lengths.pkl"),
        "fallback_items": load_pickle(cache_dir / "global_popularity.pkl")[0],
        "item_prob": load_pickle(cache_dir / "global_popularity.pkl")[1],
        "catalog_size": len(load_pickle(cache_dir / "catalog_items.pkl")),
        "item_to_category": load_pickle(cache_dir / "item_to_category.pkl"),
        "cache_dir": cache_dir,
    }


def build_itemknn_artifacts(train: pd.DataFrame, max_user_items: int, neighbors: int):
    item_weight = train.groupby("itemid")["event_weight"].sum().to_dict()
    cooc = defaultdict(Counter)

    for idx, (_, user_df) in enumerate(train.groupby("visitorid", sort=False), start=1):
        user_df = user_df.sort_values("timestamp").tail(max_user_items)
        item_scores = (
            user_df.groupby("itemid")["event_weight"]
            .sum()
            .sort_values(ascending=False)
            .to_dict()
        )
        items = list(item_scores.items())
        for i in range(len(items)):
            item_i, weight_i = int(items[i][0]), float(items[i][1])
            for j in range(i + 1, len(items)):
                item_j, weight_j = int(items[j][0]), float(items[j][1])
                score = weight_i * weight_j
                cooc[item_i][item_j] += score
                cooc[item_j][item_i] += score

        if idx % 100_000 == 0:
            print(f"processed {idx:,} user histories")

    item_neighbors = {}
    for item, counts in cooc.items():
        denom_i = np.sqrt(float(item_weight.get(item, 1.0)))
        scored = []
        for other, value in counts.items():
            denom = denom_i * np.sqrt(float(item_weight.get(other, 1.0)))
            scored.append((int(other), float(value / denom)))
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        item_neighbors[int(item)] = scored[:neighbors]
    return item_neighbors


def recommend_itemknn(user: int, core: dict, item_neighbors: dict, k: int, max_history_items: int):
    sequence = core["train_sequences"].get(user, [])
    seen = core["seen_items"].get(user, set())
    scores = defaultdict(float)

    recent_items = list(reversed(sequence[-max_history_items:]))
    for rank, item in enumerate(recent_items, start=1):
        recency_weight = 1.0 / rank
        for neighbor, sim in item_neighbors.get(int(item), []):
            if neighbor not in seen:
                scores[int(neighbor)] += recency_weight * sim

    ranked = [item for item, _ in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))]
    ranked.extend(core["fallback_items"])
    return filter_seen(ranked, seen, k)


def evaluate_itemknn(core: dict, item_neighbors: dict, k: int, max_history_items: int) -> pd.DataFrame:
    test_item = core["test"].set_index("visitorid")["itemid"].astype(int).to_dict()
    rows = []
    coverage = set()
    for idx, user in enumerate(core["eval_users"], start=1):
        recs = recommend_itemknn(user, core, item_neighbors, k, max_history_items)
        coverage.update(recs)
        true_item = int(test_item[user])
        row = {
            "model": "ItemKNN",
            "visitorid": user,
            "true_item": true_item,
            "history_length": int(core["train_history_lengths"].get(user, 0)),
        }
        row.update(ranking_metrics(recs, true_item, k))
        row[f"novelty@{k}"] = novelty_at_k(recs, core["item_prob"], core["catalog_size"], k)
        row[f"category_diversity@{k}"] = category_diversity_at_k(
            recs, core["item_to_category"], k
        )
        rows.append(row)
        if idx % 100_000 == 0:
            print(f"evaluated {idx:,} users")

    detailed = pd.DataFrame(rows)
    metric_cols = [
        f"precision@{k}",
        f"recall@{k}",
        f"ndcg@{k}",
        f"novelty@{k}",
        f"category_diversity@{k}",
    ]
    summary = detailed.groupby("model")[metric_cols].mean().reset_index()
    summary[f"catalog_coverage@{k}"] = len(coverage) / core["catalog_size"]
    short = (
        detailed[detailed["history_length"] <= 2]
        .groupby("model")[metric_cols]
        .mean()
        .reset_index()
    )
    return detailed, summary, short


def main() -> None:
    args = parse_args()
    output_dir = args.project_dir / "H3" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    core = load_core(args.project_dir, args.max_eval_users, args.seed)
    suffix = "full" if args.max_eval_users == 0 else f"sample_{len(core['eval_users'])}"
    suffix = f"{suffix}_test"
    print(f"ItemKNN run: eval_users={len(core['eval_users']):,}, suffix={suffix}")

    artifact_path = (
        core["cache_dir"]
        / f"itemknn_neighbors_usercap{args.max_user_items}_n{args.neighbors}.pkl"
    )
    item_neighbors = load_or_build(
        artifact_path,
        lambda: build_itemknn_artifacts(core["train"], args.max_user_items, args.neighbors),
    )
    print(f"items with neighbors: {len(item_neighbors):,}")

    detailed, summary, short = evaluate_itemknn(
        core, item_neighbors, args.k, args.max_history_items
    )
    detailed.to_csv(output_dir / f"step3_itemknn_detailed_{suffix}.csv", index=False)
    summary.to_csv(output_dir / f"step3_itemknn_summary_{suffix}.csv", index=False)
    short.to_csv(output_dir / f"step3_itemknn_short_history_{suffix}.csv", index=False)
    print(summary.to_string(index=False))
    print(short.to_string(index=False))


if __name__ == "__main__":
    main()
