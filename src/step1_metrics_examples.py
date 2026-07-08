from __future__ import annotations

import argparse
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable
from cache_utils import split_val_test

import numpy as np
import pandas as pd


EVENT_WEIGHTS = {
    "view": 1.0,
    "addtocart": 3.0,
    "transaction": 5.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 1 for H3: add novelty/diversity/coverage metrics and qualitative examples."
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project root containing data/ and H3/.",
    )
    parser.add_argument(
        "--max-eval-users",
        type=int,
        default=50_000,
        help="Number of evaluable users to sample. Use 0 for full evaluation.",
    )
    parser.add_argument("--k", type=int, default=10, help="Top-K cutoff.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def load_events(data_dir: Path) -> pd.DataFrame:
    events = pd.read_csv(data_dir / "events.csv")
    events["event_weight"] = events["event"].map(EVENT_WEIGHTS).fillna(1.0)
    events = events.sort_values(["visitorid", "timestamp", "itemid"]).reset_index(drop=True)
    events["event_position"] = events.groupby("visitorid").cumcount() + 1
    events["sequence_length"] = events.groupby("visitorid")["event"].transform("size")
    return events


def temporal_leave_one_out(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    evaluable = events[events["sequence_length"] >= 2].copy()
    last_pos = evaluable.groupby("visitorid")["event_position"].transform("max")
    train = evaluable[evaluable["event_position"] < last_pos].copy()
    test = evaluable[evaluable["event_position"] == last_pos].copy()
    return train, test


def sample_eval_users(test: pd.DataFrame, max_eval_users: int, seed: int) -> list[int]:
    eval_users = test["visitorid"].drop_duplicates().to_numpy()
    if max_eval_users and len(eval_users) > max_eval_users:
        rng = np.random.default_rng(seed)
        eval_users = rng.choice(eval_users, size=max_eval_users, replace=False)
    return [int(user) for user in eval_users]


def load_latest_item_categories(data_dir: Path, chunksize: int = 1_000_000) -> pd.DataFrame:
    parts = []
    for path in [data_dir / "item_properties_part1.csv", data_dir / "item_properties_part2.csv"]:
        for chunk in pd.read_csv(path, chunksize=chunksize):
            chunk = chunk.loc[
                chunk["property"].eq("categoryid"),
                ["timestamp", "itemid", "value"],
            ].copy()
            chunk["categoryid"] = pd.to_numeric(chunk["value"], errors="coerce")
            chunk = chunk.dropna(subset=["categoryid"])
            chunk["categoryid"] = chunk["categoryid"].astype(int)
            parts.append(chunk[["timestamp", "itemid", "categoryid"]])

    if not parts:
        return pd.DataFrame(columns=["itemid", "categoryid"])

    categories = pd.concat(parts, ignore_index=True)
    categories = categories.sort_values(["itemid", "timestamp"])
    latest = categories.groupby("itemid", as_index=False).tail(1)
    return latest[["itemid", "categoryid"]].reset_index(drop=True)


def build_seen_items(train: pd.DataFrame) -> dict[int, set[int]]:
    return train.groupby("visitorid")["itemid"].apply(lambda values: set(map(int, values))).to_dict()


def build_train_sequences(train: pd.DataFrame) -> dict[int, list[int]]:
    return train.groupby("visitorid")["itemid"].apply(lambda values: list(map(int, values))).to_dict()


def filter_seen(candidates: list[int], seen: set[int], k: int) -> list[int]:
    output = []
    added = set()
    for item in candidates:
        item = int(item)
        if item in seen or item in added:
            continue
        output.append(item)
        added.add(item)
        if len(output) >= k:
            break
    return output


def ranking_metrics(recommendations: list[int], true_item: int, k: int) -> dict[str, float]:
    topk = recommendations[:k]
    hit = int(true_item in topk)
    if hit:
        rank = topk.index(true_item) + 1
        ndcg = 1.0 / math.log2(rank + 1)
    else:
        ndcg = 0.0
    return {
        f"precision@{k}": hit / k,
        f"recall@{k}": float(hit),
        f"ndcg@{k}": ndcg,
    }


def novelty_at_k(recommendations: list[int], item_prob: dict[int, float], catalog_size: int, k: int) -> float:
    topk = recommendations[:k]
    if not topk:
        return 0.0
    floor_prob = 1.0 / max(catalog_size, 1)
    return float(np.mean([-math.log2(item_prob.get(int(item), floor_prob)) for item in topk]))


def category_diversity_at_k(recommendations: list[int], item_to_category: dict[int, int], k: int) -> float:
    topk = [int(item) for item in recommendations[:k] if int(item) in item_to_category]
    if len(topk) < 2:
        return 0.0
    total_pairs = 0
    diverse_pairs = 0
    for i in range(len(topk)):
        for j in range(i + 1, len(topk)):
            total_pairs += 1
            diverse_pairs += int(item_to_category[topk[i]] != item_to_category[topk[j]])
    return diverse_pairs / total_pairs if total_pairs else 0.0


def build_global_popularity(train: pd.DataFrame) -> tuple[list[int], dict[int, float], dict[int, float]]:
    popularity = (
        train.groupby("itemid")["event_weight"]
        .sum()
        .sort_values(ascending=False)
    )
    ranked_items = [int(item) for item in popularity.index]
    total_weight = float(popularity.sum())
    item_prob = {int(item): float(weight / total_weight) for item, weight in popularity.items()}
    item_score = {int(item): float(weight) for item, weight in popularity.items()}
    return ranked_items, item_prob, item_score


def build_transition_recommender(
    train: pd.DataFrame,
    train_sequences: dict[int, list[int]],
    seen_items: dict[int, set[int]],
    fallback_items: list[int],
) -> Callable[[int, int], list[int]]:
    transition_counts: dict[int, Counter[int]] = defaultdict(Counter)
    for _, user_df in train.groupby("visitorid", sort=False):
        items = [int(item) for item in user_df["itemid"].tolist()]
        weights = [float(weight) for weight in user_df["event_weight"].tolist()]
        for previous_item, next_item, next_weight in zip(items[:-1], items[1:], weights[1:]):
            if previous_item != next_item:
                transition_counts[previous_item][next_item] += next_weight

    transition_rankings = {
        item: [next_item for next_item, _ in counts.most_common(200)]
        for item, counts in transition_counts.items()
    }

    def recommend(user: int, k: int) -> list[int]:
        seen = seen_items.get(user, set())
        sequence = train_sequences.get(user, [])
        candidates = []
        if sequence:
            candidates.extend(transition_rankings.get(sequence[-1], []))
        candidates.extend(fallback_items)
        return filter_seen(candidates, seen, k)

    return recommend


def build_category_recommender(
    train: pd.DataFrame,
    item_to_category: dict[int, int],
    train_sequences: dict[int, list[int]],
    seen_items: dict[int, set[int]],
    fallback_items: list[int],
) -> Callable[[int, int], list[int]]:
    train_with_category = train.copy()
    train_with_category["categoryid"] = train_with_category["itemid"].map(item_to_category)
    train_with_category = train_with_category.dropna(subset=["categoryid"]).copy()
    train_with_category["categoryid"] = train_with_category["categoryid"].astype(int)

    category_popularity = (
        train_with_category.groupby(["categoryid", "itemid"])["event_weight"]
        .sum()
        .reset_index(name="score")
        .sort_values(["categoryid", "score"], ascending=[True, False])
    )
    category_to_items = (
        category_popularity.groupby("categoryid")["itemid"]
        .apply(lambda values: [int(item) for item in values.head(300)])
        .to_dict()
    )

    last_user_category = {}
    for user, sequence in train_sequences.items():
        for item in reversed(sequence):
            category = item_to_category.get(item)
            if category is not None:
                last_user_category[user] = int(category)
                break

    def recommend(user: int, k: int) -> list[int]:
        seen = seen_items.get(user, set())
        candidates = []
        category = last_user_category.get(user)
        if category is not None:
            candidates.extend(category_to_items.get(category, []))
        candidates.extend(fallback_items)
        return filter_seen(candidates, seen, k)

    return recommend


def rank_to_scores(items: list[int], weight: float, max_rank: int = 200) -> dict[int, float]:
    scores = {}
    for rank, item in enumerate(items[:max_rank], start=1):
        scores[int(item)] = scores.get(int(item), 0.0) + weight * (1.0 / rank)
    return scores


def build_fixed_hybrid_recommender(
    sequential_recommender: Callable[[int, int], list[int]],
    category_recommender: Callable[[int, int], list[int]],
    seen_items: dict[int, set[int]],
    fallback_items: list[int],
    seq_weight: float = 0.70,
    cat_weight: float = 0.30,
    pop_weight: float = 0.05,
) -> Callable[[int, int], list[int]]:
    def recommend(user: int, k: int) -> list[int]:
        scores = defaultdict(float)
        for item, score in rank_to_scores(sequential_recommender(user, 100), seq_weight).items():
            scores[item] += score
        for item, score in rank_to_scores(category_recommender(user, 100), cat_weight).items():
            scores[item] += score
        for item, score in rank_to_scores(fallback_items[:100], pop_weight).items():
            scores[item] += score

        ranked = [item for item, _ in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))]
        return filter_seen(ranked + fallback_items, seen_items.get(user, set()), k)

    return recommend


def build_random_recommender(
    catalog_items: list[int],
    seen_items: dict[int, set[int]],
    seed: int,
) -> Callable[[int, int], list[int]]:
    rng = random.Random(seed)

    def recommend(user: int, k: int) -> list[int]:
        seen = seen_items.get(user, set())
        output = []
        attempts = 0
        while len(output) < k and attempts < k * 200:
            attempts += 1
            item = int(rng.choice(catalog_items))
            if item not in seen and item not in output:
                output.append(item)
        return output

    return recommend


def evaluate_models(
    models: dict[str, Callable[[int, int], list[int]]],
    eval_users: list[int],
    test: pd.DataFrame,
    train_history_lengths: dict[int, int],
    item_prob: dict[int, float],
    item_to_category: dict[int, int],
    catalog_size: int,
    k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, set[int]]]:
    test_item = test.set_index("visitorid")["itemid"].astype(int).to_dict()
    rows = []
    coverage_items = {model_name: set() for model_name in models}

    for model_name, recommender in models.items():
        for user in eval_users:
            recs = recommender(user, k)
            coverage_items[model_name].update(recs)
            row = {
                "model": model_name,
                "visitorid": user,
                "true_item": int(test_item[user]),
                "history_length": int(train_history_lengths.get(user, 0)),
            }
            row.update(ranking_metrics(recs, int(test_item[user]), k))
            row[f"novelty@{k}"] = novelty_at_k(recs, item_prob, catalog_size, k)
            row[f"category_diversity@{k}"] = category_diversity_at_k(recs, item_to_category, k)
            rows.append(row)

    detailed = pd.DataFrame(rows)
    metric_cols = [
        f"precision@{k}",
        f"recall@{k}",
        f"ndcg@{k}",
        f"novelty@{k}",
        f"category_diversity@{k}",
    ]
    summary = (
        detailed.groupby("model")[metric_cols]
        .mean()
        .reset_index()
        .sort_values(f"recall@{k}", ascending=False)
    )
    summary[f"catalog_coverage@{k}"] = summary["model"].map(
        lambda model: len(coverage_items[model]) / catalog_size
    )
    return detailed, summary, coverage_items


def build_examples(
    eval_users: list[int],
    train: pd.DataFrame,
    test: pd.DataFrame,
    models: dict[str, Callable[[int, int], list[int]]],
    item_to_category: dict[int, int],
    k: int,
) -> pd.DataFrame:
    train_lengths = train.groupby("visitorid").size().to_dict()
    short_users = [user for user in eval_users if train_lengths.get(user, 0) <= 2]
    longer_users = [user for user in eval_users if train_lengths.get(user, 0) >= 5]
    selected = short_users[:2] + longer_users[:2]
    if len(selected) < 4:
        selected = eval_users[:4]

    test_item = test.set_index("visitorid")["itemid"].astype(int).to_dict()
    train_by_user = {
        int(user): user_df.sort_values("timestamp")[["event", "itemid"]].tail(6).values.tolist()
        for user, user_df in train[train["visitorid"].isin(selected)].groupby("visitorid")
    }

    rows = []
    for user in selected:
        true_item = int(test_item[user])
        history_pairs = train_by_user.get(user, [])
        history = " | ".join(f"{event}:{int(item)}" for event, item in history_pairs)
        history_categories = [
            item_to_category.get(int(item))
            for _, item in history_pairs
            if item_to_category.get(int(item)) is not None
        ]
        for model_name, recommender in models.items():
            recs = recommender(user, k)
            rec_categories = [item_to_category.get(int(item)) for item in recs]
            rows.append(
                {
                    "visitorid": user,
                    "history_length": train_lengths.get(user, 0),
                    "recent_history": history,
                    "recent_categories": history_categories,
                    "model": model_name,
                    "recommendations": recs,
                    "recommendation_categories": rec_categories,
                    "true_item": true_item,
                    "true_category": item_to_category.get(true_item),
                    "hit@k": int(true_item in recs),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir
    data_dir = project_dir / "data"
    output_dir = project_dir / "H3" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading events...")
    events = load_events(data_dir)
    train, test = temporal_leave_one_out(events)
    eval_users = sample_eval_users(test, args.max_eval_users, args.seed)
    _, eval_users = split_val_test(eval_users, args.seed)

    print(f"Train events: {len(train):,}")
    print(f"Test users: {test['visitorid'].nunique():,}")
    print(f"Evaluated users: {len(eval_users):,}")

    print("Loading item categories...")
    item_categories = load_latest_item_categories(data_dir)
    item_to_category = dict(
        zip(item_categories["itemid"].astype(int), item_categories["categoryid"].astype(int))
    )

    seen_items = build_seen_items(train)
    train_sequences = build_train_sequences(train)
    train_history_lengths = train.groupby("visitorid").size().astype(int).to_dict()
    fallback_items, item_prob, _ = build_global_popularity(train)
    catalog_items = sorted(train["itemid"].astype(int).unique().tolist())
    catalog_size = len(catalog_items)

    print("Building recommenders...")
    random_recommender = build_random_recommender(catalog_items, seen_items, args.seed)
    most_popular_recommender = lambda user, k: filter_seen(
        fallback_items, seen_items.get(user, set()), k
    )
    sequential_recommender = build_transition_recommender(
        train, train_sequences, seen_items, fallback_items
    )
    category_recommender = build_category_recommender(
        train, item_to_category, train_sequences, seen_items, fallback_items
    )
    fixed_hybrid_recommender = build_fixed_hybrid_recommender(
        sequential_recommender, category_recommender, seen_items, fallback_items
    )

    models = {
        "Random": random_recommender,
        "Most Popular": most_popular_recommender,
        "Sequential Transition": sequential_recommender,
        "Category Popularity": category_recommender,
        "Fixed Hybrid Seq+Category": fixed_hybrid_recommender,
    }

    print("Evaluating models with ranking, novelty, diversity and coverage metrics...")
    detailed, summary, _ = evaluate_models(
        models,
        eval_users,
        test,
        train_history_lengths,
        item_prob,
        item_to_category,
        catalog_size,
        args.k,
    )

    short_summary = (
        detailed[detailed["history_length"] <= 2]
        .groupby("model")[
            [
                f"precision@{args.k}",
                f"recall@{args.k}",
                f"ndcg@{args.k}",
                f"novelty@{args.k}",
                f"category_diversity@{args.k}",
            ]
        ]
        .mean()
        .reset_index()
        .sort_values(f"recall@{args.k}", ascending=False)
    )

    examples = build_examples(eval_users, train, test, models, item_to_category, args.k)

    suffix = "full" if args.max_eval_users == 0 else f"sample_{len(eval_users)}"
    suffix = f"{suffix}_test"
    summary_path = output_dir / f"step1_metrics_summary_{suffix}.csv"
    short_path = output_dir / f"step1_metrics_short_history_{suffix}.csv"
    detailed_path = output_dir / f"step1_metrics_detailed_{suffix}.csv"
    examples_path = output_dir / f"step1_qualitative_examples_{suffix}.csv"

    summary.to_csv(summary_path, index=False)
    short_summary.to_csv(short_path, index=False)
    detailed.to_csv(detailed_path, index=False)
    examples.to_csv(examples_path, index=False)

    print("\nOverall metrics:")
    print(summary.to_string(index=False))
    print("\nShort-history metrics:")
    print(short_summary.to_string(index=False))
    print("\nWrote:")
    print(summary_path)
    print(short_path)
    print(detailed_path)
    print(examples_path)


if __name__ == "__main__":
    main()
