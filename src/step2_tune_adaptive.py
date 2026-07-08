from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from cache_utils import load_or_build, load_pickle, split_val_test
from step1_metrics_examples import (
    category_diversity_at_k,
    filter_seen,
    load_events,
    load_latest_item_categories,
    novelty_at_k,
    ranking_metrics,
    sample_eval_users,
    temporal_leave_one_out,
)


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
    parser.add_argument("--mode", choices=["tune", "full"], default="tune")
    parser.add_argument("--best-config", type=Path, default=None)
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

    item_categories = load_or_build(
        cache_dir / "latest_item_categories.pkl",
        lambda: load_latest_item_categories(data_dir),
    )
    item_to_category = load_or_build(
        cache_dir / "item_to_category.pkl",
        lambda: dict(
            zip(item_categories["itemid"].astype(int), item_categories["categoryid"].astype(int))
        ),
    )
    seen_items = load_pickle(cache_dir / "seen_items.pkl")
    train_sequences = load_pickle(cache_dir / "train_sequences.pkl")
    train_history_lengths = load_pickle(cache_dir / "train_history_lengths.pkl")
    fallback_items, item_prob, _ = load_pickle(cache_dir / "global_popularity.pkl")
    catalog_items = load_pickle(cache_dir / "catalog_items.pkl")
    catalog_size = len(catalog_items)
    transition_rankings, transition_strength, confidence_scale = load_pickle(
        cache_dir / "transition_artifacts.pkl"
    )
    category_level_rankings, last_user_category_hier = load_pickle(
        cache_dir / "hierarchical_category_artifacts.pkl"
    )
    category_to_ancestors = load_pickle(cache_dir / "category_to_ancestors.pkl")

    direct_category_artifacts = load_or_build(
        cache_dir / "direct_category_artifacts.pkl",
        lambda: build_direct_category_artifacts(train, item_to_category, train_sequences),
    )

    return {
        "train": train,
        "test": test,
        "eval_users": eval_users,
        "item_to_category": item_to_category,
        "seen_items": seen_items,
        "train_sequences": train_sequences,
        "train_history_lengths": train_history_lengths,
        "fallback_items": fallback_items,
        "item_prob": item_prob,
        "catalog_size": catalog_size,
        "transition_rankings": transition_rankings,
        "transition_strength": transition_strength,
        "confidence_scale": confidence_scale,
        "category_level_rankings": category_level_rankings,
        "last_user_category_hier": last_user_category_hier,
        "category_to_ancestors": category_to_ancestors,
        "direct_category_artifacts": direct_category_artifacts,
    }


def build_direct_category_artifacts(train, item_to_category, train_sequences):
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
            category = item_to_category.get(int(item))
            if category is not None:
                last_user_category[int(user)] = int(category)
                break

    return category_to_items, last_user_category


def transition_confidence(user: int, core: dict) -> float:
    sequence = core["train_sequences"].get(user, [])
    if not sequence:
        return 0.0
    strength = core["transition_strength"].get(int(sequence[-1]), 0.0)
    scale = core["confidence_scale"]
    return float(min(1.0, math.log1p(strength) / math.log1p(scale)))


def seq_recs(user: int, core: dict, n: int) -> list[int]:
    sequence = core["train_sequences"].get(user, [])
    candidates = []
    if sequence:
        candidates.extend(core["transition_rankings"].get(int(sequence[-1]), []))
    candidates.extend(core["fallback_items"])
    return filter_seen(candidates, core["seen_items"].get(user, set()), n)


def hier_recs(user: int, core: dict, n: int) -> list[int]:
    category = core["last_user_category_hier"].get(int(user))
    seen = core["seen_items"].get(user, set())
    candidates = []
    if category is not None:
        for ancestor in core["category_to_ancestors"].get(int(category), [int(category)]):
            candidates.extend(core["category_level_rankings"].get(int(ancestor), []))
    candidates.extend(core["fallback_items"])
    return filter_seen(candidates, seen, n)


def cat_recs(user: int, core: dict, n: int) -> list[int]:
    category_to_items, last_user_category = core["direct_category_artifacts"]
    category = last_user_category.get(int(user))
    seen = core["seen_items"].get(user, set())
    candidates = []
    if category is not None:
        candidates.extend(category_to_items.get(int(category), []))
    candidates.extend(core["fallback_items"])
    return filter_seen(candidates, seen, n)


def rank_to_scores(items: list[int], weight: float, max_rank: int) -> dict[int, float]:
    scores = {}
    for rank, item in enumerate(items[:max_rank], start=1):
        scores[int(item)] = scores.get(int(item), 0.0) + weight / rank
    return scores


def weights_for(config: dict, transition_conf: float, history_length: int) -> dict[str, float]:
    history_conf = min(1.0, math.log1p(max(history_length, 0)) / math.log1p(10))
    seq_conf = config["transition_mix"] * transition_conf + (1 - config["transition_mix"]) * history_conf

    seq = config["seq_min"] + config["seq_span"] * seq_conf
    category = config["cat_base"] - config["cat_drop"] * seq_conf
    hierarchy = config["hier_max"] * ((1.0 - transition_conf) ** config["hier_power"])
    popularity = config["pop"]
    raw = {
        "seq": max(0.0, seq),
        "category": max(0.0, category),
        "hierarchy": max(0.0, hierarchy),
        "popularity": max(0.0, popularity),
    }
    total = sum(raw.values())
    return {key: value / total for key, value in raw.items()}


def combined_recs(user: int, core: dict, components: dict, config: dict, k: int, max_rank: int) -> list[int]:
    t_conf = transition_confidence(user, core)
    h_len = int(core["train_history_lengths"].get(user, 0))
    weights = weights_for(config, t_conf, h_len)
    scores = defaultdict(float)

    for item, score in rank_to_scores(components[user]["seq"], weights["seq"], max_rank).items():
        scores[item] += score
    for item, score in rank_to_scores(components[user]["cat"], weights["category"], max_rank).items():
        scores[item] += score
    if config["use_hierarchy"]:
        for item, score in rank_to_scores(components[user]["hier"], weights["hierarchy"], max_rank).items():
            scores[item] += score
    for item, score in rank_to_scores(core["fallback_items"][:max_rank], weights["popularity"], max_rank).items():
        scores[item] += score

    ranked = [item for item, _ in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))]
    return filter_seen(ranked + core["fallback_items"], core["seen_items"].get(user, set()), k)


def precompute_components(core: dict, n: int) -> dict[int, dict[str, list[int]]]:
    components = {}
    for idx, user in enumerate(core["eval_users"], start=1):
        components[user] = {
            "seq": seq_recs(user, core, n),
            "cat": cat_recs(user, core, n),
            "hier": hier_recs(user, core, n),
        }
        if idx % 50_000 == 0:
            print(f"precomputed components for {idx:,} users")
    return components


def evaluate_config(core: dict, components: dict, config: dict, k: int, max_rank: int) -> dict:
    test_item = core["test"].set_index("visitorid")["itemid"].astype(int).to_dict()
    rows = []
    recommended_items = set()

    for user in core["eval_users"]:
        true_item = int(test_item[user])
        recs = combined_recs(user, core, components, config, k, max_rank)
        recommended_items.update(recs)
        metrics = ranking_metrics(recs, true_item, k)
        metrics[f"novelty@{k}"] = novelty_at_k(recs, core["item_prob"], core["catalog_size"], k)
        metrics[f"category_diversity@{k}"] = category_diversity_at_k(
            recs, core["item_to_category"], k
        )
        metrics["history_length"] = int(core["train_history_lengths"].get(user, 0))
        rows.append(metrics)

    df = pd.DataFrame(rows)
    out = {col: float(df[col].mean()) for col in df.columns if col != "history_length"}
    short_df = df.loc[df["history_length"] <= 2]
    out[f"short_precision@{k}"] = float(short_df[f"precision@{k}"].mean())
    out[f"short_recall@{k}"] = float(short_df[f"recall@{k}"].mean())
    out[f"short_ndcg@{k}"] = float(short_df[f"ndcg@{k}"].mean())
    out[f"short_novelty@{k}"] = float(short_df[f"novelty@{k}"].mean())
    out[f"short_category_diversity@{k}"] = float(short_df[f"category_diversity@{k}"].mean())
    out[f"catalog_coverage@{k}"] = len(recommended_items) / core["catalog_size"]
    return out


def config_grid() -> list[dict]:
    configs = []
    for seq_min in [0.15, 0.20, 0.25]:
        for seq_span in [0.50, 0.60, 0.70]:
            for hier_max in [0.08, 0.12, 0.16, 0.20]:
                configs.append(
                    {
                        "name": f"smin{seq_min}_sspan{seq_span}_h{hier_max}",
                        "seq_min": seq_min,
                        "seq_span": seq_span,
                        "cat_base": 0.55,
                        "cat_drop": 0.30,
                        "hier_max": hier_max,
                        "hier_power": 1.0,
                        "pop": 0.05,
                        "transition_mix": 0.75,
                        "use_hierarchy": True,
                    }
                )
    return configs


def main() -> None:
    args = parse_args()
    output_dir = args.project_dir / "H3" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    core = load_core(args.project_dir, args.max_eval_users, args.seed)
    suffix = "full" if args.max_eval_users == 0 else f"sample_{len(core['eval_users'])}"
    suffix = f"{suffix}_test"
    print(f"mode={args.mode}, eval_users={len(core['eval_users']):,}")

    components = precompute_components(core, n=200)

    if args.mode == "tune":
        rows = []
        for idx, config in enumerate(config_grid(), start=1):
            print(f"evaluating {idx}/{len(config_grid())}: {config['name']}")
            metrics = evaluate_config(core, components, config, args.k, max_rank=200)
            row = {**config, **metrics}
            rows.append(row)
            print(
                f"  recall={metrics[f'recall@{args.k}']:.6f}, "
                f"ndcg={metrics[f'ndcg@{args.k}']:.6f}, "
                f"short_recall={metrics[f'short_recall@{args.k}']:.6f}"
            )

        results = pd.DataFrame(rows).sort_values(
            [f"recall@{args.k}", f"ndcg@{args.k}"], ascending=False
        )
        path = output_dir / f"step2_tuning_results_{suffix}.csv"
        results.to_csv(path, index=False)
        best = results.iloc[0].to_dict()
        config_keys = [
            "name",
            "seq_min",
            "seq_span",
            "cat_base",
            "cat_drop",
            "hier_max",
            "hier_power",
            "pop",
            "transition_mix",
            "use_hierarchy",
        ]
        best_config = {key: best[key] for key in config_keys}
        best_path = output_dir / f"step2_best_config_{suffix}.json"
        best_path.write_text(json.dumps(best_config, indent=2), encoding="utf-8")
        print(results.head(10).to_string(index=False))
        print(f"wrote {path}")
        print(f"wrote {best_path}")
        return

    if args.best_config is None:
        args.best_config = output_dir / "step2_best_config_sample_25000_test.json"
    config = json.loads(args.best_config.read_text(encoding="utf-8"))
    metrics = evaluate_config(core, components, config, args.k, max_rank=200)
    row = {**config, **metrics}
    result = pd.DataFrame([row])
    path = output_dir / f"step2_best_model_metrics_{suffix}.csv"
    result.to_csv(path, index=False)
    print(result.to_string(index=False))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
