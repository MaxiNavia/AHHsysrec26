from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cache_utils import load_or_build, save_pickle
from step2_tune_adaptive import (
    cat_recs,
    combined_recs,
    hier_recs,
    load_core,
    seq_recs,
    transition_confidence,
    weights_for,
)
from step1_metrics_examples import filter_seen


PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_DIR / "H3" / "outputs"
CACHE_DIR = PROJECT_DIR / "H3" / "cache"
K = 10
MAX_EVAL_USERS = 50_000
SEED = 42


def build_components_for_user(user: int, core: dict, n: int = 200) -> dict[str, list[int]]:
    return {
        "seq": seq_recs(user, core, n),
        "cat": cat_recs(user, core, n),
        "hier": hier_recs(user, core, n),
    }


def candidate_pool(user: int, core: dict, components: dict[str, list[int]], tuned: list[int], n: int = 80):
    pool = tuned + components["seq"][:80] + components["cat"][:80] + components["hier"][:80]
    return filter_seen(pool, core["seen_items"].get(user, set()), n)


def build_step4_artifacts() -> dict:
    core = load_core(PROJECT_DIR, max_eval_users=MAX_EVAL_USERS, seed=SEED)
    config = json.loads((OUTPUT_DIR / "step2_best_config_sample_50000.json").read_text())
    test_item = core["test"].set_index("visitorid")["itemid"].astype(int).to_dict()

    components_by_user = {}
    tuned_by_user = {}
    pools_by_user = {}
    profiles_by_user = {}

    for idx, user in enumerate(core["eval_users"], start=1):
        components = build_components_for_user(user, core, n=200)
        components_wrapped = {user: components}
        tuned = combined_recs(user, core, components_wrapped, config, K, max_rank=200)
        pool = candidate_pool(user, core, components, tuned, n=80)

        h_len = int(core["train_history_lengths"].get(user, 0))
        t_conf = transition_confidence(user, core)
        user_train = core["train"][core["train"]["visitorid"].eq(user)].sort_values("timestamp")
        recent_history = []
        for row in user_train[["event", "itemid", "event_weight"]].tail(8).itertuples(index=False):
            item = int(row.itemid)
            recent_history.append(
                {
                    "event": row.event,
                    "itemid": item,
                    "categoryid": core["item_to_category"].get(item),
                    "weight": float(row.event_weight),
                }
            )

        components_by_user[int(user)] = components
        tuned_by_user[int(user)] = tuned
        pools_by_user[int(user)] = pool
        profiles_by_user[int(user)] = {
            "history_length": h_len,
            "transition_confidence": t_conf,
            "weights": weights_for(config, t_conf, h_len),
            "recent_history": recent_history,
            "true_item": int(test_item[user]),
            "true_category": core["item_to_category"].get(int(test_item[user])),
        }

        if idx % 10_000 == 0:
            print(f"cached step4 artifacts for {idx:,} users")

    return {
        "config": config,
        "eval_users": [int(user) for user in core["eval_users"]],
        "components_by_user": components_by_user,
        "tuned_by_user": tuned_by_user,
        "pools_by_user": pools_by_user,
        "profiles_by_user": profiles_by_user,
    }


def main() -> None:
    path = CACHE_DIR / "step4_analysis_artifacts_sample_50000.pkl"
    artifacts = load_or_build(path, build_step4_artifacts)
    print(f"loaded/cached artifacts for {len(artifacts['eval_users']):,} users")

    manifest = {
        "cache_file": str(path),
        "users": len(artifacts["eval_users"]),
        "contains": [
            "components_by_user: seq/cat/hier candidate rankings",
            "tuned_by_user: final tuned recommendations",
            "pools_by_user: candidate pools for reranking",
            "profiles_by_user: history, confidence, weights, true item/category",
        ],
    }
    (OUTPUT_DIR / "step4_cache_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    pd.DataFrame([manifest]).to_csv(OUTPUT_DIR / "step4_cache_manifest.csv", index=False)
    print(manifest)


if __name__ == "__main__":
    main()
