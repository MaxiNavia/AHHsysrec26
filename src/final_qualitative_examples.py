from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from step2_tune_adaptive import (
    cat_recs,
    combined_recs,
    hier_recs,
    load_core,
    seq_recs,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_DIR / "H3" / "outputs"
K = 10


def fixed_hybrid_recs(user: int, core: dict, components: dict, k: int) -> list[int]:
    scores = defaultdict(float)
    for rank, item in enumerate(components[user]["seq"][:200], start=1):
        scores[int(item)] += 0.70 / rank
    for rank, item in enumerate(components[user]["cat"][:200], start=1):
        scores[int(item)] += 0.30 / rank
    for rank, item in enumerate(core["fallback_items"][:200], start=1):
        scores[int(item)] += 0.05 / rank
    seen = core["seen_items"].get(user, set())
    ranked = [item for item, _ in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))]
    out = []
    added = set()
    for item in ranked + core["fallback_items"]:
        item = int(item)
        if item in seen or item in added:
            continue
        out.append(item)
        added.add(item)
        if len(out) >= k:
            break
    return out


def main() -> None:
    core = load_core(PROJECT_DIR, max_eval_users=50_000, seed=42)
    config = json.loads((OUTPUT_DIR / "step2_best_config_sample_50000.json").read_text())
    test_item = core["test"].set_index("visitorid")["itemid"].astype(int).to_dict()

    selected = []
    labels = []
    for user in core["eval_users"]:
        hlen = core["train_history_lengths"].get(user, 0)
        if len(selected) >= 6:
            break
        components = {
            user: {
                "seq": seq_recs(user, core, 200),
                "cat": cat_recs(user, core, 200),
                "hier": hier_recs(user, core, 200),
            }
        }
        tuned = combined_recs(user, core, components, config, K, 200)
        fixed = fixed_hybrid_recs(user, core, components, K)
        true_item = test_item[user]
        tuned_hit = true_item in tuned
        fixed_hit = true_item in fixed
        if hlen <= 2 and tuned_hit and not fixed_hit:
            selected.append((user, "short_history_tuned_hit_fixed_miss"))
        elif hlen <= 2 and tuned_hit:
            selected.append((user, "short_history_tuned_hit"))
        elif hlen >= 5 and tuned_hit and not fixed_hit:
            selected.append((user, "longer_history_tuned_hit_fixed_miss"))
        elif hlen >= 5 and tuned_hit:
            selected.append((user, "longer_history_tuned_hit"))

    rows = []
    train_by_user = {
        int(user): user_df.sort_values("timestamp")[["event", "itemid"]].tail(8).values.tolist()
        for user, user_df in core["train"][
            core["train"]["visitorid"].isin([user for user, _ in selected])
        ].groupby("visitorid")
    }
    for user, label in selected:
        components = {
            user: {
                "seq": seq_recs(user, core, 200),
                "cat": cat_recs(user, core, 200),
                "hier": hier_recs(user, core, 200),
            }
        }
        tuned = combined_recs(user, core, components, config, K, 200)
        fixed = fixed_hybrid_recs(user, core, components, K)
        seq = components[user]["seq"][:K]
        true_item = int(test_item[user])
        history_pairs = train_by_user.get(user, [])
        rows.append(
            {
                "case": label,
                "visitorid": user,
                "history_length": core["train_history_lengths"].get(user, 0),
                "recent_history": " | ".join(f"{event}:{int(item)}" for event, item in history_pairs),
                "recent_categories": [
                    core["item_to_category"].get(int(item)) for _, item in history_pairs
                ],
                "sequential_recommendations": seq,
                "fixed_hybrid_recommendations": fixed,
                "tuned_adaptive_hierarchical_recommendations": tuned,
                "true_item": true_item,
                "true_category": core["item_to_category"].get(true_item),
                "sequential_hit": int(true_item in seq),
                "fixed_hybrid_hit": int(true_item in fixed),
                "tuned_hit": int(true_item in tuned),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "final_qualitative_examples_tuned.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
