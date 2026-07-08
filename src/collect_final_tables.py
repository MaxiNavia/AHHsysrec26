from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_DIR / "H3" / "outputs"
K = 10


def main() -> None:
    step1 = pd.read_csv(OUTPUT_DIR / "step1_metrics_summary_sample_25000_test.csv")
    itemknn = pd.read_csv(OUTPUT_DIR / "step3_itemknn_summary_sample_25000_test.csv")
    best = pd.read_csv(OUTPUT_DIR / "step2_best_model_metrics_sample_25000_test.csv")

    best_summary = pd.DataFrame(
        [
            {
                "model": "Adaptive Hierarchical Hybrid (tuned)",
                f"precision@{K}": best.loc[0, f"precision@{K}"],
                f"recall@{K}": best.loc[0, f"recall@{K}"],
                f"ndcg@{K}": best.loc[0, f"ndcg@{K}"],
                f"novelty@{K}": best.loc[0, f"novelty@{K}"],
                f"category_diversity@{K}": best.loc[0, f"category_diversity@{K}"],
                f"catalog_coverage@{K}": best.loc[0, f"catalog_coverage@{K}"],
            }
        ]
    )

    gru4rec = pd.read_csv(OUTPUT_DIR / "step8_gru4rec_summary_sample_25000_test.csv")

    final = pd.concat([step1, itemknn, best_summary, gru4rec], ignore_index=True)
    final = final.sort_values(f"recall@{K}", ascending=False)
    final.to_csv(OUTPUT_DIR / "final_metrics_summary_full.csv", index=False)

    step1_short = pd.read_csv(OUTPUT_DIR / "step1_metrics_short_history_sample_25000_test.csv")
    itemknn_short = pd.read_csv(OUTPUT_DIR / "step3_itemknn_short_history_sample_25000_test.csv")
    gru4rec_short = pd.read_csv(OUTPUT_DIR / "step8_gru4rec_short_history_sample_25000_test.csv")
    best_short = pd.DataFrame(
        [
            {
                "model": "Adaptive Hierarchical Hybrid (tuned)",
                f"precision@{K}": best.loc[0, f"short_precision@{K}"],
                f"recall@{K}": best.loc[0, f"short_recall@{K}"],
                f"ndcg@{K}": best.loc[0, f"short_ndcg@{K}"],
                f"novelty@{K}": best.loc[0, f"short_novelty@{K}"],
                f"category_diversity@{K}": best.loc[0, f"short_category_diversity@{K}"],
            }
        ]
    )
    final_short = pd.concat([step1_short, itemknn_short, best_short, gru4rec_short], ignore_index=True)
    final_short = final_short.sort_values(f"recall@{K}", ascending=False)
    final_short.to_csv(OUTPUT_DIR / "final_metrics_short_history_full.csv", index=False)

    print("Final overall table")
    print(final.to_string(index=False))
    print("\nFinal short-history table")
    print(final_short.to_string(index=False))


if __name__ == "__main__":
    main()
