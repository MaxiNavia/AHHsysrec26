from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR  = PROJECT_DIR / "H3" / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

K = 10

# ── Color palette (consistent across all figures) ────────────────────────────
COLORS = {
    "Adaptive Hierarchical Hybrid (tuned)": "#C0392B",   # red   — proposed
    "Fixed Hybrid Seq+Category":            "#E67E22",   # orange
    "Sequential Transition":                "#2980B9",   # blue
    "ItemKNN":                              "#27AE60",   # green
    "GRU4Rec":                              "#8E44AD",   # purple
    "Category Popularity":                  "#16A085",   # teal
    "Most Popular":                         "#7F8C8D",   # grey
    "Random":                               "#BDC3C7",   # light grey
}

MODEL_ORDER = [
    "Adaptive Hierarchical Hybrid (tuned)",
    "Fixed Hybrid Seq+Category",
    "Sequential Transition",
    "ItemKNN",
    "GRU4Rec",
    "Category Popularity",
    "Most Popular",
    "Random",
]

SHORT_NAMES = {
    "Adaptive Hierarchical Hybrid (tuned)": "AHH (tuned)",
    "Fixed Hybrid Seq+Category":            "Fixed Hybrid",
    "Sequential Transition":                "Seq. Transition",
    "ItemKNN":                              "ItemKNN",
    "GRU4Rec":                              "GRU4Rec",
    "Category Popularity":                  "Cat. Popularity",
    "Most Popular":                         "Most Popular",
    "Random":                               "Random",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_final() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(OUTPUT_DIR / "final_metrics_summary_full.csv")
    short   = pd.read_csv(OUTPUT_DIR / "final_metrics_short_history_full.csv")
    return summary, short


def sort_df(df: pd.DataFrame) -> pd.DataFrame:
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    df = df.copy()
    df["_order"] = df["model"].map(order).fillna(99)
    return df.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def bar_colors(df: pd.DataFrame) -> list[str]:
    return [COLORS.get(m, "#95A5A6") for m in df["model"]]


def save(fig: plt.Figure, name: str) -> None:
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


# ── Figure 1 — Global comparison (Recall + nDCG) ────────────────────────────

def fig1_global_comparison(summary: pd.DataFrame) -> None:
    df = sort_df(summary)
    labels = [SHORT_NAMES.get(m, m) for m in df["model"]]
    recall = df[f"recall@{K}"].values
    ndcg   = df[f"ndcg@{K}"].values
    colors = bar_colors(df)

    x = np.arange(len(labels))
    width = 0.42

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_r = ax.barh(x + width / 2, recall, width, color=colors, alpha=0.92, label=f"Recall@{K}")
    bars_n = ax.barh(x - width / 2, ndcg,   width, color=colors, alpha=0.55, label=f"nDCG@{K}",
                     hatch="///", edgecolor="white")

    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Score", fontsize=11)
    ax.set_title(f"Global Performance — Recall@{K} and nDCG@{K}", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.legend(fontsize=9)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Annotate recall values
    for bar, val in zip(bars_r, recall):
        if val > 0.001:
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=8)

    fig.tight_layout()
    save(fig, "fig1_global_comparison.png")


# ── Figure 2 — Short-history comparison ─────────────────────────────────────

def fig2_short_history(short: pd.DataFrame) -> None:
    df = sort_df(short)
    labels = [SHORT_NAMES.get(m, m) for m in df["model"]]
    recall = df[f"recall@{K}"].values
    ndcg   = df[f"ndcg@{K}"].values
    colors = bar_colors(df)

    x = np.arange(len(labels))
    width = 0.42

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_r = ax.barh(x + width / 2, recall, width, color=colors, alpha=0.92, label=f"Recall@{K}")
    ax.barh(x - width / 2, ndcg, width, color=colors, alpha=0.55, label=f"nDCG@{K}",
            hatch="///", edgecolor="white")

    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Score", fontsize=11)
    ax.set_title(f"Short-History Users (≤2 interactions) — Recall@{K} and nDCG@{K}",
                 fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.legend(fontsize=9)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    for bar, val in zip(bars_r, recall):
        if val > 0.001:
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=8)

    fig.tight_layout()
    save(fig, "fig2_short_history.png")


# ── Figure 3 — Adaptive weights for a short-history user ─────────────────────

def fig3_adaptive_weights() -> None:
    """
    Recreates the weight breakdown diagram from the poster, now showing
    the tuned values for a typical short-history user (h=1, low seq confidence).
    Values come from the best config in step2.
    """
    components = ["Sequential", "Category", "Hierarchy", "Global Pop."]
    weights    = [0.33, 0.46, 0.16, 0.05]   # from poster (tuned config, h=1 user)
    colors_bar = ["#2980B9", "#E67E22", "#27AE60", "#7F8C8D"]

    fig, ax = plt.subplots(figsize=(6, 3.2))
    bars = ax.barh(components, weights, color=colors_bar, edgecolor="white", height=0.55)
    ax.set_xlim(0, 0.6)
    ax.set_xlabel("Adaptive weight", fontsize=11)
    ax.set_title("Adaptive Weights — Short-history user (≤2 interactions)",
                 fontsize=11, fontweight="bold")
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.invert_yaxis()

    for bar, val in zip(bars, weights):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=10, fontweight="bold")

    fig.tight_layout()
    save(fig, "fig3_adaptive_weights.png")


# ── Figure 4 — Ablation study ────────────────────────────────────────────────

def fig4_ablation() -> None:
    ablation_path = OUTPUT_DIR / "step7_ablation_mixed_final.csv"
    if not ablation_path.exists():
        print("  [skip] step7_ablation_mixed_final.csv not found")
        return

    df = pd.read_csv(ablation_path)
    # Keep only the models we want to show (drop full/sample duplicates)
    keep = [
        "Category Popularity",
        "Hierarchical Category",
        "Sequential Transition",
        "Adaptive Hybrid",
        "Fixed Hybrid Seq+Category",
        "Adaptive Hierarchical Hybrid",
        "Adaptive Hierarchical Hybrid (tuned)",
    ]
    df = df[df["model"].isin(keep)].drop_duplicates(subset="model")

    # Sort by recall
    df = df.sort_values(f"recall@{K}")
    labels = df["model"].str.replace("Adaptive Hierarchical Hybrid", "AHH", regex=False)
    labels = labels.str.replace("Fixed Hybrid Seq+Category", "Fixed Hybrid", regex=False)
    recalls = df[f"recall@{K}"].values
    clr = [COLORS.get(m, "#95A5A6") for m in df["model"]]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(labels, recalls, color=clr, edgecolor="white", height=0.6)
    ax.set_xlabel(f"Recall@{K}", fontsize=11)
    ax.set_title("Ablation Study — Component Contribution", fontsize=12, fontweight="bold")
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    for bar, val in zip(bars, recalls):
        ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    fig.tight_layout()
    save(fig, "fig4_ablation.png")


# ── Figure 5 — Hyperparameter sensitivity ────────────────────────────────────

def fig5_tuning_sensitivity() -> None:
    """Shows how Recall@10 varies with hier_max (the most impactful parameter)."""
    # Try new val file first, fall back to old file
    tune_path = next(
        (p for p in [
            OUTPUT_DIR / "step2_tuning_results_sample_25000_val.csv",
            OUTPUT_DIR / "step2_tuning_results_sample_50000.csv",
        ] if p.exists()),
        None,
    )
    if tune_path is None:
        print("  [skip] tuning results not found")
        return

    df = pd.read_csv(tune_path)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    params = [
        ("hier_max",  "hier_max (hierarchy weight ceiling)"),
        ("seq_min",   "seq_min (min sequential weight)"),
        ("seq_span",  "seq_span (sequential weight range)"),
    ]

    for ax, (param, label) in zip(axes, params):
        grouped = df.groupby(param)[f"recall@{K}"].agg(["mean", "std"]).reset_index()
        ax.errorbar(grouped[param], grouped["mean"], yerr=grouped["std"],
                    marker="o", color="#C0392B", linewidth=2, capsize=4)
        ax.set_xlabel(param, fontsize=10)
        ax.set_ylabel(f"Recall@{K}" if ax == axes[0] else "", fontsize=10)
        ax.set_title(f"Sensitivity: {param}", fontsize=10, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

    fig.suptitle("Hyperparameter Sensitivity Analysis", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig5_tuning_sensitivity.png")


# ── Figure 6 — Relevance vs Diversity tradeoff ───────────────────────────────

def fig6_tradeoff(summary: pd.DataFrame) -> None:
    df = sort_df(summary)
    # Exclude Random (trivially high diversity, zero recall)
    df = df[df["model"] != "Random"].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(7, 5))

    for _, row in df.iterrows():
        color = COLORS.get(row["model"], "#95A5A6")
        ax.scatter(row[f"category_diversity@{K}"], row[f"recall@{K}"],
                   color=color, s=120, zorder=3, edgecolors="white", linewidths=0.8)
        label = SHORT_NAMES.get(row["model"], row["model"])
        ax.annotate(label,
                    xy=(row[f"category_diversity@{K}"], row[f"recall@{K}"]),
                    xytext=(6, 3), textcoords="offset points", fontsize=8.5)

    ax.set_xlabel(f"Category Diversity@{K}", fontsize=11)
    ax.set_ylabel(f"Recall@{K}", fontsize=11)
    ax.set_title(f"Relevance vs. Intra-list Diversity Tradeoff",
                 fontsize=12, fontweight="bold")
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    save(fig, "fig6_tradeoff.png")


# ── Figure 7 — Novelty and Catalog Coverage ──────────────────────────────────

def fig7_novelty_coverage(summary: pd.DataFrame) -> None:
    df = sort_df(summary)
    # Only models with catalog_coverage column
    if f"catalog_coverage@{K}" not in df.columns:
        print("  [skip] catalog_coverage column missing")
        return

    labels  = [SHORT_NAMES.get(m, m) for m in df["model"]]
    novelty  = df[f"novelty@{K}"].values
    coverage = df[f"catalog_coverage@{K}"].values
    colors   = bar_colors(df)

    x     = np.arange(len(labels))
    width = 0.38

    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax2 = ax1.twinx()

    b1 = ax1.bar(x - width / 2, novelty,  width, color=colors, alpha=0.90, label=f"Novelty@{K}")
    b2 = ax2.bar(x + width / 2, coverage, width, color=colors, alpha=0.50,
                 hatch="///", edgecolor="white", label=f"Coverage@{K}")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax1.set_ylabel(f"Novelty@{K}", fontsize=10)
    ax2.set_ylabel(f"Catalog Coverage@{K}", fontsize=10)
    ax1.set_title(f"Novelty and Catalog Coverage by Model", fontsize=12, fontweight="bold")

    handles = [
        mpatches.Patch(color="#555", alpha=0.9,  label=f"Novelty@{K}"),
        mpatches.Patch(color="#555", alpha=0.5, hatch="///", label=f"Coverage@{K}"),
    ]
    ax1.legend(handles=handles, fontsize=9, loc="upper left")
    ax1.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax1.set_axisbelow(True)

    fig.tight_layout()
    save(fig, "fig7_novelty_coverage.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data…")
    summary, short = load_final()

    print("Generating figures…")
    fig1_global_comparison(summary)
    fig2_short_history(short)
    fig3_adaptive_weights()
    fig4_ablation()
    fig5_tuning_sensitivity()
    fig6_tradeoff(summary)
    fig7_novelty_coverage(summary)

    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
