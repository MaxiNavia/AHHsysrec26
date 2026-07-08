"""
step10_paper_figures.py — Genera las 3 figuras para el paper H3.

Figuras:
    fig4_ablation.png           — Ablation study (Recall@10 por componente)
    fig5_tuning_sensitivity.png — Sensibilidad a hiperparámetros
    fig6_tradeoff.png           — Tradeoff relevancia vs diversidad

Uso:
    python3 src/step10_paper_figures.py

Las figuras se guardan en H3/outputs/figures/
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Rutas ────────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR  = PROJECT_DIR / "H3" / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

K = 10

# ── Paleta consistente con el poster ─────────────────────────────────────────
COLORS = {
    "Adaptive Hierarchical Hybrid (tuned)": "#C0392B",
    "Fixed Hybrid Seq+Category":            "#E67E22",
    "Sequential Transition":                "#2980B9",
    "ItemKNN":                              "#27AE60",
    "GRU4Rec":                              "#8E44AD",
    "Category Popularity":                  "#16A085",
    "Most Popular":                         "#7F8C8D",
    "Random":                               "#BDC3C7",
}

SHORT = {
    "Adaptive Hierarchical Hybrid (tuned)": "AHH (tuned)",
    "Fixed Hybrid Seq+Category":            "Fixed Hybrid",
    "Sequential Transition":                "Seq. Transition",
    "ItemKNN":                              "ItemKNN",
    "GRU4Rec":                              "GRU4Rec",
    "Category Popularity":                  "Cat. Popularity",
    "Most Popular":                         "Most Popular",
    "Random":                               "Random",
}

# Orden de modelos de mejor a peor recall para los gráficos de barras
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

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.dpi":       180,
})


def save(fig: plt.Figure, name: str) -> None:
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figura 4 — Ablation study
# ─────────────────────────────────────────────────────────────────────────────

def fig4_ablation() -> None:
    path = OUTPUT_DIR / "step7_ablation_mixed_final.csv"
    if not path.exists():
        print("  [skip] step7_ablation_mixed_final.csv no encontrado")
        return

    df = pd.read_csv(path)

    # Modelos que queremos mostrar en el ablation (de más simple a más complejo)
    keep_order = [
        "Category Popularity",
        "Sequential Transition",
        "Fixed Hybrid Seq+Category",
        "Adaptive Hybrid",
        "Adaptive Hierarchical Hybrid",
        "Adaptive Hierarchical Hybrid (tuned)",
    ]
    # Filtrar y preservar el orden
    df = df[df["model"].isin(keep_order)].drop_duplicates(subset="model")
    df["_ord"] = df["model"].map({m: i for i, m in enumerate(keep_order)})
    df = df.sort_values("_ord").reset_index(drop=True)

    labels = (
        df["model"]
        .str.replace("Adaptive Hierarchical Hybrid (tuned)", "AHH (tuned)", regex=False)
        .str.replace("Adaptive Hierarchical Hybrid", "AHH", regex=False)
        .str.replace("Fixed Hybrid Seq+Category", "Fixed Hybrid", regex=False)
        .str.replace("Sequential Transition", "Seq. Transition", regex=False)
        .str.replace("Category Popularity", "Cat. Popularity", regex=False)
    )
    recalls = df[f"recall@{K}"].values
    clrs    = [COLORS.get(m, "#95A5A6") for m in df["model"]]

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bars = ax.barh(labels, recalls, color=clrs, edgecolor="white",
                   height=0.6, linewidth=0.8)

    # Anotaciones de valor
    for bar, val in zip(bars, recalls):
        ax.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", ha="left", fontsize=8.5,
        )

    # Línea de referencia: Sequential Transition
    seq_val = df.loc[df["model"] == "Sequential Transition", f"recall@{K}"].values
    if len(seq_val):
        ax.axvline(seq_val[0], color="#2980B9", linestyle="--",
                   linewidth=1.2, alpha=0.6, label="Seq. Transition baseline")
        ax.legend(loc="lower right")

    ax.set_xlabel(f"Recall@{K}", fontsize=10)
    ax.set_title("Ablation Study — Contribution of Each Component",
                 fontsize=11, fontweight="bold", pad=8)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.invert_yaxis()

    # Margen derecho para las etiquetas de valor
    xmax = max(recalls) * 1.12
    ax.set_xlim(0, xmax)

    fig.tight_layout()
    save(fig, "fig4_ablation.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figura 5 — Sensibilidad a hiperparámetros
# ─────────────────────────────────────────────────────────────────────────────

def fig5_tuning_sensitivity() -> None:
    # Intentar primero el archivo nuevo (con split val), luego el viejo
    tune_path = next(
        (p for p in [
            OUTPUT_DIR / "step2_tuning_results_sample_25000_val.csv",
            OUTPUT_DIR / "step2_tuning_results_sample_50000.csv",
        ] if p.exists()),
        None,
    )
    if tune_path is None:
        print("  [skip] archivo de tuning no encontrado")
        return

    df = pd.read_csv(tune_path)
    print(f"  usando tuning desde: {tune_path.name}  ({len(df)} configs)")

    # Los 3 parámetros más importantes según el análisis
    params = [
        ("hier_max",       "hier\\_max"),
        ("seq_span",       "seq\\_span"),
        ("seq_min",        "seq\\_min"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), sharey=False)

    for ax, (param, label_tex) in zip(axes, params):
        grouped = (
            df.groupby(param)[f"recall@{K}"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values(param)
        )
        x    = grouped[param].values
        mean = grouped["mean"].values
        std  = grouped["std"].values

        ax.errorbar(
            x, mean, yerr=std,
            marker="o", markersize=5,
            color="#C0392B", linewidth=1.8,
            capsize=4, capthick=1.2,
            elinewidth=1.0,
        )

        # Resaltar el mejor valor
        best_idx = np.argmax(mean)
        ax.scatter(x[best_idx], mean[best_idx],
                   color="#C0392B", s=80, zorder=5,
                   edgecolors="black", linewidths=0.8)

        # Nombre del parámetro sin escapes LaTeX para matplotlib
        param_label = param.replace("_", "\\_") if False else param
        ax.set_xlabel(param_label, fontsize=10)
        ax.set_ylabel(f"Recall@{K}" if ax is axes[0] else "", fontsize=10)
        ax.set_title(f"Sensitivity: {param}", fontsize=10, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

        # Formatear eje y con 4 decimales
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
        ax.tick_params(axis="y", labelsize=8)

    fig.suptitle("Hyperparameter Sensitivity Analysis (Validation Set)",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig5_tuning_sensitivity.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figura 6 — Tradeoff relevancia vs diversidad
# ─────────────────────────────────────────────────────────────────────────────

def fig6_tradeoff() -> None:
    # Preferir el archivo final con GRU4Rec; si no existe, usar el viejo
    summary_path = next(
        (p for p in [
            OUTPUT_DIR / "final_metrics_summary_full.csv",
            OUTPUT_DIR / "step1_metrics_summary_full.csv",
        ] if p.exists()),
        None,
    )
    if summary_path is None:
        print("  [skip] archivo de summary no encontrado")
        return

    df = pd.read_csv(summary_path)
    print(f"  usando summary desde: {summary_path.name}  ({len(df)} modelos)")

    # Excluir Random del scatter (trivialmente alta diversidad, recall=0)
    df = df[df["model"] != "Random"].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(5.8, 4.2))

    for _, row in df.iterrows():
        color = COLORS.get(row["model"], "#95A5A6")
        is_proposed = row["model"] == "Adaptive Hierarchical Hybrid (tuned)"

        ax.scatter(
            row[f"category_diversity@{K}"],
            row[f"recall@{K}"],
            color=color,
            s=110 if is_proposed else 75,
            zorder=3,
            edgecolors="black" if is_proposed else "white",
            linewidths=1.2 if is_proposed else 0.6,
        )

        label = SHORT.get(row["model"], row["model"])

        # Ajustes de posición para evitar solapamiento
        offsets = {
            "Adaptive Hierarchical Hybrid (tuned)": (5, 4),
            "Fixed Hybrid Seq+Category":            (5, -8),
            "Sequential Transition":                (5, 4),
            "ItemKNN":                              (5, 4),
            "GRU4Rec":                              (5, -8),
            "Category Popularity":                  (-5, 6),
            "Most Popular":                         (5, 4),
        }
        dx, dy = offsets.get(row["model"], (5, 4))

        ax.annotate(
            label,
            xy=(row[f"category_diversity@{K}"], row[f"recall@{K}"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold" if is_proposed else "normal",
        )

    ax.set_xlabel(f"Category Diversity@{K}", fontsize=10)
    ax.set_ylabel(f"Recall@{K}", fontsize=10)
    ax.set_title(
        "Relevance--Category Diversity Tradeoff",
        fontsize=11, fontweight="bold", pad=8,
    )
    ax.xaxis.grid(True, linestyle="--", alpha=0.35)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    # Anotación explicativa de las esquinas
    ax.text(0.03, 0.97, "← lower diversity\nhigher recall →",
            transform=ax.transAxes, fontsize=7.5,
            va="top", color="#555555", style="italic")
    ax.text(0.97, 0.03, "higher diversity →\nlower recall ↓",
            transform=ax.transAxes, fontsize=7.5,
            va="bottom", ha="right", color="#555555", style="italic")

    fig.tight_layout()
    save(fig, "fig6_tradeoff.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Generando figuras para el paper...")
    fig4_ablation()
    fig5_tuning_sensitivity()
    fig6_tradeoff()
    print(f"\nFiguras guardadas en: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
