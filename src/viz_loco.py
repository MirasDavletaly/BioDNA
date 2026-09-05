"""Графики для протокола leave-one-cohort-out.

Главный сюжет здесь один: внутренняя кросс-валидация не отличает модели, а
перенос на чужую когорту отличает их очень хорошо. Всё остальное - детали.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", font_scale=0.95)
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.titleweight"] = "bold"

NEW_C = "#1E8449"
OLD_C = "#B03A2E"
ACCENT = "#F39C12"

COHORT_RU = {"TCGA": "TCGA\n(США, все подтипы)",
             "FUSCC": "FUSCC\n(Шанхай, TNBC)",
             "VARLEY": "Varley\n(США, ER+ и TNBC)"}


def _save(fig, results_dir, name: str) -> None:
    path = Path(results_dir) / name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"  график: {path.name}")


def plot_comparison(best: dict, baseline: dict, results_dir) -> None:
    """v3 против v2 на каждой отложенной когорте."""
    cohorts = list(best["per_cohort"].keys())
    x = np.arange(len(cohorts))
    w = 0.36

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    panels = [
        ("roc_auc", "ROC-AUC", 0.90),
        ("sensitivity", "Чувствительность", 0.85),
        ("specificity", "Специфичность", 0.85),
    ]
    for ax, (key, title, floor) in zip(axes, panels):
        new = [best["per_cohort"][c][key] for c in cohorts]
        old = [baseline["per_cohort"][c][key] for c in cohorts]
        ax.bar(x - w / 2, new, w, color=NEW_C, edgecolor="white",
               label=f"v3: {best['model']}")
        ax.bar(x + w / 2, old, w, color=OLD_C, edgecolor="white",
               label=f"v2: {baseline['model']}")
        ax.set_xticks(x)
        ax.set_xticklabels([COHORT_RU.get(c, c) for c in cohorts], fontsize=8)
        ax.set_ylim(floor, 1.005)
        ax.set_title(title)
        for xi, (n, o) in enumerate(zip(new, old)):
            ax.text(xi - w / 2, n + 0.002, f"{n:.3f}", ha="center", fontsize=8)
            ax.text(xi + w / 2, o + 0.002, f"{o:.3f}", ha="center", fontsize=8)
    axes[0].legend(fontsize=8, loc="lower left")

    fig.suptitle("Отложенная когорта: модель её больницу не видела вообще",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, results_dir, "01_loco_comparison.png")


def plot_sweep(grid: pd.DataFrame, chosen: dict, results_dir) -> None:
    """Какие семейства алгоритмов переносятся, а какие нет."""
    if grid is None or grid.empty:
        return
    agg = (grid.groupby("model")["worst_auc"]
           .agg(["max", "min"]).sort_values("max"))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    colors = [ACCENT if m == chosen["model"] else "#7F8C8D" for m in agg.index]
    axes[0].barh(agg.index, agg["max"], color=colors, edgecolor="white")
    axes[0].set_xlim(max(0.90, agg["min"].min() - 0.01), 1.002)
    axes[0].set_xlabel("Лучшая достигнутая AUC на худшей когорте")
    axes[0].set_title("Потолок семейства алгоритмов")
    for i, v in enumerate(agg["max"]):
        axes[0].text(v + 0.0008, i, f"{v:.4f}", va="center", fontsize=8)

    pivot = grid.pivot_table(index="model", columns="normalization",
                             values="worst_auc", aggfunc="max")
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="RdYlGn", ax=axes[1],
                cbar_kws={"label": "AUC на худшей когорте"}, annot_kws={"size": 8})
    axes[1].set_title("Алгоритм x нормировка")
    axes[1].set_xlabel("нормировка внутри образца")
    axes[1].set_ylabel("")

    fig.suptitle("Перебор по критерию худшей когорты", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, results_dir, "02_sweep_landscape.png")


def plot_progression(spectrum: list[dict], threshold: float, results_dir,
                     predictions: pd.DataFrame | None = None) -> None:
    """Риск вдоль прогрессии для итоговой модели v3."""
    if not spectrum:
        return
    df = pd.DataFrame(spectrum)
    palette = {"норма": "#27AE60", "ранняя неоплазия": "#F1C40F",
               "DCIS (рак на месте)": "#E67E22", "инвазивная карцинома": "#C0392B"}
    colors = [palette.get(s, "#888") for s in df["subgroup"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    if predictions is not None and len(predictions):
        order = list(df["subgroup"])
        sns.boxplot(data=predictions, x="subgroup", y="risk", order=order,
                    hue="subgroup", legend=False, palette=palette, ax=axes[0],
                    width=0.55, fliersize=0)
        sns.stripplot(data=predictions, x="subgroup", y="risk", order=order,
                      color="black", alpha=0.5, size=4, ax=axes[0])
    else:
        axes[0].bar(df["subgroup"], df["mean_risk"], color=colors, edgecolor="white")
    axes[0].axhline(threshold, color=ACCENT, ls="--", lw=1.8, label="порог решения")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Оценка риска P(опухоль)")
    axes[0].set_title("Риск вдоль прогрессии")
    axes[0].tick_params(axis="x", rotation=18, labelsize=9)
    axes[0].legend(fontsize=9)

    bars = axes[1].bar(df["subgroup"], df["flagged_rate"], color=colors,
                       edgecolor="white")
    axes[1].set_ylim(0, 1.14)
    axes[1].set_ylabel("Доля, помеченная как опухоль")
    axes[1].set_title("Что модель называет раком")
    axes[1].tick_params(axis="x", rotation=18, labelsize=9)
    for bar, v, n in zip(bars, df["flagged_rate"], df["n"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + 0.03,
                     f"{v:.0%}\nn={n}", ha="center", fontsize=9)

    fig.suptitle("Спектр прогрессии: в обучении и выборе не участвовал",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, results_dir, "03_progression_v3.png")
