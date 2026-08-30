"""Графики отчёта. Только matplotlib/seaborn, никаких «эпох обучения».

Раньше здесь рисовалась выдуманная кривая обучения нейросети — у классических
моделей эпох нет, поэтому её место заняли честные вещи: сравнение моделей,
ROC/PR, калибровка, чувствительность по стадиям и карта маркеров по геному.
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
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", font_scale=0.95)
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.titleweight"] = "bold"

NORMAL_C = "#2E86C1"
TUMOR_C = "#C0392B"
ACCENT = "#F39C12"


def _save(fig, results_dir, name: str) -> None:
    path = Path(results_dir) / name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"  график: {path.name}")


def plot_model_comparison(summary: pd.DataFrame, results_dir, best_name: str) -> None:
    """Сравнение всех классических моделей по CV и по отложенному тесту."""
    df = summary.sort_values("cv_auc_mean")
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.55 * len(df) + 3))

    colors = [TUMOR_C if n == best_name else NORMAL_C for n in df["model"]]

    axes[0].barh(df["model"], df["cv_auc_mean"], xerr=df["cv_auc_std"],
                 color=colors, edgecolor="white", capsize=3)
    axes[0].set_xlim(max(0.5, df["cv_auc_mean"].min() - 0.08), 1.005)
    axes[0].set_xlabel("ROC-AUC, кросс-валидация по пациентам")
    axes[0].set_title("Устойчивость (5-fold CV)")
    for y, v in enumerate(df["cv_auc_mean"]):
        axes[0].text(v + 0.004, y, f"{v:.3f}", va="center", fontsize=9)

    axes[1].barh(df["model"], df["test_auc"], color=colors, edgecolor="white")
    axes[1].set_xlim(max(0.5, df["test_auc"].min() - 0.08), 1.005)
    axes[1].set_xlabel("ROC-AUC, отложенный тест")
    axes[1].set_title("Обобщение на невиданных пациентах")
    axes[1].set_yticklabels([])
    for y, v in enumerate(df["test_auc"]):
        axes[1].text(v + 0.004, y, f"{v:.3f}", va="center", fontsize=9)

    fig.suptitle("Сравнение классических ML-моделей", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, results_dir, "01_model_comparison.png")


def plot_roc_pr(y_true, y_prob, results_dir, threshold: float, label: str = "тест") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    fpr, tpr, thr = roc_curve(y_true, y_prob)
    auc = np.trapezoid(tpr, fpr)
    axes[0].plot(fpr, tpr, color=TUMOR_C, lw=2.2, label=f"AUC = {auc:.4f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", lw=1)

    idx = int(np.argmin(np.abs(thr - threshold)))
    axes[0].scatter([fpr[idx]], [tpr[idx]], s=90, color=ACCENT, zorder=5,
                    edgecolor="black", linewidth=0.7,
                    label=f"скрининговый порог {threshold:.3f}\n"
                          f"чувств. {tpr[idx]:.3f} / спец. {1 - fpr[idx]:.3f}")
    axes[0].set_xlabel("1 − специфичность (ложные тревоги)")
    axes[0].set_ylabel("Чувствительность (пойманные опухоли)")
    axes[0].set_title(f"ROC-кривая ({label})")
    axes[0].legend(loc="lower right", fontsize=9)

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    baseline = float(np.mean(y_true))
    axes[1].plot(rec, prec, color=NORMAL_C, lw=2.2)
    axes[1].axhline(baseline, ls="--", color="gray", lw=1,
                    label=f"доля опухолей = {baseline:.3f}")
    axes[1].set_xlabel("Полнота (recall)")
    axes[1].set_ylabel("Точность (precision)")
    axes[1].set_title(f"Precision-Recall ({label})")
    axes[1].legend(loc="lower left", fontsize=9)

    fig.tight_layout()
    _save(fig, results_dir, "02_roc_pr_curves.png")


def plot_confusion(y_true, y_prob, results_dir, threshold: float) -> None:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, norm, title, fmt in [
        (axes[0], None, "Абсолютные значения", "d"),
        (axes[1], "true", "Доля внутри класса", ".3f"),
    ]:
        m = cm if norm is None else cm / cm.sum(axis=1, keepdims=True)
        sns.heatmap(m, annot=True, fmt=fmt, cmap="RdYlBu_r", cbar=False, ax=ax,
                    xticklabels=["Норма", "Опухоль"],
                    yticklabels=["Норма", "Опухоль"], annot_kws={"size": 13})
        ax.set_xlabel("Предсказание")
        ax.set_ylabel("Истина")
        ax.set_title(title)

    tn, fp, fn, tp = cm.ravel()
    fig.suptitle(f"Матрица ошибок при пороге {threshold:.3f}   "
                 f"(пропущено опухолей: {fn}, ложных тревог: {fp})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, results_dir, "03_confusion_matrix.png")


def plot_probability_distribution(y_true, y_prob, results_dir,
                                  low: float, high: float) -> None:
    """Распределение риска с серой зоной — как выглядит решение врача."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 1, 41)
    ax.hist(y_prob[y_true == 0], bins=bins, alpha=0.75, color=NORMAL_C,
            label=f"Норма (n={int((y_true == 0).sum())})")
    ax.hist(y_prob[y_true == 1], bins=bins, alpha=0.75, color=TUMOR_C,
            label=f"Опухоль (n={int((y_true == 1).sum())})")

    if high - low > 0.005:
        ax.axvspan(low, high, color=ACCENT, alpha=0.18)
        ax.axvline(high, color=ACCENT, ls="--", lw=1.5)
        ax.text((low + high) / 2, ax.get_ylim()[1] * 0.92, "серая зона\n(нужна биопсия)",
                ha="center", fontsize=9, color="#7D6608")
    else:
        # Классы разделились полностью — серой зоны не осталось.
        ax.text(low, ax.get_ylim()[1] * 0.95, " порог решения",
                fontsize=9, color="#7D6608", va="top")
    ax.axvline(low, color=ACCENT, ls="--", lw=1.5)

    ax.set_xlabel("Оценка риска P(опухоль)")
    ax.set_ylabel("Число образцов")
    ax.set_title("Распределение риска и зоны решения")
    ax.legend()
    fig.tight_layout()
    _save(fig, results_dir, "04_risk_distribution.png")


def plot_calibration(y_true, y_prob, results_dir) -> None:
    """Можно ли трактовать выход как вероятность, а не просто как score."""
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    n_bins = min(10, max(3, len(y_true) // 12))
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="идеальная калибровка")
    ax.plot(mean_pred, frac_pos, "o-", color=TUMOR_C, lw=2, label="модель")
    ax.set_xlabel("Предсказанная вероятность")
    ax.set_ylabel("Наблюдаемая доля опухолей")
    ax.set_title("Калибровка оценки риска")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, results_dir, "05_calibration.png")


def plot_stage_sensitivity(stage_df: pd.DataFrame, results_dir, threshold: float) -> None:
    """Главный график для «ранней стадии»: ловим ли I и II."""
    df = stage_df[stage_df["stage"] != "Unknown"].copy()
    if df.empty:
        return

    palette = {"Normal": NORMAL_C, "I": "#27AE60", "II": "#F1C40F",
               "III": "#E67E22", "IV": TUMOR_C}
    colors = [palette.get(s, "#888888") for s in df["stage"]]
    labels = ["Норма" if s == "Normal" else f"Стадия {s}" for s in df["stage"]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    bars = axes[0].bar(labels, df["rate"], color=colors, edgecolor="white")
    axes[0].set_ylim(0, 1.09)
    axes[0].set_ylabel("Доля верно распознанных")
    axes[0].set_title(f"Распознавание по стадиям (порог {threshold:.3f})")
    for bar, r, n in zip(bars, df["rate"], df["n"]):
        axes[0].text(bar.get_x() + bar.get_width() / 2, r + 0.02,
                     f"{r:.1%}\nn={n}", ha="center", fontsize=9)
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].bar(labels, df["mean_prob"], color=colors, edgecolor="white")
    axes[1].axhline(threshold, ls="--", color=ACCENT, lw=1.5, label="порог решения")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Средняя оценка риска")
    axes[1].set_title("Риск растёт со стадией")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].legend(fontsize=9)

    fig.suptitle("Ранняя диагностика: чувствительность к стадии I—II",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, results_dir, "06_stage_sensitivity.png")


def plot_top_genes(markers: pd.DataFrame, results_dir, top_n: int = 25) -> None:
    """Топ-маркеры с геномными координатами прямо в подписи."""
    df = markers.head(top_n).iloc[::-1]
    if df.empty:
        return

    labels = []
    for _, r in df.iterrows():
        locus = r.get("locus_short") or "?"
        labels.append(f"{r['gene']}  ·  {locus}")

    fig, ax = plt.subplots(figsize=(10, 0.36 * len(df) + 2))
    up = df["direction"].to_numpy() if "direction" in df else np.ones(len(df))
    colors = [TUMOR_C if u > 0 else NORMAL_C for u in up]
    ax.barh(labels, df["importance"], color=colors, edgecolor="white")
    ax.set_xlabel("Вклад в решение модели")
    ax.set_title(f"Топ-{len(df)} генов-маркеров и их локусы (hg38)")

    handles = [plt.Rectangle((0, 0), 1, 1, color=TUMOR_C),
               plt.Rectangle((0, 0), 1, 1, color=NORMAL_C)]
    ax.legend(handles, ["выше в опухоли", "ниже в опухоли"], fontsize=9, loc="lower right")
    fig.tight_layout()
    _save(fig, results_dir, "07_top_genes.png")


def plot_genome_manhattan(scores: pd.DataFrame, results_dir,
                          label_top: int = 12) -> None:
    """Манхэттен-график: сила связи гена с раком против ДНК-координаты."""
    df = scores.dropna(subset=["genome_pos", "chrom"]).copy()
    if df.empty:
        logger.warning("  манхэттен-график пропущен: нет координат")
        return

    from src.gene_annotation import CHROM_SIZES, MAIN_CHROMS, chromosome_offsets

    df["logp"] = -np.log10(np.clip(df["p_value"], 1e-300, 1.0))
    offsets = chromosome_offsets()

    fig, ax = plt.subplots(figsize=(15, 5.5))
    for i, chrom in enumerate(MAIN_CHROMS):
        sub = df[df["chrom"] == chrom]
        if sub.empty:
            continue
        ax.scatter(sub["genome_pos"], sub["logp"], s=5,
                   color=NORMAL_C if i % 2 == 0 else "#7F8C8D", alpha=0.6, linewidths=0)

    # Порог Бонферрони — сколько генов реально значимы после поправки.
    bonf = -np.log10(0.05 / max(len(df), 1))
    ax.axhline(bonf, color=TUMOR_C, ls="--", lw=1.2,
               label=f"Бонферрони (p<0.05/{len(df)})")

    top = df.nlargest(label_top, "logp")
    for _, r in top.iterrows():
        ax.annotate(r["gene"], (r["genome_pos"], r["logp"]),
                    fontsize=8, xytext=(0, 5), textcoords="offset points",
                    ha="center", color="#943126")

    centers = [offsets[c] + CHROM_SIZES[c] / 2 for c in MAIN_CHROMS]
    ax.set_xticks(centers)
    ax.set_xticklabels([c.replace("chr", "") for c in MAIN_CHROMS], fontsize=8)
    ax.set_xlim(0, offsets[MAIN_CHROMS[-1]] + CHROM_SIZES[MAIN_CHROMS[-1]])
    ax.set_xlabel("Хромосома (координаты генома hg38)")
    ax.set_ylabel("-log10(p), ANOVA норма vs опухоль")
    ax.set_title("Где в геноме лежат маркеры рака груди")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, results_dir, "08_genome_manhattan.png")


def plot_chromosome_load(scores: pd.DataFrame, results_dir, alpha: float = 0.05) -> None:
    """Сколько значимых маркеров даёт каждая хромосома (с поправкой на её размер)."""
    df = scores.dropna(subset=["chrom"]).copy()
    if df.empty:
        return

    from src.gene_annotation import MAIN_CHROMS

    bonf_p = alpha / max(len(df), 1)
    df["significant"] = df["p_value"] < bonf_p

    grouped = df.groupby("chrom").agg(total=("gene", "size"),
                                      sig=("significant", "sum"))
    grouped = grouped.reindex([c for c in MAIN_CHROMS if c in grouped.index])
    grouped["share"] = grouped["sig"] / grouped["total"]

    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.bar([c.replace("chr", "") for c in grouped.index], grouped["share"] * 100,
           color=NORMAL_C, edgecolor="white")
    ax.set_xlabel("Хромосома")
    ax.set_ylabel("% генов хромосомы, значимых по Бонферрони")
    ax.set_title("Вклад хромосом в различение нормы и опухоли")
    for i, (share, sig) in enumerate(zip(grouped["share"], grouped["sig"])):
        ax.text(i, share * 100 + 0.4, str(int(sig)), ha="center", fontsize=8)
    fig.tight_layout()
    _save(fig, results_dir, "09_chromosome_load.png")


def plot_panel_scan(panel: pd.DataFrame, results_dir, minimal: int | None = None) -> None:
    """Качество против размера генной панели — где выходит на плато."""
    if panel is None or panel.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(panel["n_genes"], panel["cv_auc_mean"], yerr=panel["cv_auc_std"],
                marker="o", color=NORMAL_C, lw=2, capsize=4, label="ROC-AUC (CV)")
    ax.plot(panel["n_genes"], panel["sensitivity"], marker="s", ls="--",
            color=TUMOR_C, lw=1.8, label="чувствительность")
    ax.set_xscale("log")
    ax.set_ylim(min(0.95, float(panel[["cv_auc_mean", "sensitivity"]].min().min()) - 0.01), 1.005)
    ax.set_xticks(panel["n_genes"])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Размер генной панели, шт (лог. шкала)")
    ax.set_ylabel("Качество на кросс-валидации")
    ax.set_title("Сколько генов реально нужно для диагноза")

    if minimal:
        ax.axvline(minimal, color=ACCENT, ls=":", lw=2,
                   label=f"плато с {minimal} генов")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    _save(fig, results_dir, "10_panel_size.png")


def create_full_report(
    summary: pd.DataFrame,
    best_name: str,
    y_true,
    y_prob,
    threshold: float,
    low: float,
    high: float,
    stage_df: pd.DataFrame,
    markers: pd.DataFrame,
    genome_scores: pd.DataFrame,
    results_dir,
    panel_scan: pd.DataFrame | None = None,
) -> None:
    logger.info("Строю графики отчёта...")
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    plot_model_comparison(summary, results_dir, best_name)
    plot_roc_pr(y_true, y_prob, results_dir, threshold)
    plot_confusion(y_true, y_prob, results_dir, threshold)
    plot_probability_distribution(y_true, y_prob, results_dir, low, high)
    plot_calibration(y_true, y_prob, results_dir)
    plot_stage_sensitivity(stage_df, results_dir, threshold)
    plot_top_genes(markers, results_dir)
    plot_genome_manhattan(genome_scores, results_dir)
    plot_chromosome_load(genome_scores, results_dir)
    if panel_scan is not None:
        minimal = int(panel_scan.loc[panel_scan['cv_auc_mean'].idxmax(), 'n_genes'])
        plot_panel_scan(panel_scan, results_dir, minimal)
