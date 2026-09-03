"""Графики для кросс-когортного исследования (GTEx-здоровые + TCGA).

Основной сюжет здесь не «какой у нас AUC», а «сколько в этом AUC биологии и
сколько — различий между консорциумами». Поэтому половина графиков посвящена
не качеству модели, а проверкам на самообман.
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
from sklearn.decomposition import PCA  # noqa: E402

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", font_scale=0.95)
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.titleweight"] = "bold"

GROUP_C = {"healthy": "#27AE60", "adjacent": "#2E86C1", "tumor": "#C0392B"}
GROUP_RU = {"healthy": "Здоровые (GTEx)", "adjacent": "Норма рядом (TCGA)",
            "tumor": "Опухоль (TCGA)"}
ACCENT = "#F39C12"


def _save(fig, results_dir, name: str) -> None:
    path = Path(results_dir) / name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"  график: {path.name}")


def plot_honesty_panel(probe: pd.DataFrame, to_gtex: pd.DataFrame,
                       to_tcga: pd.DataFrame, results_dir, chosen: str) -> None:
    """Три проверки в один экран: отпечаток когорты и оба переноса."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    norms = list(probe["normalization"])
    x = np.arange(len(norms))
    colors = [ACCENT if n == chosen else "#7F8C8D" for n in norms]

    axes[0].bar(x, probe["auc"], color=colors, edgecolor="white")
    axes[0].axhline(0.5, ls="--", color="gray", lw=1)
    axes[0].set_ylim(0.4, 1.05)
    axes[0].set_title("Отпечаток когорты\n(рака нет ни в одной группе)")
    axes[0].set_ylabel("AUC различения GTEx и TCGA")
    for i, v in enumerate(probe["auc"]):
        axes[0].text(i, v + 0.015, f"{v:.3f}", ha="center", fontsize=9)

    axes[1].bar(x, to_gtex["healthy_false_positive_rate"] * 100,
                color=colors, edgecolor="white")
    axes[1].set_title("Перенос TCGA → GTEx\nздоровых объявлено больными")
    axes[1].set_ylabel("% ложных тревог у здоровых")
    for i, v in enumerate(to_gtex["healthy_false_positive_rate"]):
        axes[1].text(i, v * 100 + 1, f"{v:.1%}", ha="center", fontsize=9)

    axes[2].bar(x, to_tcga["adjacent_error_rate"] * 100, color=colors, edgecolor="white")
    axes[2].set_title("Перенос GTEx → TCGA\nнорму чужой когорты счёл опухолью")
    axes[2].set_ylabel("% ошибок на TCGA-норме")
    for i, v in enumerate(to_tcga["adjacent_error_rate"]):
        axes[2].text(i, v * 100 + 1, f"{v:.1%}", ha="center", fontsize=9)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(norms)
        ax.set_xlabel("нормировка внутри образца")

    fig.suptitle(f"Проверки на самообман (оранжевым — выбранная нормировка: {chosen})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, results_dir, "01_honesty_checks.png")


def plot_pca(X: pd.DataFrame, meta: pd.DataFrame, results_dir,
             title_suffix: str = "", filename: str = "02_pca.png") -> None:
    """Две главные компоненты: видно, разводит ли данные болезнь или когорта."""
    n = min(2000, X.shape[1])
    var = X.var(axis=0).nlargest(n).index
    values = X[var].to_numpy(dtype=np.float32)
    values = values - values.mean(axis=0)

    pcs = PCA(n_components=2, random_state=42).fit(values)
    coords = pcs.transform(values)
    ev = pcs.explained_variance_ratio_ * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))

    for group, color in GROUP_C.items():
        m = (meta["group"] == group).to_numpy()
        if m.any():
            axes[0].scatter(coords[m, 0], coords[m, 1], s=14, alpha=0.65,
                            color=color, label=f"{GROUP_RU[group]} (n={int(m.sum())})",
                            linewidths=0)
    axes[0].set_title("Окраска по группе образцов")
    axes[0].legend(fontsize=9, loc="best")

    for cohort, color in [("GTEX", "#8E44AD"), ("TCGA", "#E67E22")]:
        m = (meta["cohort"] == cohort).to_numpy()
        if m.any():
            axes[1].scatter(coords[m, 0], coords[m, 1], s=14, alpha=0.65,
                            color=color, label=f"{cohort} (n={int(m.sum())})", linewidths=0)
    axes[1].set_title("Окраска по когорте")
    axes[1].legend(fontsize=9, loc="best")

    for ax in axes:
        ax.set_xlabel(f"PC1 ({ev[0]:.1f}% дисперсии)")
        ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)")

    fig.suptitle(f"Структура данных{title_suffix}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, results_dir, filename)


def plot_risk_by_group(test_meta: pd.DataFrame, results_dir, threshold: float) -> None:
    """Распределение риска отдельно для здоровых, нормы-рядом и опухолей."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    bins = np.linspace(0, 1, 41)
    for group, color in GROUP_C.items():
        sub = test_meta[test_meta["group"] == group]
        if len(sub):
            axes[0].hist(sub["risk"], bins=bins, alpha=0.7, color=color,
                         label=f"{GROUP_RU[group]} (n={len(sub)})")
    axes[0].axvline(threshold, color=ACCENT, ls="--", lw=1.8, label="порог решения")
    axes[0].set_xlabel("Оценка риска P(опухоль)")
    axes[0].set_ylabel("Число образцов")
    axes[0].set_title("Риск на отложенном тесте")
    axes[0].legend(fontsize=9)

    order = [g for g in GROUP_C if (test_meta["group"] == g).any()]
    sns.boxplot(data=test_meta[test_meta["group"].isin(order)], x="group", y="risk",
                order=order, hue="group", legend=False,
                palette={g: GROUP_C[g] for g in order}, ax=axes[1], width=0.5)
    sns.stripplot(data=test_meta[test_meta["group"].isin(order)], x="group", y="risk",
                  order=order, color="black", alpha=0.35, size=2.5, ax=axes[1])
    axes[1].axhline(threshold, color=ACCENT, ls="--", lw=1.8)
    axes[1].set_xticks(range(len(order)))
    axes[1].set_xticklabels([GROUP_RU[g] for g in order], fontsize=9)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Оценка риска")
    axes[1].set_title("Разделение групп")

    fig.tight_layout()
    _save(fig, results_dir, "03_risk_by_group.png")


def plot_stage(by_stage: list[dict], results_dir, threshold: float) -> None:
    if not by_stage:
        return
    df = pd.DataFrame(by_stage)
    palette = {"I": "#27AE60", "II": "#F1C40F", "III": "#E67E22", "IV": "#C0392B"}
    colors = [palette.get(s, "#888") for s in df["stage"]]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.bar([f"Стадия {s}" for s in df["stage"]], df["rate"],
                  color=colors, edgecolor="white")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Доля пойманных опухолей")
    ax.set_title(f"Чувствительность по стадиям (порог {threshold:.3f})")
    for bar, r, n in zip(bars, df["rate"], df["n"]):
        ax.text(bar.get_x() + bar.get_width() / 2, r + 0.02,
                f"{r:.1%}\nn={n}", ha="center", fontsize=9)
    fig.tight_layout()
    _save(fig, results_dir, "04_stage_sensitivity.png")


def plot_models(summary: pd.DataFrame, results_dir, best_name: str) -> None:
    df = summary.sort_values("cv_auc_mean")
    colors = ["#C0392B" if n == best_name else "#2E86C1" for n in df["model"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 0.55 * len(df) + 3))
    axes[0].barh(df["model"], df["cv_auc_mean"], xerr=df["cv_auc_std"],
                 color=colors, edgecolor="white", capsize=3)
    axes[0].set_xlim(max(0.5, df["cv_auc_mean"].min() - 0.05), 1.005)
    axes[0].set_xlabel("ROC-AUC, кросс-валидация по пациентам")
    axes[0].set_title("Устойчивость")

    axes[1].barh(df["model"], df["test_auc"], color=colors, edgecolor="white")
    axes[1].set_xlim(max(0.5, df["test_auc"].min() - 0.05), 1.005)
    axes[1].set_xlabel("ROC-AUC, отложенный тест")
    axes[1].set_title("Обобщение")
    axes[1].set_yticklabels([])

    fig.suptitle("Классические модели на объединённом наборе",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, results_dir, "05_model_comparison.png")


def plot_permutation(perm: dict, real_auc: float, results_dir) -> None:
    """Реальный AUC против AUC на перемешанных метках."""
    fig, ax = plt.subplots(figsize=(8, 4.6))
    # src.evaluation.permutation_test называет поля null_auc_*/n_permutations,
    # старый локальный вариант — mean_auc/std_auc/n_rounds. Принимаем оба.
    mean = perm.get("null_auc_mean", perm.get("mean_auc"))
    std = perm.get("null_auc_std", perm.get("std_auc"))
    n_rounds = perm.get("n_permutations", perm.get("n_rounds", 0))

    ax.axvspan(mean - 2 * std, mean + 2 * std, color="#7F8C8D", alpha=0.25,
               label=f"перемешанные метки: {mean:.3f} ± {std:.3f}")
    ax.axvline(mean, color="#7F8C8D", lw=2)
    ax.axvline(0.5, color="black", ls=":", lw=1, label="случайное угадывание = 0.5")
    ax.axvline(real_auc, color="#C0392B", lw=2.5,
               label=f"настоящие метки: {real_auc:.4f}")

    ax.set_xlim(0.4, 1.02)
    ax.set_yticks([])
    ax.set_xlabel("ROC-AUC на кросс-валидации")
    p = perm.get("p_value")
    p_txt = f", p = {p:.4g}" if isinstance(p, float) else ""
    ax.set_title(f"Перестановочный тест ({n_rounds} прогонов{p_txt})")
    ax.legend(fontsize=9, loc="center left")
    fig.tight_layout()
    _save(fig, results_dir, "06_permutation_test.png")


def plot_panel_transfer(sweep: pd.DataFrame, results_dir, chosen_norm: str,
                        chosen_k: int) -> None:
    """Ложные тревоги у здоровых как функция размера генной панели."""
    if sweep is None or sweep.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    palette = {"logtpm": "#2E86C1", "zsample": "#8E44AD", "rank": "#16A085"}

    for norm, sub in sweep.groupby("normalization"):
        sub = sub.sort_values("n_genes")
        ax.plot(sub["n_genes"], sub["healthy_fpr"] * 100, marker="o", lw=2,
                color=palette.get(norm, "#888"), label=norm)

    ax.axvline(chosen_k, color=ACCENT, ls=":", lw=2,
               label=f"выбрано: {chosen_norm}, {chosen_k} генов")
    ax.set_xscale("log")
    ax.set_xticks(sorted(sweep["n_genes"].unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Размер генной панели, шт (лог. шкала)")
    ax.set_ylabel("% здоровых женщин GTEx, объявленных больными")
    ax.set_title("Чем меньше панель, тем лучше она переносится между когортами")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, results_dir, "07_panel_vs_transfer.png")


def plot_external(ext: dict, results_dir) -> None:
    """Распределение риска на внешней когорте FUSCC (Шанхай)."""
    pred = ext.get("predictions")
    if pred is None or not len(pred):
        return
    thr = ext["threshold"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bins = np.linspace(0, 1, 41)
    tum = pred[pred["label"] == 1]["risk"]
    nor = pred[pred["label"] == 0]["risk"]

    axes[0].hist(nor, bins=bins, alpha=0.75, color=GROUP_C["adjacent"],
                 label=f"Норма FUSCC (n={len(nor)})")
    axes[0].hist(tum, bins=bins, alpha=0.75, color=GROUP_C["tumor"],
                 label=f"Опухоль TNBC (n={len(tum)})")
    axes[0].axvline(thr, color=ACCENT, ls="--", lw=1.8, label="порог из обучения")
    axes[0].set_xlabel("Оценка риска P(опухоль)")
    axes[0].set_ylabel("Число образцов")
    axes[0].set_title("Внешняя когорта: Шанхай, FUSCC")
    axes[0].legend(fontsize=9)

    labels = ["Чувствительность\n(опухоли)", "Специфичность\n(нормы)"]
    vals = [ext["sensitivity"], ext["specificity"]]
    cis = [ext["sens_ci"], ext["spec_ci"]]
    err = [[v - c[0] for v, c in zip(vals, cis)], [c[1] - v for v, c in zip(vals, cis)]]
    bars = axes[1].bar(labels, vals, yerr=err, capsize=6,
                       color=[GROUP_C["tumor"], GROUP_C["adjacent"]], edgecolor="white")
    axes[1].set_ylim(0, 1.12)
    axes[1].set_ylabel("Доля")
    axes[1].set_title(f"Качество на невиданной когорте (AUC {ext['roc_auc']:.4f})")
    for bar, v in zip(bars, vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + 0.04, f"{v:.1%}",
                     ha="center", fontsize=10, fontweight="bold")

    fig.suptitle("Проверка на третьей когорте, которой модель не видела",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, results_dir, "08_external_fuscc.png")
