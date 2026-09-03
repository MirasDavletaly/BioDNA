"""Графики доказательной части: не «какое у нас качество», а «можно ли ему верить».

Обычные графики отчёта (ROC, матрица ошибок, маркеры) живут в visualization.py
и отвечают на вопрос «сколько». Здесь — второй слой, который отвечает на
вопрос «почему этому числу можно доверять»:

  11 карта сложности задач    — AUC = 1.000 осмысленна только рядом с задачами,
                                где до потолка далеко, и с негативным контролем;
  12 пермутационный тест      — нулевое распределение против наблюдаемого AUC;
  13 разброс по фолдам        — одна цифра на одном сплите есть одна реализация
                                случайной величины, а не «результат»;
  14 кривая обучения          — держится ли качество, если норм станет меньше;
  15 устойчивость панели      — воспроизводится ли список генов при пересборке.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.visualization import ACCENT, NORMAL_C, TUMOR_C, _save  # noqa: E402

logger = logging.getLogger(__name__)

GREEN = "#1E8449"
GREY = "#7F8C8D"
RED = "#CB4335"
INK = "#2C3E50"


def plot_task_landscape(task_rows: list[dict], results_dir) -> None:
    """Карта сложности: какие клинические вопросы решаются, а какие нет.

    Главный график обновлённого проекта. Полоска «AUC = 1.000» сама по себе
    не говорит о качестве метода — она говорит лишь о том, что задача лёгкая.
    Осмысленно только сравнение: вот вопрос, где потолок уже достигнут, вот
    три, где до потолка далеко, и вот негативный контроль, где сигнала нет.
    """
    if not task_rows:
        return
    df = pd.DataFrame(task_rows).sort_values("auc_mean")

    fig, ax = plt.subplots(figsize=(12.5, 0.62 * len(df) + 3))
    colors = []
    for _, r in df.iterrows():
        if r.get("is_control"):
            colors.append(GREY)
        elif r["auc_mean"] >= 0.90:
            colors.append(GREEN)
        elif r["auc_mean"] >= 0.70:
            colors.append(ACCENT)
        else:
            colors.append(RED)

    lo = (df["auc_mean"] - df["ci_low"]).clip(lower=0)
    hi = (df["ci_high"] - df["auc_mean"]).clip(lower=0)
    ax.barh(df["title"], df["auc_mean"], color=colors, edgecolor="white",
            xerr=[lo, hi], capsize=5, error_kw={"ecolor": INK, "lw": 1.3})

    ax.axvline(0.5, color=INK, ls="--", lw=1.6)
    ax.set_xlim(0.40, 1.08)
    ax.set_xlabel("ROC-AUC, вложенная кросс-валидация по пациентам (95% ДИ)")
    ax.set_title("Карта сложности: что по экспрессии генов предсказуемо, а что нет")

    for i, (_, r) in enumerate(df.iterrows()):
        ax.text(min(float(r["ci_high"]) + 0.012, 1.03), i,
                f"{r['auc_mean']:.3f}   n={int(r['n'])}", va="center", fontsize=9)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c)
               for c in (GREEN, ACCENT, RED, GREY)]
    ax.legend(handles,
              ["решается (AUC >= 0.90)", "частично (0.70-0.90)",
               "слабый сигнал (< 0.70)", "негативный контроль"],
              fontsize=9, loc="lower right")
    fig.tight_layout()
    _save(fig, results_dir, "11_task_landscape.png")


def plot_permutation_null(perm: dict, results_dir, task_title: str = "") -> None:
    """Нулевое распределение AUC при перемешанных метках против наблюдаемого."""
    if not perm or not perm.get("null_aucs"):
        return
    null = np.asarray(perm["null_aucs"], dtype=float)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.hist(null, bins=25, color=NORMAL_C, alpha=0.8, edgecolor="white",
            label=f"метки перемешаны (n={len(null)})")
    ax.axvline(perm["observed_auc"], color=TUMOR_C, lw=2.6,
               label=f"наблюдаемый AUC = {perm['observed_auc']:.4f}")
    ax.axvline(float(null.mean()), color=INK, ls="--", lw=1.5,
               label=f"среднее нулевого = {null.mean():.3f}")
    ax.set_xlabel("ROC-AUC")
    ax.set_ylabel("Число перестановок")
    title = "Пермутационный тест: результат не случаен"
    if task_title:
        title = f"{title} ({task_title})"
    ax.set_title(title)
    ax.legend(fontsize=9, loc="upper center")

    pval = perm.get("p_value")
    ax.text(0.02, 0.97, f"p = {pval:.4g}", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", color="#943126")
    fig.tight_layout()
    _save(fig, results_dir, "12_permutation_test.png")


def plot_cv_distribution(rep: dict, results_dir,
                         holdout_auc: float | None = None) -> None:
    """Разброс AUC по всем фолдам повторной CV вместо одной цифры."""
    if not rep:
        return
    aucs = np.asarray(rep.get("fold_aucs", []), dtype=float)
    if aucs.size == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(aucs, bins=min(20, max(5, len(aucs) // 3)), color=NORMAL_C,
                 edgecolor="white", alpha=0.85)
    axes[0].axvline(float(aucs.mean()), color=TUMOR_C, lw=2.2,
                    label=f"среднее {aucs.mean():.4f}")
    if holdout_auc is not None:
        axes[0].axvline(holdout_auc, color=ACCENT, ls="--", lw=2,
                        label=f"один отложенный тест {holdout_auc:.4f}")
    axes[0].set_xlabel("ROC-AUC фолда")
    axes[0].set_ylabel("Число фолдов")
    axes[0].set_title(f"Распределение по {len(aucs)} фолдам")
    axes[0].legend(fontsize=9)

    bp = axes[1].boxplot([aucs], widths=0.45, patch_artist=True,
                         medianprops={"color": TUMOR_C, "lw": 2})
    bp["boxes"][0].set_facecolor(NORMAL_C)
    bp["boxes"][0].set_alpha(0.6)
    jitter = np.random.default_rng(0).normal(1, 0.035, len(aucs))
    axes[1].scatter(jitter, aucs, s=22, color=INK, alpha=0.55, zorder=3)
    axes[1].set_xticks([1])
    axes[1].set_xticklabels(["повторная CV"])
    axes[1].set_ylabel("ROC-AUC")
    axes[1].set_title(f"Худший фолд: {aucs.min():.4f}")

    fig.suptitle("Одна цифра на одном сплите — это одна реализация случайной величины",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, results_dir, "13_cv_distribution.png")


def plot_learning_curve(curve: pd.DataFrame, results_dir,
                        n_available: int | None = None) -> None:
    """Сколько образцов редкого класса реально нужно, чтобы держать качество."""
    if curve is None or curve.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(curve["n_minority"], curve["auc_mean"], yerr=curve["auc_std"],
                marker="o", color=NORMAL_C, lw=2, capsize=4)
    ax.fill_between(curve["n_minority"],
                    curve["auc_mean"] - curve["auc_std"],
                    curve["auc_mean"] + curve["auc_std"],
                    color=NORMAL_C, alpha=0.15)
    if n_available:
        ax.axvline(n_available, color=ACCENT, ls=":", lw=2,
                   label=f"в наборе доступно {n_available}")
        ax.legend(fontsize=9, loc="lower right")
    ax.set_xlabel("Образцов редкого класса (нормы) в обучении")
    ax.set_ylabel("ROC-AUC на отложенных фолдах")
    ax.set_title("Кривая обучения: запас прочности по числу норм")
    fig.tight_layout()
    _save(fig, results_dir, "14_learning_curve.png")


def plot_stability(stab: pd.DataFrame, results_dir, top_n: int = 30) -> None:
    """Как часто ген попадает в панель при пересборке выборки.

    Ген с частотой 1.00 войдёт в панель на любых данных; ген с частотой 0.3
    попал в отчёт по случайности и на новых данных работать не будет.
    """
    if stab is None or stab.empty:
        return
    df = stab.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 0.34 * len(df) + 2.5))
    colors = [GREEN if f >= 0.9 else (ACCENT if f >= 0.6 else RED)
              for f in df["selection_freq"]]
    ax.barh(df["gene"], df["selection_freq"], color=colors, edgecolor="white")
    ax.axvline(0.6, color=INK, ls="--", lw=1.4, label="порог надёжности 0.6")
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Доля бутстрапов, где ген вошёл в панель "
                  f"(n={int(df['n_bootstraps'].iloc[0])})")
    ax.set_title("Устойчивость генной панели к пересборке выборки")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    _save(fig, results_dir, "15_gene_stability.png")


def create_evidence_report(
    results_dir,
    task_rows: list[dict] | None = None,
    permutation: dict | None = None,
    repeated_cv: dict | None = None,
    holdout_auc: float | None = None,
    learning: pd.DataFrame | None = None,
    n_minority_available: int | None = None,
    stability: pd.DataFrame | None = None,
) -> None:
    """Второй пакет графиков — про доверие к цифрам, а не про сами цифры."""
    logger.info("Строю графики доказательной части...")
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    if task_rows:
        plot_task_landscape(task_rows, results_dir)
    if permutation:
        plot_permutation_null(permutation, results_dir, "норма vs опухоль")
    if repeated_cv:
        plot_cv_distribution(repeated_cv, results_dir, holdout_auc)
    if learning is not None:
        plot_learning_curve(learning, results_dir, n_minority_available)
    if stability is not None:
        plot_stability(stability, results_dir)
