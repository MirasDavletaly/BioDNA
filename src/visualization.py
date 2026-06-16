import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve,
)

PALETTE = {
    "normal": "#3498db",
    "tumor":  "#e74c3c",
    "accent": "#2ecc71",
    "bg":     "#f8f9fa",
    "grid":   "#dee2e6",
}

plt.rcParams.update({
    "figure.facecolor": PALETTE["bg"],
    "axes.facecolor":   PALETTE["bg"],
    "axes.grid":        True,
    "grid.color":       PALETTE["grid"],
    "grid.alpha":       0.7,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
})


def plot_training_history(history: Dict, save_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Процесс обучения DNABERT Cancer Classifier", fontsize=14, fontweight="bold", y=1.02)

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], "o-", color=PALETTE["normal"],  label="Train Loss", lw=2)
    axes[0].plot(epochs, history["val_loss"],   "s-", color=PALETTE["tumor"],   label="Val Loss",   lw=2)
    axes[0].set_title("Loss", fontweight="bold")
    axes[0].set_xlabel("Эпоха")
    axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], "o-", color=PALETTE["normal"],  label="Train Acc", lw=2)
    axes[1].plot(epochs, history["val_acc"],   "s-", color=PALETTE["tumor"],   label="Val Acc",   lw=2)
    axes[1].set_title("Accuracy", fontweight="bold")
    axes[1].set_xlabel("Эпоха")
    axes[1].set_ylim([0.5, 1.05])
    axes[1].legend()
    axes[2].plot(epochs, history["val_auc"], "D-", color=PALETTE["accent"], label="Val AUC-ROC", lw=2)
    axes[2].axhline(y=0.9, color="gray", linestyle="--", alpha=0.5, label="Порог 0.90")
    axes[2].set_title("AUC-ROC на валидации", fontweight="bold")
    axes[2].set_xlabel("Эпоха")
    axes[2].set_ylim([0.5, 1.05])
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Сохранено: {save_path}")


def plot_roc_and_pr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Диагностические кривые", fontsize=14, fontweight="bold")

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc     = auc(fpr, tpr)

    ax1.plot(fpr, tpr, color=PALETTE["tumor"], lw=2.5,
             label=f"ROC (AUC = {roc_auc:.4f})")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    ax1.fill_between(fpr, tpr, alpha=0.1, color=PALETTE["tumor"])
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC-кривая", fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.05])
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    ax2.plot(recall, precision, color=PALETTE["normal"], lw=2.5,
             label=f"PR (AUC = {pr_auc:.4f})")
    ax2.fill_between(recall, precision, alpha=0.1, color=PALETTE["normal"])
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall кривая", fontweight="bold")
    ax2.legend(loc="lower left")
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Сохранено: {save_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    class_names: List[str] = ["Normal (Норма)", "Tumor (Опухоль)"],
) -> None:
    cm = confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Матрица ошибок классификации", fontsize=14, fontweight="bold")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], linewidths=0.5, cbar=False,
                annot_kws={"size": 14, "weight": "bold"})
    axes[0].set_title("Абсолютные значения", fontweight="bold")
    axes[0].set_ylabel("Истинный класс")
    axes[0].set_xlabel("Предсказанный класс")
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="RdYlGn",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], linewidths=0.5, cbar=False,
                annot_kws={"size": 13, "weight": "bold"})
    axes[1].set_title("Нормализованные (доли)", fontweight="bold")
    axes[1].set_ylabel("Истинный класс")
    axes[1].set_xlabel("Предсказанный класс")
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fig.text(0.5, -0.02,
             f"Sensitivity (Recall): {sensitivity:.1%}  |  "
             f"Specificity: {specificity:.1%}  |  "
             f"TP={tp}  FP={fp}  TN={tn}  FN={fn}",
             ha="center", fontsize=11, style="italic")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Сохранено: {save_path}")


def plot_probability_distribution(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    probs_normal = y_prob[y_true == 0]
    probs_tumor  = y_prob[y_true == 1]

    ax.hist(probs_normal, bins=30, alpha=0.6, color=PALETTE["normal"],
            label=f"Normal (n={len(probs_normal)})", density=True)
    ax.hist(probs_tumor,  bins=30, alpha=0.6, color=PALETTE["tumor"],
            label=f"Tumor  (n={len(probs_tumor)})",  density=True)

    ax.axvline(x=0.5, color="black", linestyle="--", alpha=0.7, label="Порог 0.5")
    ax.set_xlabel("Предсказанная вероятность (P[Tumor])")
    ax.set_ylabel("Плотность")
    ax.set_title("Распределение вероятностей опухоли", fontweight="bold")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Сохранено: {save_path}")


def plot_gene_importance(
    gene_names: List[str],
    importances: np.ndarray,
    top_n: int = 20,
    save_path: str = None,
) -> None:
    top_idx = np.argsort(importances)[-top_n:][::-1]
    top_genes  = [gene_names[i] for i in top_idx]
    top_scores = importances[top_idx]
    top_scores = (top_scores - top_scores.min()) / (top_scores.max() - top_scores.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, top_n))

    bars = ax.barh(range(top_n), top_scores[::-1], color=colors[::-1], edgecolor="white", height=0.7)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_genes[::-1], fontsize=9)
    ax.set_xlabel("Относительная важность (нормализованная)")
    ax.set_title(f"Топ-{top_n} диагностических генов", fontweight="bold", fontsize=13)
    ax.set_xlim([0, 1.1])

    # Значения на барах
    for i, (bar, score) in enumerate(zip(bars, top_scores[::-1])):
        ax.text(score + 0.01, bar.get_y() + bar.get_height()/2,
                f"{score:.3f}", va="center", ha="left", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Сохранено: {save_path}")
    else:
        plt.show()


def create_full_report(
    history: Dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    gene_names: List[str],
    gene_importances: np.ndarray,
    results_dir: str,
) -> None:
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    plot_training_history(history,     f"{results_dir}/01_training_history.png")
    plot_roc_and_pr(y_true, y_prob,    f"{results_dir}/02_roc_pr_curves.png")
    plot_confusion_matrix(y_true, y_pred, f"{results_dir}/03_confusion_matrix.png")
    plot_probability_distribution(y_true, y_prob, f"{results_dir}/04_probability_distribution.png")
    plot_gene_importance(gene_names, gene_importances, top_n=20,
                         save_path=f"{results_dir}/05_top_genes.png")

    print(f"\nВсе графики сохранены в: {results_dir}/")