"""Классические ML-модели для детекции рака по экспрессии генов.

Никаких нейросетей и трансформеров: логистическая регрессия, SVM,
случайный лес, бустинг, наивный Байес, kNN и их ансамбль.

Два принципа, без которых цифры были бы враньём:
  1. Отбор генов и масштабирование — ВНУТРИ Pipeline, обучаются на train-фолде.
  2. Разбиение — по ПАЦИЕНТАМ (StratifiedGroupKFold): у части больных в наборе
     есть и опухоль, и своя же норма; случайный сплит растащил бы их по
     train/test и модель узнавала бы пациента, а не болезнь.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

RANDOM_STATE = 42


def _front(n_genes: int) -> list[tuple[str, object]]:
    """Общая «голова» пайплайна: пропуски -> константы -> масштаб -> отбор генов."""
    return [
        ("impute", SimpleImputer(strategy="median")),
        ("varfilter", VarianceThreshold(threshold=1e-8)),
        ("scale", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=n_genes)),
    ]


def build_models(n_genes: int = 300) -> dict[str, Pipeline]:
    """Зоопарк классических классификаторов, каждый — самодостаточный Pipeline."""
    base = {
        "Logistic Regression (L2)": LogisticRegression(
            C=1.0, max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE),
        "Logistic Regression (L1)": LogisticRegression(
            C=0.1, l1_ratio=1.0, solver="liblinear", max_iter=5000,
            class_weight="balanced", random_state=RANDOM_STATE),
        # SVC сам вероятностей не даёт — оборачиваем в Платт-калибровку.
        "SVM (RBF)": CalibratedClassifierCV(
            SVC(C=1.0, kernel="rbf", gamma="scale",
                class_weight="balanced", random_state=RANDOM_STATE),
            method="sigmoid", ensemble=False, cv=3),
        "Random Forest": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, n_jobs=-1,
            class_weight="balanced_subsample", random_state=RANDOM_STATE),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=2, n_jobs=-1,
            class_weight="balanced", random_state=RANDOM_STATE),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
            l2_regularization=1.0, class_weight="balanced",
            random_state=RANDOM_STATE),
        "Naive Bayes": GaussianNB(),
        "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5, weights="distance"),
    }

    models = {name: Pipeline(_front(n_genes) + [("clf", clf)]) for name, clf in base.items()}

    # Мягкое голосование трёх разнотипных моделей: линейная + ядровая + деревья.
    models["Ensemble (soft vote)"] = Pipeline(_front(n_genes) + [
        ("clf", VotingClassifier(
            estimators=[
                ("lr", base["Logistic Regression (L2)"]),
                ("svm", base["SVM (RBF)"]),
                ("rf", base["Random Forest"]),
            ],
            voting="soft",
        ))
    ])
    return models


def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    """Полный набор метрик. sensitivity = доля пойманных опухолей (главное для скрининга)."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "threshold": float(threshold),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def cross_validate_oof(
    model: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
) -> tuple[np.ndarray, list[float]]:
    """Out-of-fold вероятности: каждый образец предсказан моделью, его не видевшей."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y), dtype=float)
    fold_aucs: list[float] = []

    for train_idx, val_idx in cv.split(X, y, groups=groups):
        model.fit(X[train_idx], y[train_idx])
        probs = model.predict_proba(X[val_idx])[:, 1]
        oof[val_idx] = probs
        fold_aucs.append(float(roc_auc_score(y[val_idx], probs)))

    return oof, fold_aucs


def threshold_for_sensitivity(y_true, y_prob, target: float = 0.98) -> float:
    """Наименьший порог, дающий чувствительность >= target.

    Для скрининга это правильная постановка: пропущенная опухоль (FN) стоит
    несоизмеримо дороже ложной тревоги (FP), которую снимет биопсия.
    """
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    ok = np.where(tpr >= target)[0]
    if len(ok) == 0:
        return 0.5
    best = ok[int(np.argmin(fpr[ok]))]
    return float(np.clip(thr[best], 1e-6, 1 - 1e-6))


def threshold_for_specificity(y_true, y_prob, target: float = 0.99) -> float:
    """Наибольший порог со специфичностью >= target — граница «уверенно опухоль»."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    ok = np.where((1 - fpr) >= target)[0]
    if len(ok) == 0:
        return 0.5
    best = ok[int(np.argmax(tpr[ok]))]
    return float(np.clip(thr[best], 1e-6, 1 - 1e-6))


def risk_zones(y_prob, low: float, high: float) -> np.ndarray:
    """Три зоны вместо бинарного ответа: норма / серая зона / высокий риск."""
    p = np.asarray(y_prob, dtype=float)
    zones = np.full(len(p), "borderline", dtype=object)
    zones[p < low] = "low"
    zones[p >= high] = "high"
    return zones


def sensitivity_by_stage(meta: pd.DataFrame, y_true, y_prob, threshold: float) -> pd.DataFrame:
    """Ключевая таблица проекта: ловим ли мы стадию I так же, как стадию IV."""
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    rows = []
    stages = meta["stage"].to_numpy()

    for stage in ["Normal", "I", "II", "III", "IV", "Unknown"]:
        m = stages == stage
        if not m.any():
            continue
        hit = (y_pred[m] == 0) if stage == "Normal" else (y_pred[m] == 1)
        rows.append({
            "stage": stage,
            "n": int(m.sum()),
            "detected": int(hit.sum()),
            "rate": float(hit.mean()),
            "metric": "специфичность" if stage == "Normal" else "чувствительность",
            "mean_prob": float(y_prob[m].mean()),
        })
    return pd.DataFrame(rows)


def selected_gene_scores(model: Pipeline, gene_names: list[str]) -> pd.DataFrame:
    """Гены, отобранные пайплайном, с F-статистикой и важностью классификатора."""
    kept = np.asarray(gene_names)[model.named_steps["varfilter"].get_support()]

    sel = model.named_steps["select"]
    mask = sel.get_support()
    genes = kept[mask]
    f_scores = sel.scores_[mask]
    p_values = sel.pvalues_[mask]

    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importance = np.asarray(clf.feature_importances_, dtype=float)
    elif hasattr(clf, "coef_"):
        importance = np.abs(np.ravel(clf.coef_)).astype(float)
    else:
        importance = np.asarray([], dtype=float)

    # Ансамбль и Байес важностей не дают — тогда ранжируем по силе F-теста.
    if len(importance) != len(genes):
        importance = f_scores / (f_scores.max() or 1.0)

    return pd.DataFrame({
        "gene": genes,
        "f_score": f_scores,
        "p_value": p_values,
        "importance": importance,
    }).sort_values("importance", ascending=False).reset_index(drop=True)


def genome_wide_scores(X: np.ndarray, y: np.ndarray, gene_names: list[str]) -> pd.DataFrame:
    """ANOVA F-тест по ВСЕМ генам — сырьё для манхэттен-графика по координатам."""
    X = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
    f_scores, p_values = f_classif(X, y)
    return pd.DataFrame({
        "gene": gene_names,
        "f_score": np.nan_to_num(f_scores),
        "p_value": np.nan_to_num(p_values, nan=1.0),
    })
