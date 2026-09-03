"""Честная статистика вокруг метрик: доверительные интервалы, повторная CV,
пермутационный тест и сравнение моделей по DeLong.

Зачем этот модуль вообще появился
---------------------------------
Задача «норма vs опухоль» на TCGA-BRCA решается идеально: девять разных
классических моделей дают ROC-AUC = 1.000. Но в отложенном тесте всего
12 нормальных образцов, и специфичность 1.000 на n=12 — это на самом деле
интервал [0.76, 1.00]. Точечная единица без интервала выглядит как дефект
методики, а не как результат.

Поэтому здесь:
  * bootstrap_ci        — 95% ДИ для любой метрики (стратифицированный бутстрап);
  * wilson_ci           — корректный ДИ для долей (чувствительность/специфичность);
  * repeated_group_cv   — повторная CV по пациентам вместо одного сплита,
                          чтобы КАЖДЫЙ из 61 нормального образца побывал в тесте;
  * permutation_test    — эмпирический p-value: а не случайность ли наш AUC;
  * delong_test         — значимо ли модель A лучше модели B на одной выборке;
  * learning_curve_minority — сколько нормальных образцов реально нужно.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

logger = logging.getLogger(__name__)

RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
#  Доверительные интервалы
# --------------------------------------------------------------------------- #
def wilson_ci(successes: int, total: int, conf: float = 0.95) -> tuple[float, float]:
    """ДИ Уилсона для доли. В отличие от нормального приближения корректно
    работает на краю: 12 из 12 -> [0.76, 1.00], а не вырожденное [1.00, 1.00]."""
    if total == 0:
        return (0.0, 1.0)
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def bootstrap_ci(
    y_true,
    y_prob,
    metric_fn,
    n_boot: int = 2000,
    conf: float = 0.95,
    random_state: int = RANDOM_STATE,
) -> tuple[float, float, float]:
    """Перцентильный бутстрап-ДИ. Ресэмплим ВНУТРИ классов: иначе в маленьких
    выборках попадаются реплики без единого нормального образца."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    rng = np.random.default_rng(random_state)

    idx_pos = np.where(y_true == 1)[0]
    idx_neg = np.where(y_true == 0)[0]
    point = float(metric_fn(y_true, y_prob))

    if len(idx_pos) == 0 or len(idx_neg) == 0:
        return point, float("nan"), float("nan")

    vals = []
    for _ in range(n_boot):
        take = np.concatenate([
            rng.choice(idx_pos, size=len(idx_pos), replace=True),
            rng.choice(idx_neg, size=len(idx_neg), replace=True),
        ])
        try:
            vals.append(float(metric_fn(y_true[take], y_prob[take])))
        except ValueError:
            continue

    if not vals:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(vals, [(1 - conf) / 2 * 100, (1 + conf) / 2 * 100])
    return point, float(lo), float(hi)


def metrics_with_ci(y_true, y_prob, threshold: float, n_boot: int = 2000) -> dict:
    """Метрики + интервалы. Для долей берём Уилсона (точнее на малых n),
    для площадей под кривыми — бутстрап."""
    from sklearn.metrics import average_precision_score, confusion_matrix

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    auc, auc_lo, auc_hi = bootstrap_ci(y_true, y_prob, roc_auc_score, n_boot)
    ap, ap_lo, ap_hi = bootstrap_ci(y_true, y_prob, average_precision_score, n_boot)
    sens_lo, sens_hi = wilson_ci(int(tp), int(tp + fn))
    spec_lo, spec_hi = wilson_ci(int(tn), int(tn + fp))

    return {
        "roc_auc": auc, "roc_auc_ci": [auc_lo, auc_hi],
        "pr_auc": ap, "pr_auc_ci": [ap_lo, ap_hi],
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "sensitivity_ci": [sens_lo, sens_hi],
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "specificity_ci": [spec_lo, spec_hi],
        "n_pos": int(tp + fn), "n_neg": int(tn + fp),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "threshold": float(threshold),
    }


# --------------------------------------------------------------------------- #
#  Повторная кросс-валидация по пациентам
# --------------------------------------------------------------------------- #
def repeated_group_cv(
    model,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 10,
    random_state: int = RANDOM_STATE,
) -> dict:
    """N повторов K-fold со сменой сида.

    Один отложенный тест на 118 образцов — это одна реализация случайной
    величины. 10x5 фолдов дают 50 оценок AUC и распределение вместо точки,
    причём каждый образец выборки успевает побывать в валидации 10 раз.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    fold_aucs: list[float] = []
    oof_matrix = np.full((n_repeats, len(y)), np.nan)

    for r in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                  random_state=random_state + r)
        for tr, va in cv.split(X, y, groups=groups):
            est = clone(model)
            est.fit(X[tr], y[tr])
            p = est.predict_proba(X[va])[:, 1]
            oof_matrix[r, va] = p
            if len(np.unique(y[va])) > 1:
                fold_aucs.append(float(roc_auc_score(y[va], p)))

    fold_aucs_arr = np.asarray(fold_aucs, dtype=float)
    # Средняя вероятность по повторам устойчивее одиночного OOF.
    oof_mean = np.nanmean(oof_matrix, axis=0)

    return {
        "oof_mean": oof_mean,
        "oof_matrix": oof_matrix,
        "fold_aucs": fold_aucs_arr,
        "auc_mean": float(fold_aucs_arr.mean()),
        "auc_std": float(fold_aucs_arr.std()),
        "auc_p2_5": float(np.percentile(fold_aucs_arr, 2.5)),
        "auc_p97_5": float(np.percentile(fold_aucs_arr, 97.5)),
        "auc_min": float(fold_aucs_arr.min()),
        "n_estimates": int(len(fold_aucs_arr)),
    }


# --------------------------------------------------------------------------- #
#  Пермутационный тест
# --------------------------------------------------------------------------- #
def permutation_test(
    model,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    observed_auc: float,
    n_perm: int = 200,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Насколько наш AUC отличается от того, что даёт та же схема на шуме.

    Метки перемешиваются НА УРОВНЕ ПАЦИЕНТОВ: иначе у пациента с парой
    норма+опухоль перестановка внутри пары ломает групповую структуру и
    нулевое распределение получается нечестно узким.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)

    # Метка пациента = метка его первого образца; перемешиваем по пациентам.
    uniq_groups, first_idx = np.unique(groups, return_index=True)
    group_labels = y[first_idx]

    null_aucs: list[float] = []
    for _ in range(n_perm):
        permuted = rng.permutation(group_labels)
        mapping = dict(zip(uniq_groups, permuted))
        y_perm = np.array([mapping[g] for g in groups], dtype=int)
        if len(np.unique(y_perm)) < 2:
            continue

        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                  random_state=int(rng.integers(1_000_000)))
        aucs = []
        for tr, va in cv.split(X, y_perm, groups=groups):
            if len(np.unique(y_perm[va])) < 2 or len(np.unique(y_perm[tr])) < 2:
                continue
            est = clone(model)
            est.fit(X[tr], y_perm[tr])
            aucs.append(roc_auc_score(y_perm[va], est.predict_proba(X[va])[:, 1]))
        if aucs:
            null_aucs.append(float(np.mean(aucs)))

    null = np.asarray(null_aucs, dtype=float)
    # +1 в числителе и знаменателе — поправка Дэвисона-Хинкли, чтобы p-value
    # никогда не оказался ровно нулём (мы сделали конечное число перестановок).
    p_value = float((np.sum(null >= observed_auc) + 1) / (len(null) + 1))

    return {
        "observed_auc": float(observed_auc),
        "null_auc_mean": float(null.mean()) if len(null) else float("nan"),
        "null_auc_std": float(null.std()) if len(null) else float("nan"),
        "null_auc_p95": float(np.percentile(null, 95)) if len(null) else float("nan"),
        "null_aucs": null.tolist(),
        "n_permutations": int(len(null)),
        "p_value": p_value,
    }


# --------------------------------------------------------------------------- #
#  DeLong: значимо ли одна модель лучше другой
# --------------------------------------------------------------------------- #
def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def delong_test(y_true, prob_a, prob_b) -> dict:
    """Тест DeLong для двух ROC-AUC на ОДНОЙ выборке.

    Фраза «у модели A AUC выше» ничего не значит, пока не учтено, что обе
    модели оценены на одних и тех же образцах и их ошибки коррелированы.
    """
    y_true = np.asarray(y_true)
    preds = np.vstack([np.asarray(prob_a, dtype=float),
                       np.asarray(prob_b, dtype=float)])

    pos = preds[:, y_true == 1]
    neg = preds[:, y_true == 0]
    m, n = pos.shape[1], neg.shape[1]
    if m < 2 or n < 2:
        return {"auc_a": float("nan"), "auc_b": float("nan"),
                "delta": float("nan"), "z": float("nan"), "p_value": float("nan")}

    k = 2
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = _midrank(pos[r])
        ty[r] = _midrank(neg[r])
        tz[r] = _midrank(np.concatenate([pos[r], neg[r]]))

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n

    contrast = np.array([[1.0, -1.0]])
    var = float(np.squeeze(contrast @ cov @ contrast.T))
    diff = float(aucs[0] - aucs[1])
    if var <= 0:
        return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]),
                "delta": diff, "z": float("nan"), "p_value": 1.0}

    z = diff / np.sqrt(var)
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]),
            "delta": diff, "z": float(z), "p_value": p}


# --------------------------------------------------------------------------- #
#  Кривая обучения по редкому классу
# --------------------------------------------------------------------------- #
def learning_curve_minority(
    model,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    sizes=(5, 10, 15, 20, 30, 40, 48),
    n_repeats: int = 5,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """AUC как функция числа образцов РЕДКОГО класса в обучении.

    Практический вопрос: 61 нормальный образец — это много или мало?
    Если кривая выходит на плато уже на 15 нормах, значит запас прочности
    есть и результат не держится на нескольких счастливых образцах.
    """
    X = np.asarray(X); y = np.asarray(y)
    minority = 0 if (y == 0).sum() <= (y == 1).sum() else 1
    rows = []

    for n_min in sizes:
        aucs = []
        for r in range(n_repeats):
            rng = np.random.default_rng(random_state + r)
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                      random_state=random_state + r)
            for tr, va in cv.split(X, y, groups=groups):
                pool = tr[y[tr] == minority]
                if len(pool) < n_min or len(np.unique(y[va])) < 2:
                    continue
                keep_min = rng.choice(pool, size=n_min, replace=False)
                tr_sub = np.concatenate([tr[y[tr] != minority], keep_min])
                est = clone(model)
                est.fit(X[tr_sub], y[tr_sub])
                aucs.append(roc_auc_score(y[va], est.predict_proba(X[va])[:, 1]))
        if aucs:
            rows.append({"n_minority": n_min, "auc_mean": float(np.mean(aucs)),
                         "auc_std": float(np.std(aucs)), "n_folds": len(aucs)})
            logger.info(f"  редкого класса в обучении {n_min:3d} -> "
                        f"AUC {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    return pd.DataFrame(rows)
