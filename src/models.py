"""Классические ML-модели для предсказаний по экспрессии генов.

Никаких нейросетей и трансформеров: логистическая регрессия, дискриминантный
анализ, PLS-DA, сжатые центроиды, SVM, леса, бустинг, наивный Байес, kNN,
их ансамбль и стекинг.

Три принципа, без которых цифры были бы враньём:
  1. Отбор генов и масштабирование — ВНУТРИ Pipeline, обучаются на train-фолде.
  2. Разбиение — по ПАЦИЕНТАМ (StratifiedGroupKFold): у части больных в наборе
     есть и опухоль, и своя же норма; случайный сплит растащил бы их по
     train/test и модель узнавала бы пациента, а не болезнь.
  3. Подбор гиперпараметров — во ВЛОЖЕННОЙ CV (nested_cv_score): если выбирать
     модель по тем же фолдам, на которых её потом оценивают, оценка завышается.

Про состав зоопарка
-------------------
Данные здесь — 17 814 генов на 590 образцов, то есть p >> n. В таком режиме
хорошо работают не «мощные» модели, а сильно регуляризованные и линейные:
LDA с усадкой Ледуа-Вольфа, PLS-DA и сжатые центроиды (PAM) — это рабочие
лошадки омиксного анализа, и их здесь не хватало.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
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
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
#  Классификаторы, которых нет в sklearn «из коробки»
# --------------------------------------------------------------------------- #
class PLSDA(BaseEstimator, ClassifierMixin):
    """PLS-DA: регрессия частных наименьших квадратов на метку 0/1 + сигмоида.

    Стандартный метод хемометрики и омиксов. В отличие от PCA ищет компоненты,
    максимально КОВАРИИРУЮЩИЕ с меткой, поэтому на p >> n сжимает 17 тысяч
    генов в десяток осмысленных осей и почти не переобучается.
    """

    def __init__(self, n_components: int = 10):
        self.n_components = n_components

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n = int(min(self.n_components, X.shape[1], max(1, X.shape[0] - 1)))
        self.pls_ = PLSRegression(n_components=n, scale=False).fit(X, y.astype(float))
        scores = self.pls_.predict(X).ravel()
        self.mu_ = float(scores.mean())
        self.sd_ = float(scores.std()) or 1.0
        return self

    def decision_function(self, X):
        return (self.pls_.predict(np.asarray(X, dtype=float)).ravel() - self.mu_) / self.sd_

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-np.clip(self.decision_function(X), -30, 30)))
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class ShrunkenCentroids(BaseEstimator, ClassifierMixin):
    """Сжатые центроиды (PAM, Tibshirani 2002) — канонический классификатор
    микрочипов.

    Считает средний профиль каждого класса и «сжимает» его к общему среднему
    на величину delta. Гены, у которых разница классов меньше сжатия,
    обнуляются и выпадают из модели — отбор признаков получается бесплатно,
    вместе с обучением, а результат читается как готовая генная панель.
    """

    def __init__(self, shrinkage: float = 2.0):
        self.shrinkage = shrinkage

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n, p = X.shape

        overall = X.mean(axis=0)
        # Объединённое внутриклассовое СКО + s0 (медиана) для устойчивости
        # генов с почти нулевой дисперсией — как в оригинальном PAM.
        pooled = np.zeros(p)
        for c in self.classes_:
            Xc = X[y == c]
            pooled += ((Xc - Xc.mean(axis=0)) ** 2).sum(axis=0)
        pooled = np.sqrt(pooled / max(1, n - len(self.classes_)))
        self.s0_ = float(np.median(pooled))
        s = pooled + self.s0_

        self.centroids_ = {}
        for c in self.classes_:
            Xc = X[y == c]
            nc = len(Xc)
            m = np.sqrt(1.0 / nc - 1.0 / n) if nc < n else np.sqrt(1.0 / nc)
            d = (Xc.mean(axis=0) - overall) / (m * s)
            d_shrunk = np.sign(d) * np.maximum(np.abs(d) - self.shrinkage, 0.0)
            self.centroids_[c] = overall + m * s * d_shrunk

        self.overall_ = overall
        self.s_ = s
        self.n_genes_kept_ = int(np.sum(
            np.any([self.centroids_[c] != overall for c in self.classes_], axis=0)))
        self.priors_ = {c: float((y == c).mean()) for c in self.classes_}
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        d = []
        for c in self.classes_:
            dist = (((X - self.centroids_[c]) / self.s_) ** 2).sum(axis=1)
            d.append(-0.5 * dist + np.log(max(self.priors_[c], 1e-9)))
        d = np.array(d).T
        return d[:, 1] - d[:, 0]

    def predict_proba(self, X):
        z = np.clip(self.decision_function(X), -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# --------------------------------------------------------------------------- #
#  Пайплайны
# --------------------------------------------------------------------------- #
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
    # Начиная с scikit-learn 1.8 аргумент penalty объявлен устаревшим:
    # тип регуляризации задаётся через l1_ratio (0 = L2, 1 = L1, между —
    # эластичная сеть). Пишем сразу в новом виде, иначе лог тонет в
    # предупреждениях, а в 1.10 код просто перестанет работать.
    base = {
        "Logistic Regression (L2)": LogisticRegression(
            C=1.0, l1_ratio=0.0, max_iter=5000,
            class_weight="balanced", random_state=RANDOM_STATE),
        "Logistic Regression (L1)": LogisticRegression(
            C=0.1, l1_ratio=1.0, solver="liblinear", max_iter=5000,
            class_weight="balanced", random_state=RANDOM_STATE),
        "Elastic Net": LogisticRegression(
            C=0.05, l1_ratio=0.5, solver="saga", max_iter=3000,
            class_weight="balanced", random_state=RANDOM_STATE),
        # p >> n: усадка ковариации по Ледуа-Вольфу вместо её оценки «в лоб».
        "LDA (Ledoit-Wolf)": LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto"),
        "PLS-DA": PLSDA(n_components=10),
        "Shrunken Centroids (PAM)": ShrunkenCentroids(shrinkage=2.0),
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

    models = {name: Pipeline(_front(n_genes) + [("clf", clf)])
              for name, clf in base.items()}

    # Мягкое голосование разнотипных моделей: линейная + ядровая + деревья.
    models["Ensemble (soft vote)"] = Pipeline(_front(n_genes) + [
        ("clf", VotingClassifier(
            estimators=[
                ("lr", clone(base["Logistic Regression (L2)"])),
                ("svm", clone(base["SVM (RBF)"])),
                ("rf", clone(base["Random Forest"])),
            ],
            voting="soft",
        ))
    ])

    # Стекинг: мета-логрегрессия учится, КОМУ из базовых моделей верить,
    # а не усредняет их вслепую. На трудных задачах обычно выигрывает у
    # голосования, потому что умеет давать разный вес разным моделям.
    models["Stacking (LR meta)"] = Pipeline(_front(n_genes) + [
        ("clf", StackingClassifier(
            estimators=[
                ("lr", LogisticRegression(C=0.01, max_iter=5000,
                                          class_weight="balanced",
                                          random_state=RANDOM_STATE)),
                ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
                ("rf", RandomForestClassifier(
                    n_estimators=400, min_samples_leaf=3, n_jobs=-1,
                    class_weight="balanced_subsample", random_state=RANDOM_STATE)),
                ("pls", PLSDA(n_components=10)),
            ],
            final_estimator=LogisticRegression(C=1.0, max_iter=2000),
            cv=5, n_jobs=1,
        ))
    ])
    return models


# --------------------------------------------------------------------------- #
#  Пространство поиска гиперпараметров
# --------------------------------------------------------------------------- #
def search_space(n_features: int) -> list[dict]:
    """Сетка для подбора: и число генов, и сила регуляризации, и сам классификатор.

    Число отбираемых генов k — такой же гиперпараметр, как C. На насыщенной
    задаче хватает 5 генов, на статусе лимфоузлов оптимум лежит в сотнях;
    зашивать k = 300 константой значит заранее проиграть на части задач.
    """
    ks = [k for k in (50, 300, 1000, 3000) if k <= n_features] or [n_features]
    # n_jobs=1 у леса намеренно: параллелится сам GridSearchCV, и вложенная
    # параллельность вместо ускорения даёт борьбу за ядра.
    return [
        {"select__k": ks,
         "clf": [LogisticRegression(max_iter=5000, class_weight="balanced",
                                    random_state=RANDOM_STATE)],
         "clf__C": [0.003, 0.03, 1.0]},
        {"select__k": ks,
         "clf": [LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")]},
        {"select__k": ks,
         "clf": [PLSDA()],
         "clf__n_components": [3, 10]},
        {"select__k": ks,
         "clf": [ShrunkenCentroids()],
         "clf__shrinkage": [0.5, 2.0]},
        {"select__k": ks,
         "clf": [RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                        n_jobs=1, class_weight="balanced_subsample",
                                        random_state=RANDOM_STATE)],
         "clf__max_features": ["sqrt", 0.05]},
    ]


def tuned_pipeline(n_features: int, inner_splits: int = 4,
                   n_jobs: int = 1) -> GridSearchCV:
    """Пайплайн, который сам подбирает себе k, тип модели и регуляризацию.

    Внутренняя CV — тоже по пациентам: иначе подбор гиперпараметров подглядит
    в те же образцы, на которых потом меряется качество.
    """
    pipe = Pipeline(_front(300) + [("clf", LogisticRegression(max_iter=5000))])
    return GridSearchCV(
        pipe,
        param_grid=search_space(n_features),
        scoring="roc_auc",
        cv=StratifiedGroupKFold(n_splits=inner_splits, shuffle=True,
                                random_state=RANDOM_STATE),
        n_jobs=n_jobs,
        refit=True,
        error_score=np.nan,
    )


def nested_cv_score(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    outer_splits: int = 5,
    inner_splits: int = 4,
    n_jobs: int = 1,
) -> dict:
    """Вложенная CV: внешние фолды меряют, внутренние — подбирают.

    Это единственная схема, которая честно отвечает на вопрос «сколько даст
    ВСЯ процедура на новых данных», включая стоимость подбора гиперпараметров.
    Плоская CV с подбором на тех же фолдах систематически завышает AUC.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    outer = StratifiedGroupKFold(n_splits=outer_splits, shuffle=True,
                                 random_state=RANDOM_STATE)

    oof = np.full(len(y), np.nan)
    oof_rank = np.full(len(y), np.nan)
    fold_aucs, chosen = [], []

    for fold, (tr, va) in enumerate(outer.split(X, y, groups=groups), 1):
        search = tuned_pipeline(X.shape[1], inner_splits=inner_splits, n_jobs=n_jobs)
        search.fit(X[tr], y[tr], groups=groups[tr])

        p = search.predict_proba(X[va])[:, 1]
        oof[va] = p
        # Внутри разных фолдов побеждают РАЗНЫЕ модели, и их вероятности живут
        # в разных шкалах: 0.7 от PLS-DA и 0.7 от сжатых центроидов — величины
        # несопоставимые. Если сложить такие оценки в один вектор и посчитать
        # общий AUC, он окажется заметно НИЖЕ среднего по фолдам просто из-за
        # рассогласования шкал. Поэтому в общий пул кладём ранги внутри фолда:
        # порядок сохраняется, шкала становится единой.
        oof_rank[va] = (pd.Series(p).rank(method="average").to_numpy()
                        / (len(p) + 1.0))
        auc = float(roc_auc_score(y[va], p)) if len(np.unique(y[va])) > 1 else np.nan
        fold_aucs.append(auc)

        best = search.best_params_
        label = f"{type(best['clf']).__name__} k={best.get('select__k')}"
        chosen.append(label)
        logger.info(f"    фолд {fold}: AUC {auc:.4f}  <- подобрано: {label}")

    aucs = np.asarray([a for a in fold_aucs if not np.isnan(a)], dtype=float)
    return {
        "oof": oof,
        "oof_rank": oof_rank,
        "fold_aucs": aucs.tolist(),
        "auc_mean": float(aucs.mean()) if len(aucs) else np.nan,
        "auc_std": float(aucs.std()) if len(aucs) else np.nan,
        "chosen_per_fold": chosen,
    }


# --------------------------------------------------------------------------- #
#  Метрики
# --------------------------------------------------------------------------- #
def compute_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    """Полный набор метрик. sensitivity = доля пойманных положительных случаев."""
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
        est = clone(model)
        est.fit(X[train_idx], y[train_idx])
        probs = est.predict_proba(X[val_idx])[:, 1]
        oof[val_idx] = probs
        if len(np.unique(y[val_idx])) > 1:
            fold_aucs.append(float(roc_auc_score(y[val_idx], probs)))

    return oof, fold_aucs


# --------------------------------------------------------------------------- #
#  Пороги и зоны риска
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
#  Гены: отбор, важности, устойчивость
# --------------------------------------------------------------------------- #
def selected_gene_scores(model: Pipeline, gene_names: list[str]) -> pd.DataFrame:
    """Гены, отобранные пайплайном, с F-статистикой и важностью классификатора."""
    kept = np.asarray(gene_names)[model.named_steps["varfilter"].get_support()]

    sel = model.named_steps["select"]
    mask = sel.get_support()
    genes = kept[mask]
    f_scores = sel.scores_[mask]
    p_values = sel.pvalues_[mask]

    clf = model.named_steps["clf"]
    importance = np.asarray([], dtype=float)
    if hasattr(clf, "feature_importances_"):
        importance = np.asarray(clf.feature_importances_, dtype=float)
    elif hasattr(clf, "coef_"):
        importance = np.abs(np.ravel(clf.coef_)).astype(float)
    elif isinstance(clf, PLSDA) and hasattr(clf, "pls_"):
        importance = np.abs(np.ravel(clf.pls_.coef_)).astype(float)
    elif isinstance(clf, ShrunkenCentroids) and hasattr(clf, "centroids_"):
        c = clf.centroids_
        importance = np.abs(c[clf.classes_[1]] - c[clf.classes_[0]]).astype(float)

    # Ансамбль и Байес важностей не дают — тогда ранжируем по силе F-теста.
    if len(importance) != len(genes):
        importance = f_scores / (f_scores.max() or 1.0)

    return pd.DataFrame({
        "gene": genes,
        "f_score": f_scores,
        "p_value": p_values,
        "importance": importance,
    }).sort_values("importance", ascending=False).reset_index(drop=True)


def stability_selection(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    gene_names: list[str],
    k: int = 50,
    n_boot: int = 200,
    subsample: float = 0.75,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Как часто ген попадает в топ-k, если пересобирать выборку.

    Список генов, полученный ОДНОЙ подгонкой на всех данных, невоспроизводим:
    выкиньте десяток пациентов — и половина списка поменяется. Здесь выборка
    пересобирается n_boot раз ПО ПАЦИЕНТАМ, и для каждого гена считается доля
    повторов, в которых он вошёл в топ-k. Частота 1.00 означает «ген выбирается
    всегда», 0.30 — «попал в отчёт случайно».
    """
    X = np.asarray(X)
    y = np.asarray(y)
    genes = np.asarray(gene_names)
    rng = np.random.default_rng(random_state)

    uniq = np.unique(groups)
    n_take = max(2, int(len(uniq) * subsample))
    counts = np.zeros(len(genes), dtype=int)
    sign_sum = np.zeros(len(genes), dtype=float)
    n_valid = 0

    for _ in range(n_boot):
        take_groups = set(rng.choice(uniq, size=n_take, replace=False))
        m = np.array([g in take_groups for g in groups])
        if len(np.unique(y[m])) < 2:
            continue

        Xs = X[m]
        Xs = np.where(np.isnan(Xs), np.nanmedian(Xs, axis=0), Xs)
        with np.errstate(invalid="ignore", divide="ignore"):
            f, _ = f_classif(Xs, y[m])
        f = np.nan_to_num(f)

        top = np.argsort(f)[::-1][:k]
        counts[top] += 1
        # Направление изменения: выше или ниже в положительном классе.
        delta = np.nanmean(Xs[y[m] == 1], axis=0) - np.nanmean(Xs[y[m] == 0], axis=0)
        sign_sum[top] += np.sign(delta[top])
        n_valid += 1

    freq = counts / max(1, n_valid)
    order = np.argsort(freq)[::-1]
    return pd.DataFrame({
        "gene": genes[order],
        "selection_freq": freq[order],
        "direction_consistency": np.divide(
            np.abs(sign_sum[order]), np.maximum(counts[order], 1)),
        "times_selected": counts[order],
        "n_bootstraps": n_valid,
    })


def genome_wide_scores(X: np.ndarray, y: np.ndarray, gene_names: list[str]) -> pd.DataFrame:
    """ANOVA F-тест по ВСЕМ генам — сырьё для манхэттен-графика по координатам."""
    X = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
    with np.errstate(invalid="ignore", divide="ignore"):
        f_scores, p_values = f_classif(X, y)
    return pd.DataFrame({
        "gene": gene_names,
        "f_score": np.nan_to_num(f_scores),
        "p_value": np.nan_to_num(p_values, nan=1.0),
    })
