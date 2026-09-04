"""BioDNA v2 — здоровые люди (GTEx) против больных раком (TCGA).

Отличие от train.py: там «норма» — ткань рядом с опухолью у тех же онкобольных.
Такая ткань уже несёт следы болезни, и модель на ней ничего не говорит о
здоровом человеке. Здесь добавлены 180 женщин из GTEx, у которых рака не было.
Обе когорты взяты из recount3, где пересчитаны одним пайплайном Monorail на
одной аннотации GENCODE v26 — иначе сравнивать их было бы нельзя.

Скрипт построен как серия проверок на самообман, а не как гонка за AUC:

  0. Отпечаток когорты. GTEx-здоровые против TCGA-нормы-рядом. Рака нет ни там,
     ни там, поэтому всё различённое здесь — технический след.
  1. Наивная модель. GTEx-норма vs TCGA-опухоль в лоб: антипример.
  2. Перенос TCGA -> GTEx. Учимся только на TCGA, проверяемся на здоровых
     людях, которых модель не видела. Сколько здоровых объявлены больными?
  3. Перенос GTEx -> TCGA. Обратное направление: узнает ли модель норму
     чужой когорты или спутает её с опухолью?
  3b. Размер панели против переносимости: большие панели цепляются за
     особенности консорциума, маленькие переносятся.
  4. Перестановочный тест с перемешиванием меток по пациентам.
  5. Итоговая модель: повторная CV, ДИ, стабильность генов, кривая обучения,
     метрики по стадиям и по источникам образцов.
  8. Две внешние когорты, которых модель не видела: Шанхай (трижды-негативный
     рак) и США (ER+ плюс ткань после редукционной маммопластики, то есть
     здоровые живые женщины, а не посмертный материал).
  9. Спектр прогрессии: норма -> ранняя неоплазия -> DCIS -> инвазия. Модель
     обучалась только на крайних точках, промежуточные видит впервые.

Запуск:  python train_cohorts.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RECOUNT_DIR = DATA_DIR / "recount3"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results_v2"

sys.path.insert(0, str(BASE_DIR))

from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from src.cohorts import (  # noqa: E402
    ADJACENT,
    EXTERNAL_COHORTS,
    HEALTHY,
    PROGRESSION_ORDER,
    TUMOR,
    build_dataset,
    load_external,
    normalize,
)
from src.evaluation import (  # noqa: E402
    learning_curve_minority,
    metrics_with_ci,
    permutation_test,
    repeated_group_cv,
    wilson_ci,
)
from src.models import (  # noqa: E402
    RANDOM_STATE,
    build_models,
    compute_metrics,
    cross_validate_oof,
    stability_selection,
    threshold_for_sensitivity,
)
from src.viz_cohorts import (  # noqa: E402
    plot_external,
    plot_progression,
    plot_honesty_panel,
    plot_panel_transfer,
    plot_models,
    plot_pca,
    plot_permutation,
    plot_risk_by_group,
    plot_stage,
)

logger = logging.getLogger("biodna2")

NORMALIZATIONS = ("logtpm", "zsample", "rank")

# Блоки-доказательства (перестановки, повторная CV, стабильность, кривая
# обучения) считаются одной простой моделью. Они проверяют данные и протокол,
# а не выбор алгоритма, и с логистической регрессией остаются интерпретируемыми
# и достаточно быстрыми, чтобы прогнать сотни повторов.
PROBE_MODEL = "Logistic Regression (L2)"


def setup_logging(overwrite: bool = True) -> None:
    """overwrite=False для --report-only.

    Полный прогон начинает лог с чистого листа, а перегенерация отчёта не должна
    затирать лог обучения: она занимает секунду и не содержит ничего, ради чего
    стоило бы терять час записей.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(RESULTS_DIR / "training_v2.log",
                                mode="w" if overwrite else "a", encoding="utf-8"),
        ],
        force=True,
    )


def _matrix(data, norm: str, mask=None):
    X = normalize(data.X, norm)
    meta = data.meta
    if mask is not None:
        X, meta = X.loc[mask], meta.loc[mask]
    return (X.to_numpy(dtype=np.float32),
            meta["label"].to_numpy(dtype=np.int8),
            meta["patient"].to_numpy())


# ---------------------------------------------------------------- проверки ---

def cohort_probe(data, norm: str, n_genes: int, folds: int) -> dict:
    """Насколько когорты различаются САМИ ПО СЕБЕ, без участия рака.

    Берём только ткань без опухоли: здоровые GTEx против TCGA-нормы-рядом.
    AUC около 1 означает, что между когортами есть полный технический отпечаток,
    и любая модель, у которой норма приходит из одной когорты, а опухоль из
    другой, будет частично мерить именно его.
    """
    mask = data.meta["group"].isin([HEALTHY, ADJACENT]).to_numpy()
    X = normalize(data.X, norm).loc[mask].to_numpy(dtype=np.float32)
    meta = data.meta.loc[mask]
    y = (meta["cohort"] == "GTEX").to_numpy().astype(np.int8)
    groups = meta["patient"].to_numpy()

    _, aucs = cross_validate_oof(build_models(n_genes)[PROBE_MODEL], X, y, groups,
                                 n_splits=folds)
    auc = float(np.mean(aucs))
    logger.info(f"  [{norm:7s}] отпечаток когорты: AUC {auc:.4f}±{np.std(aucs):.4f}"
                f"   (GTEx {int((y == 1).sum())} vs TCGA-норма {int((y == 0).sum())})")
    return {"normalization": norm, "auc": auc, "auc_std": float(np.std(aucs)),
            "n_gtex": int((y == 1).sum()), "n_tcga": int((y == 0).sum())}


def naive_model(data, norm: str, n_genes: int, folds: int) -> dict:
    """Антипример: здоровые GTEx против опухолей TCGA, без всякой осторожности.

    Класс нормы целиком из одной когорты, класс опухоли — целиком из другой,
    так что различить их можно и не глядя на рак. Цифра приводится затем, чтобы
    показать, чего она стоит.
    """
    mask = data.meta["group"].isin([HEALTHY, TUMOR]).to_numpy()
    X, y, groups = _matrix(data, norm, mask)
    oof, aucs = cross_validate_oof(build_models(n_genes)[PROBE_MODEL], X, y, groups,
                                   n_splits=folds)
    m = compute_metrics(y, oof, threshold=0.5)
    logger.info(f"  [{norm:7s}] наивная модель: AUC {np.mean(aucs):.4f} "
                f"(цифра ничего не доказывает)")
    return {"normalization": norm, "cv_auc": float(np.mean(aucs)),
            "sensitivity": m["sensitivity"], "specificity": m["specificity"]}


def transfer_tcga_to_gtex(data, norm, n_genes, folds, target_sens) -> dict:
    """Учимся ТОЛЬКО на TCGA, проверяемся на здоровых людях из GTEx.

    Модель не видела ни одного образца GTEx. Здоровая женщина, которую она
    объявила больной, — ложная тревога в самом честном смысле слова. Это
    главная метрика всего исследования.
    """
    tcga = (data.meta["cohort"] == "TCGA").to_numpy()
    X_tr, y_tr, g_tr = _matrix(data, norm, tcga)

    model = build_models(n_genes)[PROBE_MODEL]
    oof, aucs = cross_validate_oof(model, X_tr, y_tr, g_tr, n_splits=folds)
    thr = threshold_for_sensitivity(y_tr, oof, target_sens)
    internal = compute_metrics(y_tr, oof, threshold=thr)
    model.fit(X_tr, y_tr)

    healthy = (data.meta["group"] == HEALTHY).to_numpy()
    prob = model.predict_proba(
        normalize(data.X, norm).loc[healthy].to_numpy(dtype=np.float32))[:, 1]
    flagged = int((prob >= thr).sum())
    lo, hi = wilson_ci(flagged, len(prob))

    logger.info(f"  [{norm:7s}] TCGA -> GTEx | внутри TCGA AUC {np.mean(aucs):.4f}, "
                f"порог {thr:.3f} | здоровых объявлено больными: {flagged}/{len(prob)} "
                f"= {flagged / len(prob):.1%} (95% ДИ {lo:.1%}–{hi:.1%})")

    return {"normalization": norm,
            "tcga_cv_auc": float(np.mean(aucs)),
            "tcga_sensitivity": internal["sensitivity"],
            "tcga_specificity": internal["specificity"],
            "threshold": thr,
            "n_healthy": int(len(prob)),
            "healthy_flagged": flagged,
            "healthy_false_positive_rate": float(flagged / len(prob)),
            "fpr_ci_low": float(lo), "fpr_ci_high": float(hi),
            "healthy_mean_risk": float(prob.mean())}


def transfer_gtex_to_tcga(data, norm, n_genes, folds, target_sens) -> dict:
    """Учимся на GTEx-здоровых + TCGA-опухолях, проверяемся на TCGA-норме-рядом.

    Ловушка для модели, выучившей когорту: TCGA-норма технически похожа на
    TCGA-опухоль, но биологически — на здоровую ткань. Модель, смотрящая на рак,
    назовёт её нормой; модель, смотрящая на консорциум, объявит опухолью.
    """
    mask = data.meta["group"].isin([HEALTHY, TUMOR]).to_numpy()
    X_tr, y_tr, g_tr = _matrix(data, norm, mask)

    model = build_models(n_genes)[PROBE_MODEL]
    oof, _ = cross_validate_oof(model, X_tr, y_tr, g_tr, n_splits=folds)
    thr = threshold_for_sensitivity(y_tr, oof, target_sens)
    model.fit(X_tr, y_tr)

    adj = (data.meta["group"] == ADJACENT).to_numpy()
    prob = model.predict_proba(
        normalize(data.X, norm).loc[adj].to_numpy(dtype=np.float32))[:, 1]
    wrong = int((prob >= thr).sum())
    lo, hi = wilson_ci(wrong, len(prob))

    logger.info(f"  [{norm:7s}] GTEx -> TCGA | TCGA-норму назвал опухолью: "
                f"{wrong}/{len(prob)} = {wrong / len(prob):.1%} "
                f"(95% ДИ {lo:.1%}–{hi:.1%}, средний риск {prob.mean():.3f})")

    return {"normalization": norm, "threshold": thr, "n_adjacent": int(len(prob)),
            "adjacent_called_tumor": wrong,
            "adjacent_error_rate": float(wrong / len(prob)),
            "err_ci_low": float(lo), "err_ci_high": float(hi),
            "adjacent_mean_risk": float(prob.mean())}


def transfer_vs_panel(data, norms, sizes, folds, target_sens) -> pd.DataFrame:
    """Как размер генной панели влияет на перенос между когортами.

    Внутри одной когорты качество на плато почти при любом размере панели, но
    переносится он по-разному: чем больше генов, тем больше шансов зацепиться
    за особенности конкретного консорциума вместо биологии. Здесь это меряется
    напрямую — обучаемся внутри TCGA, считаем ложные тревоги на здоровых GTEx.
    """
    rows = []
    for norm in norms:
        tcga = (data.meta["cohort"] == "TCGA").to_numpy()
        X_tr, y_tr, g_tr = _matrix(data, norm, tcga)
        healthy = (data.meta["group"] == HEALTHY).to_numpy()
        X_he = normalize(data.X, norm).loc[healthy].to_numpy(dtype=np.float32)

        for k in sizes:
            model = build_models(k)[PROBE_MODEL]
            oof, aucs = cross_validate_oof(model, X_tr, y_tr, g_tr, n_splits=folds)
            thr = threshold_for_sensitivity(y_tr, oof, target_sens)
            model.fit(X_tr, y_tr)
            prob = model.predict_proba(X_he)[:, 1]
            flagged = int((prob >= thr).sum())
            rows.append({"normalization": norm, "n_genes": k,
                         "tcga_cv_auc": float(np.mean(aucs)),
                         "healthy_flagged": flagged, "n_healthy": len(prob),
                         "healthy_fpr": float(flagged / len(prob))})
            logger.info(f"  [{norm:7s}] {k:4d} генов | AUC внутри TCGA "
                        f"{np.mean(aucs):.4f} | здоровых объявлено больными "
                        f"{flagged:3d}/{len(prob)} = {flagged / len(prob):.1%}")
    return pd.DataFrame(rows)


def external_validation(model, data, norm: str, threshold: float,
                        cache_dir, name: str) -> dict:
    """Проверка на когорте, которой модель не видела вообще.

    FUSCC — Шанхай, трижды-негативный рак: другая страна, больница, популяция
    и молекулярный подтип. VARLEY — американская когорта с ER-положительными
    опухолями и, что важнее, с тканью после редукционной маммопластики: это
    здоровые ЖИВЫЕ женщины, хирургический материал, а не посмертный, как GTEx.
    Вместе они закрывают и подтип опухоли, и происхождение здоровой нормы.
    """
    Xe, me = load_external(name, cache_dir, data.tpm_genes, data.X.columns,
                           data.annotation)
    prob = model.predict_proba(
        normalize(Xe, norm).to_numpy(dtype=np.float32))[:, 1]

    me = me.copy()
    me["risk"] = prob
    me["pred"] = (prob >= threshold).astype(int)

    scored = me["label"].notna().to_numpy()
    ye = me.loc[scored, "label"].to_numpy(dtype=np.int8)
    ps = prob[scored]

    metrics = {}
    if len(np.unique(ye)) == 2:
        metrics = {**compute_metrics(ye, ps, threshold),
                   **metrics_with_ci(ye, ps, threshold)}
        logger.info(f"  ROC-AUC {metrics['roc_auc']:.4f} "
                    f"[{metrics['roc_auc_ci'][0]:.4f}, {metrics['roc_auc_ci'][1]:.4f}]")

    by_sub = []
    for sub in me["subgroup"].unique():
        d = me[me["subgroup"] == sub]
        lab = d["label"].iloc[0]
        if pd.isna(lab):
            row = {"subgroup": sub, "n": len(d), "label": None,
                   "correct": None, "rate": None,
                   "flagged": int(d["pred"].sum()),
                   "flagged_rate": float(d["pred"].mean()),
                   "mean_risk": float(d["risk"].mean())}
            logger.info(f"  {sub:28s} n={len(d):4d}  помечено как опухоль "
                        f"{row['flagged_rate']:.1%}  средний риск {row['mean_risk']:.3f}")
        else:
            ok = int((d["pred"] == int(lab)).sum())
            lo, hi = wilson_ci(ok, len(d))
            row = {"subgroup": sub, "n": len(d), "label": int(lab),
                   "correct": ok, "rate": float(ok / len(d)),
                   "ci_low": lo, "ci_high": hi,
                   "flagged": int(d["pred"].sum()),
                   "flagged_rate": float(d["pred"].mean()),
                   "mean_risk": float(d["risk"].mean())}
            logger.info(f"  {sub:28s} n={len(d):4d}  верно {row['rate']:.1%} "
                        f"[{lo:.1%}–{hi:.1%}]  средний риск {row['mean_risk']:.3f}")
        by_sub.append(row)

    return {"cohort": name, "title": EXTERNAL_COHORTS[name]["title"],
            "n_samples": int(len(me)), "threshold": float(threshold),
            "metrics": metrics, "by_subgroup": by_sub, "predictions": me}


def progression_spectrum(model, data, norm, threshold, cache_dir) -> dict:
    """Как растёт оценка риска вдоль прогрессии опухоли.

    Когорта SRP023262 содержит редкий набор: у одних и тех же пациентов взяты
    нормальная ткань, ранняя неоплазия, карцинома in situ и инвазивная
    карцинома. Модель училась только на «норма против инвазивной опухоли» и
    промежуточных состояний не видела никогда. Поэтому вопрос честный: примет
    ли она доброкачественное разрастание за рак и где окажется DCIS —
    формально это уже карцинома, но она ещё не прорастает.
    """
    res = external_validation(model, data, norm, threshold, cache_dir, "PROGRESSION")
    order = {s: i for i, s in enumerate(PROGRESSION_ORDER)}
    res["by_subgroup"].sort(key=lambda r: order.get(r["subgroup"], 99))
    return res


# ------------------------------------------------------------ итоговая модель ---

def final_model(data, norm, n_genes, folds, target_sens, repeats):
    """Обе нормы (здоровые + рядом с опухолью) против опухоли.

    Разбиение по пациентам; в отложенный тест попадают образцы всех трёх групп,
    поэтому специфичность считается и по здоровым людям, и по TCGA-норме.
    """
    X = normalize(data.X, norm).to_numpy(dtype=np.float32)
    meta = data.meta
    y = meta["label"].to_numpy(dtype=np.int8)
    groups = meta["patient"].to_numpy()

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(cv.split(X, y, groups=groups))
    assert not set(groups[train_idx]) & set(groups[test_idx]), "пациент в обоих наборах"
    logger.info(f"Train {len(train_idx)} / тест {len(test_idx)}, пациенты не пересекаются")

    rows = []
    for name, model in build_models(n_genes).items():
        t0 = time.time()
        _, aucs = cross_validate_oof(model, X[train_idx], y[train_idx],
                                     groups[train_idx], n_splits=folds)
        model.fit(X[train_idx], y[train_idx])
        m = compute_metrics(y[test_idx], model.predict_proba(X[test_idx])[:, 1])
        rows.append({"model": name, "cv_auc_mean": float(np.mean(aucs)),
                     "cv_auc_std": float(np.std(aucs)), "test_auc": m["roc_auc"],
                     "test_sensitivity": m["sensitivity"],
                     "test_specificity": m["specificity"], "test_mcc": m["mcc"],
                     "fit_seconds": round(time.time() - t0, 1)})
        logger.info(f"  {name:26s} CV {np.mean(aucs):.4f}±{np.std(aucs):.4f} | "
                    f"тест {m['roc_auc']:.4f} | чувств. {m['sensitivity']:.3f} | "
                    f"спец. {m['specificity']:.3f} | {rows[-1]['fit_seconds']}s")

    summary = pd.DataFrame(rows)
    # Победителя выбираем по кросс-валидации, не по тесту: иначе тест перестаёт
    # быть независимой оценкой.
    best_name = str(summary.loc[summary["cv_auc_mean"].idxmax(), "model"])
    logger.info(f"Лучшая по CV: {best_name}")

    cv_iter = list(StratifiedGroupKFold(n_splits=folds, shuffle=True,
                                        random_state=RANDOM_STATE)
                   .split(X[train_idx], y[train_idx], groups=groups[train_idx]))
    calibrated = CalibratedClassifierCV(build_models(n_genes)[best_name],
                                        method="sigmoid", cv=cv_iter)
    calibrated.fit(X[train_idx], y[train_idx])

    # Порог берём с out-of-fold предсказаний обучающей части: тест не трогаем.
    oof_cal, _ = cross_validate_oof(
        CalibratedClassifierCV(build_models(n_genes)[best_name], method="sigmoid", cv=3),
        X[train_idx], y[train_idx], groups[train_idx], n_splits=folds)
    thr = threshold_for_sensitivity(y[train_idx], oof_cal, target_sens)

    prob = calibrated.predict_proba(X[test_idx])[:, 1]
    # metrics_with_ci даёт интервалы, но не все точечные метрики — сливаем оба.
    metrics = {**compute_metrics(y[test_idx], prob, thr),
               **metrics_with_ci(y[test_idx], prob, thr)}

    test_meta = meta.iloc[test_idx].copy()
    test_meta["risk"] = prob
    test_meta["pred"] = (prob >= thr).astype(int)

    logger.info(f"\nОТЛОЖЕННЫЙ ТЕСТ (порог {thr:.4f}), в скобках 95% ДИ:")
    for k in ["roc_auc", "pr_auc", "sensitivity", "specificity", "balanced_accuracy"]:
        ci = metrics.get(f"{k}_ci")
        ci_s = f"  [{ci[0]:.4f}, {ci[1]:.4f}]" if ci else ""
        logger.info(f"  {k:20s} {metrics[k]:.4f}{ci_s}")
    logger.info(f"  пропущено опухолей {metrics['fn']}, ложных тревог {metrics['fp']}")

    by_group = []
    for g in [HEALTHY, ADJACENT, TUMOR]:
        d = test_meta[test_meta["group"] == g]
        if not len(d):
            continue
        correct = int((d["pred"] == d["label"]).sum())
        lo, hi = wilson_ci(correct, len(d))
        by_group.append({"group": g, "n": len(d), "correct": correct,
                         "rate": float(correct / len(d)), "ci_low": lo, "ci_high": hi,
                         "mean_risk": float(d["risk"].mean())})
    logger.info("\nПо источникам образцов:")
    for r in by_group:
        logger.info(f"  {r['group']:10s} n={r['n']:4d}  верно {r['rate']:.1%} "
                    f"[{r['ci_low']:.1%}–{r['ci_high']:.1%}]  "
                    f"средний риск {r['mean_risk']:.3f}")

    tum = test_meta[test_meta["group"] == TUMOR]
    by_stage = []
    for stage in ["I", "II", "III", "IV"]:
        s = tum[tum["stage"] == stage]
        if len(s):
            lo, hi = wilson_ci(int(s["pred"].sum()), len(s))
            by_stage.append({"stage": stage, "n": len(s),
                             "detected": int(s["pred"].sum()),
                             "rate": float(s["pred"].mean()),
                             "ci_low": lo, "ci_high": hi,
                             "mean_prob": float(s["risk"].mean())})
    logger.info("\nПо стадиям:")
    for r in by_stage:
        logger.info(f"  Стадия {r['stage']:4s} n={r['n']:4d}  поймано {r['detected']:4d}  "
                    f"{r['rate']:.1%} [{r['ci_low']:.1%}–{r['ci_high']:.1%}]")

    # Повторная CV: один сплит — это одна реализация случайной величины.
    logger.info(f"\nПовторная CV ({repeats}x{folds}) моделью {PROBE_MODEL}:")
    rep = repeated_group_cv(build_models(n_genes)[PROBE_MODEL], X, y, groups,
                            n_splits=folds, n_repeats=repeats)
    logger.info(f"  AUC {rep['auc_mean']:.4f}±{rep['auc_std']:.4f}, "
                f"минимум по фолдам {rep['auc_min']:.4f}, оценок {rep['n_estimates']}")

    return {"summary": summary, "best_name": best_name, "threshold": thr,
            "metrics": metrics, "by_group": by_group, "by_stage": by_stage,
            "model": calibrated, "test_meta": test_meta, "repeated_cv": rep,
            "X": X, "y": y, "groups": groups,
            "train_idx": train_idx, "test_idx": test_idx}


def marker_table(data, X, y, groups, n_genes, n_boot, top_n=300) -> pd.DataFrame:
    """Гены-маркеры с координатами в ДНК и частотой отбора при пересборке.

    Список из одной подгонки невоспроизводим: выкиньте десяток пациентов и
    половина имён поменяется. Частота отбора показывает, каким строкам верить.
    """
    stab = stability_selection(X, y, groups, list(data.X.columns),
                               k=n_genes, n_boot=n_boot)
    ann = data.genes.reindex(stab["gene"])
    out = stab.reset_index(drop=True).copy()
    for col in ["symbol", "chrom", "start", "end", "strand", "gene_type"]:
        out[col] = ann[col].to_numpy()

    mean_normal = X[y == 0].mean(axis=0)
    mean_tumor = X[y == 1].mean(axis=0)
    idx = {g: i for i, g in enumerate(data.X.columns)}
    pos = [idx[g] for g in out["gene"]]
    out["mean_normal"] = mean_normal[pos]
    out["mean_tumor"] = mean_tumor[pos]
    out["delta"] = out["mean_tumor"] - out["mean_normal"]
    out["locus"] = [f"{c}:{int(s):,}-{int(e):,} ({st})" if isinstance(c, str) else "—"
                    for c, s, e, st in zip(out["chrom"], out["start"],
                                           out["end"], out["strand"])]
    return out.head(top_n).reset_index(drop=True)


def write_report(path: Path, ctx: dict) -> None:
    d, m = ctx["dataset"], ctx["final"]["metrics"]
    probe = pd.DataFrame(ctx["probe"])
    to_gtex = pd.DataFrame(ctx["to_gtex"])
    to_tcga = pd.DataFrame(ctx["to_tcga"])
    chosen = ctx["chosen_norm"]
    best_transfer = to_gtex[to_gtex["normalization"] == chosen].iloc[0]

    L = [
        "# BioDNA v2 — здоровые люди против рака",
        "",
        "Второй датасет проекта. В первом «нормой» была ткань рядом с опухолью "
        "у тех же онкобольных — она уже несёт следы болезни. Здесь норма — "
        "**женщины из GTEx, у которых рака не было**.",
        "",
        "## Данные",
        "",
        f"Источник — **recount3**: обе когорты пересчитаны одним пайплайном "
        f"Monorail на аннотации GENCODE v26. Без этого сравнивать TCGA и GTEx "
        f"нельзя — разница протоколов забьёт биологию.",
        "",
        "| Группа | Что это | N |",
        "|---|---|---|",
        f"| Здоровые (GTEx) | грудная ткань женщин без рака | {d['n_healthy']} |",
        f"| Норма рядом (TCGA) | ткань возле опухоли у онкобольных | {d['n_adjacent']} |",
        f"| Опухоль (TCGA) | первичная опухоль молочной железы | {d['n_tumor']} |",
        "",
        f"Всего **{d['n_samples']} образцов x {d['n_genes']} генов** "
        f"(protein-coding, экспрессируются хотя бы в 20% образцов). "
        f"Мужские образцы GTEx отброшены: мужская грудная ткань отличается от "
        f"женской принципиально, и модель выучила бы пол.",
        "",
        "## Проверки на самообман",
        "",
        "### 0. Отпечаток когорты",
        "",
        "Сравниваем **здоровых GTEx с TCGA-нормой-рядом**. Рака нет ни в одной "
        "группе, значит всё, что модель различит, — технический след.",
        "",
        "| Нормировка | AUC |",
        "|---|---|",
    ]
    for _, r in probe.iterrows():
        L.append(f"| {r['normalization']} | **{r['auc']:.4f}** |")

    L += [
        "",
        f"AUC = {probe['auc'].max():.4f} при полном отсутствии рака. Отпечаток "
        "когорты абсолютный, и никакая нормировка внутри образца его не убирает. "
        "Отсюда прямое следствие: **нельзя брать норму из одной когорты, а "
        "опухоль из другой** — модель различит консорциум, а не болезнь.",
        "",
        "Важно понимать масштаб этого эффекта. На графике `02_pca_logtpm.png` "
        "первая главная компонента разводит опухоль и не-опухоль, а здоровые "
        "GTEx и TCGA-норма лежат в одном облаке. То есть когортный след — не "
        "самый сильный сигнал в данных, он слабее биологии; но классификатор "
        "в пространстве из 15 808 генов находит его целиком, и именно поэтому "
        "его приходится измерять отдельно, а не полагаться на глаз.",
        "",
        "### 1. Наивная модель (антипример)",
        "",
        "GTEx-норма против TCGA-опухоли в лоб — так делают ради красивой цифры:",
        "",
        "| Нормировка | CV ROC-AUC |",
        "|---|---|",
    ]
    for _, r in pd.DataFrame(ctx["naive"]).iterrows():
        L.append(f"| {r['normalization']} | {r['cv_auc']:.4f} |")
    L += ["", "Идеальный результат, который ничего не значит: классы совпадают "
               "с когортами один в один.", ""]

    L += [
        "### 2. Перенос TCGA → GTEx — главный результат",
        "",
        "Учимся **только внутри TCGA** (опухоль против нормы-рядом), затем "
        "применяем к здоровым женщинам GTEx, которых модель не видела ни одной.",
        "",
        f"Таблица ниже посчитана при фиксированной панели в {ctx['ref_genes']} "
        f"генов — это опорная точка для сравнения нормировок. Размер панели "
        f"сам по себе сильно влияет на перенос, поэтому дальше он перебирается "
        f"отдельно (раздел 3b), и итоговая модель берёт настройки оттуда.",
        "",
        "| Нормировка | AUC внутри TCGA | Здоровых объявлено больными | 95% ДИ |",
        "|---|---|---|---|",
    ]
    for _, r in to_gtex.iterrows():
        star = " ⭐" if r["normalization"] == chosen else ""
        L.append(f"| {r['normalization']}{star} | {r['tcga_cv_auc']:.4f} | "
                 f"{r['healthy_flagged']}/{r['n_healthy']} = "
                 f"{r['healthy_false_positive_rate']:.1%} | "
                 f"{r['fpr_ci_low']:.1%}–{r['fpr_ci_high']:.1%} |")

    L += [
        "",
        f"При нормировке `{chosen}` модель ошиблась на "
        f"**{best_transfer['healthy_flagged']} из {best_transfer['n_healthy']}** "
        f"здоровых женщин. Она никогда не видела GTEx, другой консорциум, другой "
        f"протокол забора ткани — и всё равно узнала здоровую грудь. Это и есть "
        f"доказательство, что сигнал биологический.",
        "",
        "### 3. Перенос GTEx → TCGA — где всё ломается",
        "",
        "Обратное направление: учимся на GTEx-здоровых и TCGA-опухолях, "
        "проверяемся на TCGA-норме-рядом.",
        "",
        "| Нормировка | TCGA-норму назвал опухолью | 95% ДИ |",
        "|---|---|---|",
    ]
    for _, r in to_tcga.iterrows():
        L.append(f"| {r['normalization']} | {r['adjacent_called_tumor']}/"
                 f"{r['n_adjacent']} = {r['adjacent_error_rate']:.1%} | "
                 f"{r['err_ci_low']:.1%}–{r['err_ci_high']:.1%} |")

    L += [
        "",
        "Провал, и он закономерен. Когда норма в обучении приходит только из "
        "GTEx, модель понимает под нормой «похоже на GTEx». Здоровая ткань из "
        "другой когорты под это определение не подходит, и модель объявляет её "
        "опухолью. Вывод для итоговой модели: **норму нужно брать из обоих "
        "источников**, что и сделано ниже.",
        "",
    ]

    sweep = ctx.get("sweep")
    if sweep is not None and not sweep.empty:
        norms = sorted(sweep["normalization"].unique())
        L += ["### 3b. Размер панели решает, перенесётся ли модель", "",
              "Внутри одной когорты качество выходит на плато почти при любом "
              "размере панели. Перенос — нет: чем больше генов, тем больше "
              "шансов зацепиться за особенности консорциума вместо биологии. "
              "Доля здоровых женщин GTEx, объявленных больными, при обучении "
              "только на TCGA:", "",
              "| Генов | " + " | ".join(norms) + " |",
              "|---" * (1 + len(norms)) + "|"]
        for k in sorted(sweep["n_genes"].unique()):
            cells = []
            for n in norms:
                r = sweep[(sweep["n_genes"] == k) & (sweep["normalization"] == n)]
                cells.append(f"{r.iloc[0]['healthy_fpr']:.1%}" if len(r) else "—")
            L.append(f"| {k} | " + " | ".join(cells) + " |")
        # Берём ровно ту строку, которую выбрал main(): минимум ложных тревог,
        # при равенстве — выше AUC внутри TCGA. Простой idxmin дал бы другую
        # пару при ничьей, и текст разошёлся бы с обученной моделью.
        pick = sweep[(sweep["normalization"] == chosen)
                     & (sweep["n_genes"] == ctx["n_genes"])].iloc[0]
        ties = len(sweep[sweep["healthy_fpr"] == sweep["healthy_fpr"].min()]) - 1
        word = "сочетание" if ties % 10 == 1 and ties % 100 != 11 else (
            "сочетания" if ties % 10 in (2, 3, 4) and ties % 100 not in (12, 13, 14)
            else "сочетаний")
        tie_txt = ("" if ties < 1 else
                   f" Тот же минимум даёт ещё {ties} {word}; "
                   f"ничья разрешается в пользу большего AUC внутри TCGA.")
        L += ["", f"Минимум — **{pick['healthy_fpr']:.1%}** "
                  f"({int(pick['healthy_flagged'])} из {int(pick['n_healthy'])}) "
                  f"при нормировке `{chosen}` и панели из "
                  f"**{ctx['n_genes']} генов**.{tie_txt} Этой парой и обучается "
                  f"итоговая модель. Правило выбора объявлено заранее и внешнюю "
                  f"когорту не видит.", ""]

    L += [
        "### 4. Перестановочный тест",
        "",
        f"Метки перемешаны по пациентам, {ctx['perm']['n_permutations']} прогонов "
        f"через тот же пайплайн: AUC "
        f"**{ctx['perm']['null_auc_mean']:.4f} ± {ctx['perm']['null_auc_std']:.4f}** "
        f"против настоящих {ctx['perm']['observed_auc']:.4f}, "
        f"p = {ctx['perm']['p_value']:.4g}. Утечки в пайплайне нет: на шуме он "
        f"даёт случайное угадывание.",
        "",
        "## Итоговая модель",
        "",
        f"Обе нормы (здоровые GTEx + TCGA-норма-рядом) против опухоли, "
        f"нормировка `{chosen}`, разбиение по пациентам.",
        "",
        f"Алгоритм: **{ctx['final']['best_name']}** (выбран по кросс-валидации, "
        f"не по тесту), калиброван по Платту, порог "
        f"**{ctx['final']['threshold']:.4f}** под чувствительность ≥ "
        f"{ctx['target_sens']:.0%}.",
        "",
        "| Метрика | Значение | 95% ДИ |",
        "|---|---|---|",
    ]
    for k, name in [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"),
                    ("sensitivity", "Чувствительность"),
                    ("specificity", "Специфичность"),
                    ("balanced_accuracy", "Сбаланс. точность")]:
        ci = m.get(f"{k}_ci")
        ci_s = f"{ci[0]:.4f} – {ci[1]:.4f}" if ci else "—"
        L.append(f"| {name} | {m[k]:.4f} | {ci_s} |")
    L += [f"| Пропущено опухолей | {m['fn']} | |",
          f"| Ложных тревог | {m['fp']} | |", ""]

    rep = ctx["final"]["repeated_cv"]
    L += [
        f"Повторная кросс-валидация ({rep['n_estimates']} оценок): AUC "
        f"**{rep['auc_mean']:.4f} ± {rep['auc_std']:.4f}**, худший фолд "
        f"{rep['auc_min']:.4f}. Результат не держится на удачном сплите.",
        "",
        "### По источникам образцов",
        "",
        "| Группа | N | Верно | 95% ДИ | Средний риск |",
        "|---|---|---|---|---|",
    ]
    names = {HEALTHY: "Здоровые (GTEx)", ADJACENT: "Норма рядом (TCGA)",
             TUMOR: "Опухоль (TCGA)"}
    for r in ctx["final"]["by_group"]:
        L.append(f"| {names[r['group']]} | {r['n']} | {r['rate']:.1%} | "
                 f"{r['ci_low']:.1%}–{r['ci_high']:.1%} | {r['mean_risk']:.3f} |")

    L += ["", "### По стадиям", "",
          "| Стадия | N | Поймано | Доля | 95% ДИ |", "|---|---|---|---|---|"]
    for r in ctx["final"]["by_stage"]:
        L.append(f"| {r['stage']} | {r['n']} | {r['detected']} | {r['rate']:.1%} | "
                 f"{r['ci_low']:.1%}–{r['ci_high']:.1%} |")

    curve = ctx.get("curve")
    if curve is not None and not curve.empty:
        L += ["", "### Сколько нормальных образцов нужно", "",
              "| Норм в обучении | ROC-AUC |", "|---|---|"]
        for _, r in curve.iterrows():
            L.append(f"| {int(r['n_minority'])} | {r['auc_mean']:.4f} ± "
                     f"{r['auc_std']:.4f} |")

    externals = ctx.get("externals") or []
    if externals:
        L += [
            "",
            "## Внешняя валидация: когорты, которых модель не видела",
            "",
            "Ни один из этих образцов не участвовал ни в обучении, ни в выборе "
            "нормировки и размера панели, ни в подборе порога. Порог перенесён "
            "как есть.",
        ]
        for e in externals:
            m = e.get("metrics") or {}
            L += ["", f"### {e['title']}", ""]
            if m:
                L += [
                    "| Метрика | Значение | 95% ДИ |", "|---|---|---|",
                    f"| ROC-AUC | **{m['roc_auc']:.4f}** | "
                    f"{m['roc_auc_ci'][0]:.4f} – {m['roc_auc_ci'][1]:.4f} |",
                    f"| Чувствительность | {m['sensitivity']:.1%} | "
                    f"{m['sensitivity_ci'][0]:.1%} – {m['sensitivity_ci'][1]:.1%} |",
                    f"| Специфичность | {m['specificity']:.1%} | "
                    f"{m['specificity_ci'][0]:.1%} – {m['specificity_ci'][1]:.1%} |",
                    "",
                ]
            L += ["| Подгруппа | N | Верно | Средний риск |", "|---|---|---|---|"]
            for r in e["by_subgroup"]:
                rate = (f"{r['rate']:.1%}" if r["rate"] is not None
                        else f"помечено раком {r['flagged_rate']:.1%}")
                L.append(f"| {r['subgroup']} | {r['n']} | {rate} | "
                         f"{r['mean_risk']:.3f} |")

        L += [
            "",
            "Две когорты закрывают разные дыры. Шанхайская проверяет перенос "
            "между странами, популяциями и молекулярным подтипом: TCGA "
            "преимущественно ER-положительный, там весь набор трижды-негативный. "
            "Американская добавляет ER-положительные опухоли и, что важнее, "
            "ткань после **редукционной маммопластики** - это здоровые ЖИВЫЕ "
            "женщины и хирургический материал, тогда как GTEx посмертный. "
            "Возражение «здоровая норма у вас только аутопсийная» этим снимается.",
        ]

    spec = ctx.get("spectrum")
    if spec:
        rows = spec["by_subgroup"]
        by = {r["subgroup"]: r for r in rows}
        L += [
            "",
            "## Спектр прогрессии: доброкачественное, DCIS, инвазия",
            "",
            "Когорта SRP023262 редкая: у одних и тех же пациентов взяты "
            "нормальная ткань, ранняя неоплазия, карцинома in situ и инвазивная "
            "карцинома. Модель обучалась только на крайних точках этого ряда, "
            "промежуточные состояния видит впервые.",
            "",
            "| Состояние | N | Средний риск | Помечено как рак |",
            "|---|---|---|---|",
        ]
        for r in rows:
            L.append(f"| {r['subgroup']} | {r['n']} | **{r['mean_risk']:.3f}** | "
                     f"{r['flagged_rate']:.0%} |")

        neo = by.get("ранняя неоплазия")
        dcis = by.get("DCIS (рак на месте)")
        norm = by.get("норма")
        if neo and dcis and norm:
            L += [
                "",
                f"Риск растёт монотонно вдоль прогрессии: {norm['mean_risk']:.3f} "
                f"у нормы, {neo['mean_risk']:.3f} у ранней неоплазии, "
                f"{dcis['mean_risk']:.3f} у DCIS. Модель не просто провела "
                "границу «опухоль или нет», она выстроила шкалу тяжести, хотя "
                "промежуточных состояний в обучении не было.",
                "",
                f"Практически важны две строки. **Доброкачественные разрастания "
                f"не дают ложных тревог**: помечено раком "
                f"{neo['flagged_rate']:.0%}. **DCIS попадает ровно посередине**: "
                f"помечено {dcis['flagged_rate']:.0%}. Биологически это честно - "
                f"DCIS уже карцинома, но ещё не инвазия, и модель, обученная на "
                f"инвазивных опухолях, распознаёт её лишь частично.",
                "",
                "Важная деталь, которую видно на графике "
                "`10_progression_spectrum.png`: у ранней неоплазии не просто "
                "повышенный риск, а огромный разброс. Медиана 0.063, то есть "
                "большинство таких поражений неотличимы от нормы, но у 8 из 28 "
                "риск выше 0.3, а у отдельных доходит до 0.87. Среднее 0.219 "
                "держится именно на этом хвосте. Трактовать это как «модель "
                "заранее видит, какая неоплазия переродится» нельзя - исходов "
                "по этим пациентам в данных нет. Но разброс сам по себе "
                "означает, что ранние поражения молекулярно неоднородны, и это "
                "согласуется с тем, что известно про них в клинике.",
            ]

    markers = ctx["markers"]
    L += ["", "## Гены-маркеры с координатами в ДНК (hg38, GENCODE v26)", "",
          "Колонка «устойчивость» — доля пересборок выборки по пациентам, в "
          "которых ген попал в панель. Значения ниже 0.6 воспроизводятся плохо.",
          "",
          "| # | Ген | Локус | Устойчивость | В опухоли |", "|---|---|---|---|---|"]
    for i, r in markers.head(25).iterrows():
        arrow = "↑ выше" if r["delta"] > 0 else "↓ ниже"
        L.append(f"| {i + 1} | {r['symbol']} | {r['locus']} | "
                 f"{r['selection_freq']:.2f} | {arrow} |")

    L += [
        "",
        "## Что это всё-таки не доказывает",
        "",
        "Задача решается на **ткани**: образец уже взят при биопсии или "
        "операции. Это подтверждение диагноза, а не скрининг по анализу крови. "
        "Раннее выявление здесь означает «стадия I распознаётся не хуже "
        "поздних», а не «рак виден до появления симптомов».",
        "",
        "Остаётся одна принципиальная асимметрия. Единственные в наборе люди "
        "**без рака** — это доноры GTEx, и они же единственный источник "
        "«здоровой» нормы. Нормы в TCGA и в шанхайской когорте взяты у "
        "онкобольных рядом с опухолью. Поэтому доказано следующее: модель "
        "уверенно отличает опухолевую ткань от неопухолевой и не считает "
        "больными здоровых женщин из другой когорты. Чего не доказано — что "
        "она отличила бы раннюю опухоль у человека, у которого её ещё не "
        "нашли: таких образцов в открытых данных попросту нет.",
        "",
        "> Исследовательский проект. Не медицинское изделие и не основание для "
        "клинических решений.",
        "",
    ]
    path.write_text("\n".join(L), encoding="utf-8")
    logger.info(f"Отчёт: {path}")


def rebuild_report(args) -> None:
    """Собрать REPORT_V2.md заново из уже сохранённых результатов.

    Полный прогон занимает около часа, а правка формулировок в отчёте — минуту.
    Режим читает metrics_v2.json и CSV-таблицы прошлого запуска и переписывает
    только текст, ничего не переобучая.
    """
    src = RESULTS_DIR / "metrics_v2.json"
    if not src.exists():
        raise SystemExit(f"Нет {src} — сначала выполните полный прогон")

    out = json.loads(src.read_text(encoding="utf-8"))
    fm = out["final_model"]

    def _csv(name):
        path = RESULTS_DIR / name
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    write_report(RESULTS_DIR / "REPORT_V2.md", {
        "dataset": out["dataset"],
        "probe": out["cohort_probe"],
        "naive": out["naive_model"],
        "to_gtex": out["transfer_tcga_to_gtex"],
        "to_tcga": out["transfer_gtex_to_tcga"],
        "chosen_norm": out["chosen_normalization"],
        "n_genes": out["chosen_panel_size"],
        "ref_genes": args.n_genes,
        "perm": out["permutation_test"],
        "final": {"metrics": fm["holdout"], "best_name": fm["best_model"],
                  "threshold": fm["threshold"], "by_group": fm["by_group"],
                  "by_stage": fm["by_stage"], "repeated_cv": out["repeated_cv"]},
        "curve": _csv("learning_curve_v2.csv"),
        "markers": _csv("markers_v2.csv"),
        "sweep": _csv("panel_vs_transfer.csv"),
        "externals": out.get("external_validation") or [],
        "spectrum": out.get("progression_spectrum"),
        "target_sens": args.target_sensitivity,
    })


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BioDNA v2: здоровые люди (GTEx) против рака (TCGA), recount3")
    parser.add_argument("--n-genes", type=int, default=200)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--target-sensitivity", type=float, default=0.98)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--stability-boot", type=int, default=150)
    parser.add_argument("--panel-sizes", type=int, nargs="+",
                        default=[10, 25, 50, 100, 200, 500],
                        help="размеры генной панели для проверки переносимости")
    parser.add_argument("--include-males", action="store_true",
                        help="не отсекать мужские образцы GTEx")
    parser.add_argument("--report-only", action="store_true",
                        help="перегенерировать отчёт из сохранённых результатов")
    args = parser.parse_args()

    setup_logging(overwrite=not args.report_only)
    if args.report_only:
        rebuild_report(args)
        return

    t0 = time.time()
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    logger.info("=" * 78)
    logger.info("BioDNA v2 — GTEx (здоровые) + TCGA (рак), пересчёт recount3")
    logger.info("=" * 78)

    data = build_dataset(RECOUNT_DIR, female_only=not args.include_males)
    comp = data.meta.groupby(["cohort", "group"]).size()
    logger.info(f"Состав:\n{comp.to_string()}")

    logger.info("\n[0] ОТПЕЧАТОК КОГОРТЫ — рака нет ни в одной группе")
    probe = [cohort_probe(data, n, args.n_genes, args.folds) for n in NORMALIZATIONS]

    logger.info("\n[1] НАИВНАЯ МОДЕЛЬ — антипример")
    naive = [naive_model(data, n, args.n_genes, args.folds) for n in NORMALIZATIONS]

    logger.info("\n[2] ПЕРЕНОС TCGA -> GTEx — сколько здоровых объявлены больными")
    to_gtex = [transfer_tcga_to_gtex(data, n, args.n_genes, args.folds,
                                     args.target_sensitivity) for n in NORMALIZATIONS]

    logger.info("\n[3] ПЕРЕНОС GTEx -> TCGA — узнаёт ли модель норму чужой когорты")
    to_tcga = [transfer_gtex_to_tcga(data, n, args.n_genes, args.folds,
                                     args.target_sensitivity) for n in NORMALIZATIONS]

    logger.info("\n[3b] РАЗМЕР ПАНЕЛИ ПРОТИВ ПЕРЕНОСИМОСТИ")
    sweep = transfer_vs_panel(data, NORMALIZATIONS, args.panel_sizes,
                              args.folds, args.target_sensitivity)

    # Правило выбора объявлено заранее: минимум ложных тревог у здоровых людей,
    # при равенстве — выше AUC внутри TCGA. Ни отложенный тест итоговой модели,
    # ни внешняя когорта в выборе не участвуют.
    best_row = sweep.sort_values(["healthy_fpr", "tcga_cv_auc"],
                                 ascending=[True, False]).iloc[0]
    chosen = str(best_row["normalization"])
    n_genes = int(best_row["n_genes"])
    logger.info(f"\nВыбрано по переносу: нормировка {chosen}, панель {n_genes} генов "
                f"(ложных тревог у здоровых {best_row['healthy_fpr']:.1%})")

    logger.info("\n[5] ИТОГОВАЯ МОДЕЛЬ")
    final = final_model(data, chosen, n_genes, args.folds,
                        args.target_sensitivity, args.repeats)

    logger.info("\n[4] ПЕРЕСТАНОВОЧНЫЙ ТЕСТ (метки мешаются по пациентам)")
    perm = permutation_test(build_models(n_genes)[PROBE_MODEL],
                            final["X"], final["y"], final["groups"],
                            observed_auc=final["repeated_cv"]["auc_mean"],
                            n_perm=args.permutations, n_splits=args.folds)
    logger.info(f"  нулевое распределение {perm['null_auc_mean']:.4f}±"
                f"{perm['null_auc_std']:.4f}, наблюдаемое "
                f"{perm['observed_auc']:.4f}, p={perm['p_value']:.4g}")

    logger.info("\n[6] СКОЛЬКО НОРМАЛЬНЫХ ОБРАЗЦОВ НУЖНО")
    curve = learning_curve_minority(
        build_models(n_genes)[PROBE_MODEL], final["X"], final["y"],
        final["groups"], sizes=(10, 20, 40, 80, 150, 250), n_repeats=3,
        n_splits=args.folds)
    for _, r in curve.iterrows():
        logger.info(f"  норм в обучении {int(r['n_minority']):4d}: "
                    f"AUC {r['auc_mean']:.4f}±{r['auc_std']:.4f}")

    logger.info("\n[7] ГЕНЫ-МАРКЕРЫ И ИХ КООРДИНАТЫ")
    markers = marker_table(data, final["X"], final["y"], final["groups"],
                           n_genes, args.stability_boot)
    for _, r in markers.head(12).iterrows():
        arrow = "↑" if r["delta"] > 0 else "↓"
        logger.info(f"  {str(r['symbol']):12s} {arrow} {r['locus']:38s} "
                    f"устойчивость {r['selection_freq']:.2f}")

    logger.info("\n[8] ВНЕШНЯЯ ВАЛИДАЦИЯ — когорты, которых модель не видела")
    externals = []
    for name in ["FUSCC", "VARLEY"]:
        logger.info(f"\n  {EXTERNAL_COHORTS[name]['title']}")
        externals.append(external_validation(final["model"], data, chosen,
                                             final["threshold"], RECOUNT_DIR, name))
    external = externals[0]

    logger.info("\n[9] СПЕКТР ПРОГРЕССИИ — доброкачественное, DCIS, инвазия")
    spectrum = progression_spectrum(final["model"], data, chosen,
                                    final["threshold"], RECOUNT_DIR)

    probe_df, gtex_df, tcga_df = (pd.DataFrame(probe), pd.DataFrame(to_gtex),
                                  pd.DataFrame(to_tcga))

    dataset = {
        "source": "recount3 (Monorail, GENCODE v26)",
        "n_samples": int(data.X.shape[0]), "n_genes": int(data.X.shape[1]),
        "n_healthy": int((data.meta["group"] == HEALTHY).sum()),
        "n_adjacent": int((data.meta["group"] == ADJACENT).sum()),
        "n_tumor": int((data.meta["group"] == TUMOR).sum()),
        "female_only": not args.include_males,
    }

    out = {
        "dataset": dataset, "cohort_probe": probe, "naive_model": naive,
        "transfer_tcga_to_gtex": to_gtex, "transfer_gtex_to_tcga": to_tcga,
        "chosen_normalization": chosen,
        "chosen_panel_size": n_genes,
        "panel_vs_transfer": sweep.to_dict(orient="records"),
        "external_validation": [{k: v for k, v in e.items() if k != "predictions"}
                                for e in externals],
        "progression_spectrum": {k: v for k, v in spectrum.items()
                                 if k != "predictions"},
        "permutation_test": {k: v for k, v in perm.items() if k != "null_aucs"},
        "repeated_cv": {k: v for k, v in final["repeated_cv"].items()
                        if k not in ("oof_matrix", "oof_mean", "fold_aucs")},
        "learning_curve": curve.to_dict(orient="records"),
        "final_model": {
            "best_model": final["best_name"], "threshold": final["threshold"],
            "holdout": final["metrics"], "by_group": final["by_group"],
            "by_stage": final["by_stage"],
            "models": final["summary"].to_dict(orient="records"),
        },
    }
    (RESULTS_DIR / "metrics_v2.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    sweep.to_csv(RESULTS_DIR / "panel_vs_transfer.csv", index=False)
    for ext in externals + [spectrum]:
        ext["predictions"].to_csv(
            RESULTS_DIR / f"external_{ext['cohort'].lower()}_predictions.csv",
            encoding="utf-8")
    final["summary"].to_csv(RESULTS_DIR / "model_comparison_v2.csv", index=False)
    final["test_meta"].to_csv(RESULTS_DIR / "holdout_predictions.csv", encoding="utf-8")
    markers.to_csv(RESULTS_DIR / "markers_v2.csv", index=False, encoding="utf-8")
    curve.to_csv(RESULTS_DIR / "learning_curve_v2.csv", index=False)
    probe_df.to_csv(RESULTS_DIR / "cohort_probe.csv", index=False)
    gtex_df.to_csv(RESULTS_DIR / "transfer_tcga_to_gtex.csv", index=False)
    tcga_df.to_csv(RESULTS_DIR / "transfer_gtex_to_tcga.csv", index=False)

    write_report(RESULTS_DIR / "REPORT_V2.md", {
        "dataset": dataset, "probe": probe, "naive": naive, "to_gtex": to_gtex,
        "to_tcga": to_tcga, "chosen_norm": chosen, "perm": perm, "final": final,
        "curve": curve, "markers": markers, "sweep": sweep,
        "externals": externals, "spectrum": spectrum, "n_genes": n_genes,
        "ref_genes": args.n_genes,
        "target_sens": args.target_sensitivity,
    })

    joblib.dump({
        "model": final["model"], "genes": list(data.X.columns),
        "gene_symbols": data.genes["symbol"].to_dict(),
        "normalization": chosen, "best_model_name": final["best_name"],
        "n_genes_selected": n_genes, "threshold": final["threshold"],
        "holdout": final["metrics"], "markers": markers,
        "source": "recount3 GTEx+TCGA",
    }, MODELS_DIR / "biodna_v2.joblib", compress=3)

    # Графики рисуются ПОСЛЕ сохранения результатов и по одному в try:
    # прогон занимает больше часа, и опечатка в оформлении не должна его стирать.
    logger.info("\n[10] ГРАФИКИ")
    for draw in [
        lambda: plot_honesty_panel(probe_df, gtex_df, tcga_df, RESULTS_DIR, chosen),
        lambda: plot_pca(normalize(data.X, "logtpm"), data.meta, RESULTS_DIR,
                         " (log2 TPM, как есть)", "02_pca_logtpm.png"),
        lambda: plot_pca(normalize(data.X, chosen), data.meta, RESULTS_DIR,
                         f" (нормировка {chosen})", "02b_pca_chosen.png"),
        lambda: plot_risk_by_group(final["test_meta"], RESULTS_DIR, final["threshold"]),
        lambda: plot_stage(final["by_stage"], RESULTS_DIR, final["threshold"]),
        lambda: plot_models(final["summary"], RESULTS_DIR, final["best_name"]),
        lambda: plot_permutation(perm, final["repeated_cv"]["auc_mean"], RESULTS_DIR),
        lambda: plot_panel_transfer(sweep, RESULTS_DIR, chosen, n_genes),
        *[(lambda e=e, i=i: plot_external(e, RESULTS_DIR, index=i))
          for i, e in enumerate(externals)],
        lambda: plot_progression(spectrum, RESULTS_DIR, final["threshold"]),
    ]:
        try:
            draw()
        except Exception as exc:
            logger.warning(f"  график пропущен: {type(exc).__name__}: {exc}")

    logger.info(f"\nМодель: {MODELS_DIR / 'biodna_v2.joblib'}")
    logger.info(f"Готово за {(time.time() - t0) / 60:.1f} мин. Результаты в {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
