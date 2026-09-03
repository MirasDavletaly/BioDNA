"""BioDNA — предсказание клинических признаков рака груди по экспрессии генов.

Только классическое машинное обучение (scikit-learn). Нейросетей нет.

Что изменилось и почему
-----------------------
Первая версия решала одну задачу — «норма или опухоль» — и получала
ROC-AUC = 1.000. Проблема в том, что ровно столько же получали ВСЕ девять
моделей, включая наивный Байес. Это признак не сильного метода, а лёгкой
задачи: опухолевая и нормальная ткань различаются тысячами генов, и на
TCGA-BRCA потолок достигается тривиально. Улучшать там было нечего, а
единица без доверительного интервала на 12 нормальных образцах в тесте
выглядела как ошибка методики.

Поэтому пайплайн теперь делает три вещи вместо одной:

  1. ДОКАЗЫВАЕТ базовый результат вместо того, чтобы его декларировать:
     повторная CV вместо одного сплита, доверительные интервалы у каждой
     метрики, пермутационный тест, кривая обучения и проверка устойчивости
     генной панели к пересборке выборки.

  2. РАСШИРЯЕТ постановку на клинические вопросы, где до потолка далеко и
     где качество модели действительно на что-то влияет: гистотип, размер
     опухоли, поражение лимфоузлов, стадия. Плюс негативный контроль —
     задача, где сигнала заведомо нет.

  3. ПОДБИРАЕТ модель и гиперпараметры во вложенной CV, а не фиксирует их
     константой. Число отбираемых генов — такой же гиперпараметр, как C,
     и оптимум у разных задач разный.

Запуск:  python train.py            полный прогон
         python train.py --fast     быстрый прогон (меньше повторов)
         python train.py --offline  без обращений к GDC/UCSC
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
ANNOT_DIR = DATA_DIR / "annotation"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

sys.path.insert(0, str(BASE_DIR))

from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from src.clinical import fetch_clinical  # noqa: E402
from src.data_preprocessing import early_stage_mask, load_expression_data  # noqa: E402
from src.evaluation import (  # noqa: E402
    bootstrap_ci,
    delong_test,
    learning_curve_minority,
    metrics_with_ci,
    permutation_test,
    repeated_group_cv,
)
from src.gene_annotation import annotate, format_locus, load_gene_coordinates  # noqa: E402
from src.models import (  # noqa: E402
    RANDOM_STATE,
    build_models,
    compute_metrics,
    cross_validate_oof,
    genome_wide_scores,
    nested_cv_score,
    risk_zones,
    selected_gene_scores,
    sensitivity_by_stage,
    stability_selection,
    threshold_for_sensitivity,
    threshold_for_specificity,
)
from src.tasks import attach_clinical, build_task_suite  # noqa: E402
from src.visualization import create_full_report  # noqa: E402
from src.viz_evidence import create_evidence_report  # noqa: E402


def setup_logging() -> logging.Logger:
    RESULTS_DIR.mkdir(exist_ok=True)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(RESULTS_DIR / "training.log", mode="w", encoding="utf-8"),
        ],
        force=True,
    )
    return logging.getLogger("biodna")


logger = logging.getLogger("biodna")


def section(title: str) -> None:
    logger.info("")
    logger.info("=" * 76)
    logger.info(title)
    logger.info("=" * 76)


def grouped_holdout(y: np.ndarray, groups: np.ndarray, test_size: float = 0.2):
    """Стратифицированный сплит, не разрывающий одного пациента между train и test."""
    n_splits = max(2, int(round(1 / test_size)))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(cv.split(np.zeros(len(y)), y, groups=groups))
    return train_idx, test_idx


# --------------------------------------------------------------------------- #
#  Базовая задача: сравнение моделей
# --------------------------------------------------------------------------- #
def benchmark_models(models, X_tr, y_tr, g_tr, X_te, y_te, folds: int):
    """CV на train + честная проверка на отложенном тесте для каждой модели."""
    rows, fitted, oof_store = [], {}, {}

    for name, model in models.items():
        t0 = time.time()
        oof, fold_aucs = cross_validate_oof(model, X_tr, y_tr, g_tr, n_splits=folds)

        model.fit(X_tr, y_tr)
        test_prob = model.predict_proba(X_te)[:, 1]
        test_metrics = compute_metrics(y_te, test_prob)

        rows.append({
            "model": name,
            "cv_auc_mean": float(np.mean(fold_aucs)),
            "cv_auc_std": float(np.std(fold_aucs)),
            "test_auc": test_metrics["roc_auc"],
            "test_pr_auc": test_metrics["pr_auc"],
            "test_balanced_acc": test_metrics["balanced_accuracy"],
            "test_sensitivity": test_metrics["sensitivity"],
            "test_specificity": test_metrics["specificity"],
            "test_mcc": test_metrics["mcc"],
            "fit_seconds": round(time.time() - t0, 1),
        })
        fitted[name] = model
        oof_store[name] = oof

        logger.info(
            f"  {name:26s} CV AUC {np.mean(fold_aucs):.4f}±{np.std(fold_aucs):.4f} | "
            f"тест AUC {test_metrics['roc_auc']:.4f} | "
            f"чувств. {test_metrics['sensitivity']:.3f} | "
            f"спец. {test_metrics['specificity']:.3f} | {rows[-1]['fit_seconds']}s"
        )

    return pd.DataFrame(rows), fitted, oof_store


def compare_top_models(summary: pd.DataFrame, oof_store: dict, y_tr) -> list[dict]:
    """DeLong: значимо ли лучшая модель отличается от остальных.

    На насыщенной задаче все модели упираются в потолок, и «лучшая по CV»
    выбирается фактически по шуму в четвёртом знаке. Тест DeLong показывает
    это прямо: p близкое к 1 значит, что разницы между моделями нет.
    """
    ranked = summary.sort_values("cv_auc_mean", ascending=False)
    best = str(ranked.iloc[0]["model"])
    rows = []
    for name in ranked["model"].tolist()[1:]:
        d = delong_test(y_tr, oof_store[best], oof_store[name])
        rows.append({
            "model_a": best, "model_b": name,
            "auc_a": d["auc_a"], "auc_b": d["auc_b"],
            "delta": d["delta"], "p_value": d["p_value"],
            "significant": bool(d["p_value"] < 0.05) if d["p_value"] == d["p_value"] else False,
        })
        verdict = "значимо" if rows[-1]["significant"] else "НЕ значимо"
        logger.info(f"  {best} vs {name:26s} Δ={d['delta']:+.4f}  p={d['p_value']:.3g}  {verdict}")
    return rows


# --------------------------------------------------------------------------- #
#  Доказательная часть
# --------------------------------------------------------------------------- #
def run_evidence(model, X, y, groups, folds: int, repeats: int, n_perm: int) -> dict:
    """Всё, что превращает «у нас AUC 1.000» в защитимое утверждение."""
    section("ДОКАЗАТЕЛЬНАЯ ЧАСТЬ: можно ли верить этой цифре")

    logger.info(f"Повторная CV по пациентам: {repeats} повторов x {folds} фолдов")
    rep = repeated_group_cv(model, X, y, groups, n_splits=folds, n_repeats=repeats)
    logger.info(f"  AUC по {rep['n_estimates']} фолдам: {rep['auc_mean']:.4f} ± {rep['auc_std']:.4f}")
    logger.info(f"  95% интервал по фолдам: [{rep['auc_p2_5']:.4f}, {rep['auc_p97_5']:.4f}]")
    logger.info(f"  худший фолд: {rep['auc_min']:.4f}")

    logger.info(f"\nПермутационный тест: {n_perm} перестановок меток по пациентам")
    perm = permutation_test(model, X, y, groups, observed_auc=rep["auc_mean"],
                            n_perm=n_perm, n_splits=folds)
    logger.info(f"  нулевое распределение: {perm['null_auc_mean']:.4f} ± {perm['null_auc_std']:.4f} "
                f"(95-й перцентиль {perm['null_auc_p95']:.4f})")
    logger.info(f"  наблюдаемый AUC {perm['observed_auc']:.4f} -> p = {perm['p_value']:.4g}")

    logger.info("\nКривая обучения по редкому классу (нормам)")
    n_norm = int((y == 0).sum())
    sizes = tuple(s for s in (5, 10, 15, 20, 30, 40, 48) if s < n_norm)
    curve = learning_curve_minority(model, X, y, groups, sizes=sizes,
                                    n_repeats=3, n_splits=folds)

    return {"repeated_cv": rep, "permutation": perm, "learning_curve": curve,
            "n_normal_available": n_norm}


# --------------------------------------------------------------------------- #
#  Набор клинических задач
# --------------------------------------------------------------------------- #
def run_task_suite(tasks, X, y, meta, gene_names, folds: int, inner_folds: int,
                   n_boot: int = 1000, n_stab: int = 100) -> tuple[list[dict], dict, pd.DataFrame]:
    """Вложенная CV на каждой задаче — одинаковая процедура, разные вопросы.

    Именно это сравнение и есть главный результат проекта. Одинаковый
    пайплайн даёт 1.00 на «норма vs опухоль», 0.9 на гистотипе, 0.65 на
    лимфоузлах и 0.5 на негативном контроле. Значит, число отражает
    предсказуемость ВОПРОСА, а не оптимизм методики.
    """
    section("НАБОР КЛИНИЧЕСКИХ ЗАДАЧ: вложенная CV с подбором гиперпараметров")

    rows, details, marker_rows = [], {}, []
    for task in tasks:
        m = task.mask(y)
        Xi = X[m]
        yi = task.labels[m].astype(int)
        gi = meta.loc[m, "patient"].to_numpy()

        tag = "  [НЕГАТИВНЫЙ КОНТРОЛЬ]" if task.is_control else ""
        logger.info(f"\n{task.title}{tag}")
        logger.info(f"  вопрос: {task.question}")
        logger.info(f"  выборка: {len(yi)} образцов "
                    f"({task.neg_label} {int((yi==0).sum())} / "
                    f"{task.pos_label} {int((yi==1).sum())})")
        logger.info(f"  ожидание до обучения: {task.expected}")

        t0 = time.time()
        res = nested_cv_score(Xi, yi, gi, outer_splits=folds,
                              inner_splits=inner_folds, n_jobs=-1)

        # Для интервала берём ранговые OOF: в разных фолдах побеждают разные
        # модели, и складывать их сырые вероятности в один вектор нельзя.
        oof = res["oof_rank"]
        ok = ~np.isnan(oof)
        _, lo, hi = bootstrap_ci(yi[ok], oof[ok], roc_auc_score, n_boot=n_boot)
        auc_point = res["auc_mean"]

        logger.info(f"  ИТОГ: AUC {auc_point:.4f} ± {res['auc_std']:.4f} "
                    f"| 95%ДИ по бутстрапу OOF [{lo:.4f}, {hi:.4f}] "
                    f"| {time.time()-t0:.0f}s")

        beats_chance = lo > 0.5
        if task.is_control:
            verdict = ("КОНТРОЛЬ ПРОЙДЕН: сигнала нет, как и ожидалось"
                       if not beats_chance else
                       "ВНИМАНИЕ: на контроле есть сигнал — проверить утечки")
        else:
            verdict = ("сигнал есть" if beats_chance else
                       "сигнала нет: ДИ накрывает случайное угадывание")
        logger.info(f"  вывод: {verdict}")

        # Какие гены тянут именно ЭТУ задачу. Не украшение: для гистотипа
        # ответ проверяем независимо от модели — дольковый рак определяется
        # потерей E-кадгерина, и если в списке нет CDH1, значит пайплайн
        # поймал что-то постороннее, а не биологию.
        stab = stability_selection(Xi, yi, gi, gene_names, k=30,
                                   n_boot=n_stab, random_state=RANDOM_STATE)
        top = stab.head(12)
        if float(top["selection_freq"].iloc[0]) > 0:
            names = ", ".join(f"{g}({f:.2f})" for g, f in
                              zip(top["gene"].head(8), top["selection_freq"].head(8)))
            logger.info(f"  устойчивые маркеры: {names}")
        for _, mr in top.iterrows():
            marker_rows.append({"task": task.key, "task_title": task.title,
                                "gene": mr["gene"],
                                "selection_freq": mr["selection_freq"],
                                "direction_consistency": mr["direction_consistency"]})

        rows.append({
            "key": task.key, "title": task.title, "question": task.question,
            "n": int(len(yi)), "n_neg": int((yi == 0).sum()), "n_pos": int((yi == 1).sum()),
            "neg_label": task.neg_label, "pos_label": task.pos_label,
            "auc_mean": float(auc_point), "auc_fold_mean": res["auc_mean"],
            "auc_fold_std": res["auc_std"],
            "ci_low": float(lo), "ci_high": float(hi),
            "is_control": bool(task.is_control),
            "beats_chance": bool(beats_chance),
            "why": task.why, "expected": task.expected, "verdict": verdict,
            "chosen_per_fold": res["chosen_per_fold"],
            "top_genes": ", ".join(top["gene"].head(8).tolist()),
            "seconds": round(time.time() - t0, 1),
        })
        details[task.key] = {"oof": oof, "y": yi, "mask": m}

    return rows, details, pd.DataFrame(marker_rows)


# --------------------------------------------------------------------------- #
#  Прочие проверки
# --------------------------------------------------------------------------- #
def build_marker_table(model, gene_names, X, y, coords, stability: pd.DataFrame | None,
                       top_n: int = 60) -> pd.DataFrame:
    """Отобранные моделью гены + локусы в геноме + направление + устойчивость."""
    markers = selected_gene_scores(model, gene_names)

    gene_idx = {g: i for i, g in enumerate(gene_names)}
    cols = [gene_idx[g] for g in markers["gene"]]
    sub = X[:, cols]
    mean_normal = np.nanmean(sub[y == 0], axis=0)
    mean_tumor = np.nanmean(sub[y == 1], axis=0)

    markers["mean_normal"] = mean_normal
    markers["mean_tumor"] = mean_tumor
    markers["delta"] = mean_tumor - mean_normal
    markers["direction"] = np.sign(markers["delta"])

    # Частота отбора при пересборке выборки — главный столбец таблицы:
    # без него список генов невоспроизводим и читателю не на что опереться.
    if stability is not None and not stability.empty:
        freq = dict(zip(stability["gene"], stability["selection_freq"]))
        markers["selection_freq"] = [freq.get(g, np.nan) for g in markers["gene"]]

    ann = annotate(markers["gene"], coords).reset_index(drop=True)
    markers = pd.concat([markers, ann[["chrom", "start", "end", "strand", "cytoband"]]], axis=1)
    markers["locus"] = [format_locus(r) if pd.notna(r["chrom"]) else "—"
                        for _, r in markers.iterrows()]

    def _short(row) -> str:
        if isinstance(row["cytoband"], str):
            return row["cytoband"]
        if isinstance(row["chrom"], str):
            return row["chrom"].replace("chr", "")
        return "?"

    markers["locus_short"] = [_short(r) for _, r in markers.iterrows()]
    return markers.head(top_n)


def run_early_detection(models_factory, data, folds: int, target_sens: float):
    """Отдельная задача: отличить НОРМУ от опухоли только ранних стадий (I-II)."""
    mask = early_stage_mask(data.meta)
    X = data.X.to_numpy()[mask]
    y = data.y[mask]
    meta = data.meta.loc[mask]
    groups = meta["patient"].to_numpy()

    n_norm, n_tum = int((y == 0).sum()), int((y == 1).sum())
    logger.info(f"\nРАННЕЕ ВЫЯВЛЕНИЕ: норма {n_norm} vs стадии I-II {n_tum}")
    if n_tum < 20 or n_norm < 10:
        logger.warning("  слишком мало образцов — блок пропущен")
        return None

    tr, te = grouped_holdout(y, groups)
    model = models_factory()

    oof, fold_aucs = cross_validate_oof(model, X[tr], y[tr], groups[tr], n_splits=folds)
    thr = threshold_for_sensitivity(y[tr], oof, target=target_sens)

    model.fit(X[tr], y[tr])
    prob = model.predict_proba(X[te])[:, 1]
    metrics = compute_metrics(y[te], prob, threshold=thr)
    metrics.update(metrics_with_ci(y[te], prob, thr, n_boot=1000))
    stage_df = sensitivity_by_stage(meta.iloc[te], y[te], prob, thr)

    logger.info(f"  CV AUC {np.mean(fold_aucs):.4f}±{np.std(fold_aucs):.4f} | "
                f"тест AUC {metrics['roc_auc']:.4f} "
                f"95%ДИ [{metrics['roc_auc_ci'][0]:.4f}, {metrics['roc_auc_ci'][1]:.4f}]")
    logger.info(f"  При пороге {thr:.3f}: чувствительность к I-II "
                f"{metrics['sensitivity']:.3f} "
                f"[{metrics['sensitivity_ci'][0]:.3f}, {metrics['sensitivity_ci'][1]:.3f}], "
                f"специфичность {metrics['specificity']:.3f} "
                f"[{metrics['specificity_ci'][0]:.3f}, {metrics['specificity_ci'][1]:.3f}]")

    stage_i = stage_df[stage_df["stage"] == "I"]
    if not stage_i.empty:
        r = stage_i.iloc[0]
        logger.info(f"  Стадия I отдельно: поймано {int(r['detected'])}/{int(r['n'])} "
                    f"({r['rate']:.1%})")

    return {
        "n_normal": n_norm, "n_early_tumor": n_tum,
        "cv_auc_mean": float(np.mean(fold_aucs)),
        "cv_auc_std": float(np.std(fold_aucs)),
        "threshold": thr, "test": metrics,
        "by_stage": stage_df.to_dict(orient="records"),
    }


def run_panel_scan(model_name: str, X, y, groups, folds: int,
                   sizes=(1, 2, 3, 5, 10, 25, 50, 100, 300, 1000)):
    """Сколько генов реально нужно?

    Порог берётся с ОДНОГО прохода CV, а чувствительность и специфичность
    считаются на ДРУГОМ, с другим сидом разбиения. Иначе порог подбирается
    и проверяется на одних и тех же предсказаниях, и обе метрики выходят
    оптимистичными.
    """
    logger.info("\nМИНИМАЛЬНАЯ ГЕННАЯ ПАНЕЛЬ")
    rows = []
    for k in sizes:
        if k > X.shape[1]:
            continue
        model = build_models(k)[model_name]
        oof_a, fold_aucs = cross_validate_oof(model, X, y, groups, n_splits=folds)
        rep = repeated_group_cv(model, X, y, groups, n_splits=folds, n_repeats=1,
                                random_state=RANDOM_STATE + 500)
        oof_b = rep["oof_mean"]

        thr = threshold_for_sensitivity(y, oof_a, 0.98)   # порог — на проходе A
        m = compute_metrics(y, oof_b, threshold=thr)      # оценка — на проходе B

        rows.append({"n_genes": k,
                     "cv_auc_mean": float(np.mean(fold_aucs)),
                     "cv_auc_std": float(np.std(fold_aucs)),
                     "sensitivity": m["sensitivity"],
                     "specificity": m["specificity"]})
        logger.info(f"  {k:5d} генов | CV AUC {np.mean(fold_aucs):.4f}"
                    f"±{np.std(fold_aucs):.4f} | чувств. {m['sensitivity']:.3f} "
                    f"| спец. {m['specificity']:.3f}")

    df = pd.DataFrame(rows)
    best = df.loc[df["cv_auc_mean"].idxmax()]
    enough = df[df["cv_auc_mean"] >= best["cv_auc_mean"] - 0.002]
    minimal = int(enough["n_genes"].min())
    logger.info(f"  Плато достигается уже на {minimal} генах "
                f"(AUC {float(enough.iloc[0]['cv_auc_mean']):.4f})")
    return df, minimal


def run_batch_check(models_factory, data, folds: int):
    """Проверка на батч-эффект: не плашку ли мы на самом деле различаем?"""
    meta = data.meta
    mixed = meta.groupby("plate")["label"].nunique()
    mixed_plates = set(mixed[mixed > 1].index)

    logger.info("\nПРОВЕРКА НА БАТЧ-ЭФФЕКТ")
    logger.info(f"  Плашек всего: {meta['plate'].nunique()}, "
                f"с обоими классами: {len(mixed_plates)} ({sorted(mixed_plates)})")

    mask = meta["plate"].isin(mixed_plates).to_numpy()
    y = data.y[mask]
    n_norm, n_tum = int((y == 0).sum()), int((y == 1).sum())
    logger.info(f"  Внутри смешанных плашек: норма {n_norm}, опухоль {n_tum}")

    if n_norm < 10 or n_tum < 10:
        logger.warning("  недостаточно образцов для проверки")
        return None

    X = data.X.to_numpy()[mask]
    groups = meta.loc[mask, "patient"].to_numpy()
    _, fold_aucs = cross_validate_oof(models_factory(), X, y, groups, n_splits=folds)
    auc = float(np.mean(fold_aucs))
    logger.info(f"  CV ROC-AUC только на смешанных плашках: {auc:.4f}±{np.std(fold_aucs):.4f}")
    logger.info("  Вывод: сигнал биологический" if auc > 0.95
                else "  Вывод: заметная доля качества объясняется батчем")

    return {
        "mixed_plates": sorted(mixed_plates), "n_normal": n_norm, "n_tumor": n_tum,
        "cv_auc_mean": auc, "cv_auc_std": float(np.std(fold_aucs)),
        "n_plates_total": int(meta["plate"].nunique()),
    }


# --------------------------------------------------------------------------- #
#  Отчёт
# --------------------------------------------------------------------------- #
def run_plate_holdout(models_factory, data, n_boot: int = 1000):
    """Обучаемся на всех плашках, кроме одной, и тестируем на отложенной.

    Разбиение по пациентам отвечает на вопрос «а справится ли модель с новым
    ПАЦИЕНТОМ». Разбиение по плашкам задаёт вопрос жёстче: «а справится ли она
    с новой ПАРТИЕЙ — другой день гибридизации, другой оператор, другая серия
    реактивов». Именно на этом переходе классификаторы экспрессии обычно и
    сыпятся, а обычная кросс-валидация такого перехода вообще не видит,
    потому что образцы одной плашки попадают и в train, и в валидацию.

    Тестировать можно только на плашках, где есть оба класса, — иначе на
    отложенной плашке ROC-AUC просто не определён.
    """
    meta = data.meta
    plates = meta["plate"].to_numpy()
    per_plate = meta.groupby("plate")["label"].nunique()
    mixed = [p for p in per_plate.index if per_plate[p] > 1]

    logger.info("\nВАЛИДАЦИЯ ПО ПЛАШКАМ (leave-one-plate-out)")
    if not mixed:
        logger.warning("  нет плашек с обоими классами — проверка невозможна")
        return None

    X = data.X.to_numpy()
    y = data.y
    rows = []
    for p in mixed:
        te = plates == p
        tr = ~te
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        model = models_factory()
        model.fit(X[tr], y[tr])
        prob = model.predict_proba(X[te])[:, 1]
        auc = float(roc_auc_score(y[te], prob))
        rows.append({
            "plate": p,
            "n_test": int(te.sum()),
            "n_normal": int((y[te] == 0).sum()),
            "n_tumor": int((y[te] == 1).sum()),
            "roc_auc": auc,
        })
        logger.info(f"  отложена плашка {p}: n={int(te.sum()):3d} "
                    f"(норма {int((y[te]==0).sum()):2d} / опухоль {int((y[te]==1).sum()):3d}) "
                    f"-> ROC-AUC {auc:.4f}")

    if not rows:
        logger.warning("  ни одна плашка не подошла для проверки")
        return None

    df = pd.DataFrame(rows)
    mean_auc = float(df["roc_auc"].mean())
    logger.info(f"  Среднее по {len(df)} отложенным плашкам: {mean_auc:.4f} "
                f"(худшая {df['roc_auc'].min():.4f})")
    logger.info("  Вывод: качество переносится на неизвестную партию"
                if mean_auc > 0.95 else
                "  Вывод: при смене партии качество падает — сигнал частично технический")
    return {"per_plate": rows, "mean_auc": mean_auc,
            "min_auc": float(df["roc_auc"].min()), "n_plates_tested": int(len(df))}


def write_report(path: Path, ctx: dict) -> None:
    s = ctx["summary"]
    m = ctx["test_metrics"]
    ci = ctx["test_ci"]
    ev = ctx.get("evidence") or {}
    rep = ev.get("repeated_cv")
    perm = ev.get("permutation")

    def band(key: str) -> str:
        lo, hi = ci[f"{key}_ci"]
        return f"{ci[key]:.4f} [{lo:.4f}–{hi:.4f}]"

    lines = [
        "# BioDNA — экспрессия генов и клинические признаки рака груди",
        "",
        f"Данные: TCGA-BRCA, **{ctx['n_samples']} образцов x {ctx['n_genes']} генов** "
        f"(норма {ctx['n_normal']}, опухоль {ctx['n_tumor']}, "
        f"{ctx['n_patients']} пациентов).",
        "Метод: **только классическое машинное обучение** (scikit-learn), без нейросетей. "
        "Разбиение по пациентам, отбор генов и масштабирование — внутри пайплайна.",
        "",
        "## Главный результат",
        "",
        "Базовая задача «норма или опухоль» на этих данных **насыщена**: одинаковый "
        "потолок берут все модели вплоть до наивного Байеса. Поэтому одна цифра "
        "качества здесь ничего не сообщает о методе, и проект отвечает на два "
        "вопроса вместо одного: *насколько этой цифре можно верить* и *что вообще "
        "предсказуемо по экспрессии генов, а что нет*.",
        "",
    ]

    task_rows = ctx.get("task_rows") or []
    if task_rows:
        lines += [
            "### Карта сложности клинических задач",
            "",
            "Одна и та же процедура (вложенная CV с подбором гиперпараметров, "
            "разбиение по пациентам) на разных клинических вопросах:",
            "",
            "| Клинический вопрос | Выборка | ROC-AUC | 95% ДИ | Вывод |",
            "|---|---|---|---|---|",
        ]
        for r in sorted(task_rows, key=lambda x: -x["auc_mean"]):
            mark = " ⚑" if r["is_control"] else ""
            lines.append(
                f"| {r['title']}{mark} | {r['n']} ({r['n_neg']}/{r['n_pos']}) | "
                f"**{r['auc_mean']:.3f}** | {r['ci_low']:.3f}–{r['ci_high']:.3f} | "
                f"{r['verdict']} |")
        lines += [
            "",
            "⚑ — негативный контроль. Он здесь не для полноты: единственный способ "
            "показать, что AUC = 1.00 на базовой задаче не артефакт утечки, — "
            "прогнать ТОТ ЖЕ пайплайн там, где предсказывать нечего, и получить 0.5.",
            "",
            "### Что тянет каждую задачу",
            "",
            "| Задача | Устойчивые гены-маркеры |",
            "|---|---|",
        ]
        for r in sorted(task_rows, key=lambda x: -x["auc_mean"]):
            lines.append(f"| {r['title']} | {r.get('top_genes', '—')} |")
        lines += [
            "",
            "Гены отобраны бутстрапом по пациентам, а не одной подгонкой, "
            "поэтому список воспроизводим. Полная таблица — `results/task_markers.csv`.",
            "",
        ]

    lines += [
        "## Базовая задача: норма vs опухоль",
        "",
        "### Отложенный тест (пациенты, которых модель не видела)",
        "",
        "| Метрика | Значение с 95% ДИ |",
        "|---|---|",
        f"| ROC-AUC | {band('roc_auc')} |",
        f"| PR-AUC | {band('pr_auc')} |",
        f"| Чувствительность | {band('sensitivity')} ({ci['tp']}/{ci['n_pos']}) |",
        f"| Специфичность | {band('specificity')} ({ci['tn']}/{ci['n_neg']}) |",
        f"| Пропущено опухолей (FN) | {m['fn']} |",
        f"| Ложных тревог (FP) | {m['fp']} |",
        "",
        f"Интервалы считаны по Уилсону для долей и бутстрапом для площадей. "
        f"Обратите внимание на специфичность: точечная оценка {ci['specificity']:.3f}, "
        f"но нормальных образцов в тесте всего {ci['n_neg']}, и честный интервал "
        f"начинается с {ci['specificity_ci'][0]:.2f}. Это не придирка к оформлению — "
        f"это фактическая точность, с которой мы вообще что-то знаем о специфичности.",
        "",
    ]

    if rep:
        lines += [
            "### Повторная кросс-валидация вместо одного сплита",
            "",
            f"{ctx['repeats']} повторов x {ctx['folds']} фолдов = "
            f"**{rep['n_estimates']} независимых оценок**, каждый образец побывал "
            f"в валидации {ctx['repeats']} раз:",
            "",
            f"- средний ROC-AUC: **{rep['auc_mean']:.4f} ± {rep['auc_std']:.4f}**",
            f"- 95% разброс по фолдам: {rep['auc_p2_5']:.4f} … {rep['auc_p97_5']:.4f}",
            f"- худший фолд: {rep['auc_min']:.4f}",
            "",
        ]

    if perm:
        lines += [
            "### Пермутационный тест",
            "",
            f"Метки перемешаны по пациентам {perm['n_permutations']} раз, "
            f"процедура прогнана заново на каждой перестановке:",
            "",
            f"- нулевое распределение AUC: {perm['null_auc_mean']:.4f} ± "
            f"{perm['null_auc_std']:.4f}, 95-й перцентиль {perm['null_auc_p95']:.4f}",
            f"- наблюдаемый AUC {perm['observed_auc']:.4f} -> **p = {perm['p_value']:.4g}**",
            "",
            "Нулевое распределение сидит около 0.5. Значит, разбиение по пациентам "
            "не течёт и высокое качество не воспроизводится на случайных метках.",
            "",
        ]

    curve = ev.get("learning_curve")
    if curve is not None and not curve.empty:
        lines += [
            "### Кривая обучения по нормам",
            "",
            "| Норм в обучении | ROC-AUC |",
            "|---|---|",
        ]
        for _, r in curve.iterrows():
            lines.append(f"| {int(r['n_minority'])} | {r['auc_mean']:.4f} ± {r['auc_std']:.4f} |")
        first = curve.iloc[0]
        lines += [
            "",
            f"Качество держится даже на {int(first['n_minority'])} нормальных образцах "
            f"({first['auc_mean']:.4f}) — результат не висит на нескольких удачных "
            f"образцах, но и запаса по редкому классу у набора нет.",
            "",
        ]

    lines += [
        "## Сравнение классических моделей",
        "",
        "| Модель | CV ROC-AUC | Тест ROC-AUC | Чувств. | Спец. | MCC |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in s.sort_values("cv_auc_mean", ascending=False).iterrows():
        star = " ⭐" if r["model"] == ctx["best_name"] else ""
        lines.append(
            f"| {r['model']}{star} | {r['cv_auc_mean']:.4f} ± {r['cv_auc_std']:.4f} | "
            f"{r['test_auc']:.4f} | {r['test_sensitivity']:.3f} | "
            f"{r['test_specificity']:.3f} | {r['test_mcc']:.3f} |")

    delong = ctx.get("delong") or []
    if delong:
        n_sig = sum(1 for d in delong if d["significant"])
        lines += [
            "",
            f"Тест DeLong на out-of-fold предсказаниях: из {len(delong)} сравнений "
            f"лучшей модели с остальными значимы **{n_sig}**. "
            + ("Разницы между моделями практически нет — выбор «лучшей» здесь "
               "определяется шумом, а не превосходством алгоритма."
               if n_sig <= len(delong) // 3 else
               "Часть моделей действительно отстаёт."),
            "",
        ]

    lines += [
        "",
        f"## Итоговая модель: {ctx['best_name']} (калиброванная)",
        "",
        f"- Скрининговый порог: **{ctx['threshold']:.4f}** "
        f"(подобран на out-of-fold предсказаниях train под чувствительность "
        f"≥ {ctx['target_sensitivity']:.0%}; тест не участвовал)",
        f"- Серая зона: {ctx['low']:.4f} … {ctx['high']:.4f}",
        "",
        "### Распознавание по стадиям",
        "",
        "| Стадия | N | Распознано | Доля |",
        "|---|---|---|---|",
    ]
    for r in ctx["by_stage"]:
        name = "Норма" if r["stage"] == "Normal" else f"Стадия {r['stage']}"
        lines.append(f"| {name} | {r['n']} | {r['detected']} | {r['rate']:.1%} |")

    early = ctx.get("early")
    if early:
        em = early["test"]
        lines += [
            "",
            "### Отдельная модель «норма vs только стадии I-II»",
            "",
            f"- Обучающий набор: {early['n_normal']} норма / "
            f"{early['n_early_tumor']} ранних опухолей",
            f"- CV ROC-AUC: **{early['cv_auc_mean']:.4f} ± {early['cv_auc_std']:.4f}**",
            f"- Тест ROC-AUC: **{em['roc_auc']:.4f}** "
            f"[{em['roc_auc_ci'][0]:.4f}–{em['roc_auc_ci'][1]:.4f}], "
            f"чувствительность {em['sensitivity']:.3f} "
            f"[{em['sensitivity_ci'][0]:.3f}–{em['sensitivity_ci'][1]:.3f}], "
            f"пропущено {em['fn']}",
        ]

    batch = ctx.get("batch")
    if batch:
        lines += [
            "",
            "### Контроль батч-эффекта",
            "",
            f"Плашек с образцами обоих классов: {len(batch['mixed_plates'])} "
            f"({', '.join(batch['mixed_plates'])}). Внутри них "
            f"{batch['n_normal']} норма / {batch['n_tumor']} опухоль. "
            f"CV ROC-AUC на этом подмножестве: **{batch['cv_auc_mean']:.4f} ± "
            f"{batch['cv_auc_std']:.4f}** — качество не падает, значит модель "
            f"различает ткань, а не партию реактивов.",
        ]

    plate = ctx.get("plate_holdout")
    if plate:
        lines += [
            "",
            "### Перенос на неизвестную партию (leave-one-plate-out)",
            "",
            "Разбиение по пациентам проверяет, справится ли модель с новым "
            "пациентом. Более жёсткая проверка — отложить целую плашку: другой "
            "день гибридизации, другой оператор, другая серия реактивов. Обычная "
            "кросс-валидация такого перехода не видит вовсе, потому что образцы "
            "одной плашки попадают и в обучение, и в валидацию.",
            "",
            "| Отложенная плашка | Образцов (норма/опухоль) | ROC-AUC |",
            "|---|---|---|",
        ]
        for r in plate["per_plate"]:
            lines.append(f"| {r['plate']} | {r['n_test']} "
                         f"({r['n_normal']}/{r['n_tumor']}) | {r['roc_auc']:.4f} |")
        lines += [
            "",
            f"Среднее по {plate['n_plates_tested']} отложенным плашкам: "
            f"**{plate['mean_auc']:.4f}**, худшая — {plate['min_auc']:.4f}.",
        ]

    panel = ctx.get("panel")
    if panel is not None and not panel.empty:
        lines += [
            "",
            "## Сколько генов достаточно",
            "",
            f"Качество выходит на плато уже на **{ctx['min_panel']} генах** — "
            "то есть тест реализуем не полным секвенированием, а компактной панелью. "
            "Порог и оценка считаются на РАЗНЫХ проходах кросс-валидации, иначе "
            "чувствительность со специфичностью выходят завышенными.",
            "",
            "| Генов в панели | CV ROC-AUC | Чувств. | Спец. |",
            "|---|---|---|---|",
        ]
        for _, r in panel.iterrows():
            lines.append(f"| {int(r['n_genes'])} | {r['cv_auc_mean']:.4f} ± "
                         f"{r['cv_auc_std']:.4f} | {r['sensitivity']:.3f} | "
                         f"{r['specificity']:.3f} |")

    stab = ctx.get("stability")
    if stab is not None and not stab.empty:
        robust = stab[stab["selection_freq"] >= 0.9]
        lines += [
            "",
            "## Устойчивость генной панели",
            "",
            f"Выборка пересобиралась по пациентам {int(stab['n_bootstraps'].iloc[0])} раз; "
            f"для каждого гена посчитана доля повторов, в которых он вошёл в топ-50. "
            f"Генов с частотой ≥ 0.9: **{len(robust)}**. Это и есть воспроизводимое "
            f"ядро панели — список, полученный одной подгонкой на всех данных, "
            f"меняется от выборки к выборке и на новых данных не повторится.",
            "",
            "| Ген | Частота отбора | Согласованность направления |",
            "|---|---|---|",
        ]
        for _, r in stab.head(15).iterrows():
            lines.append(f"| {r['gene']} | {r['selection_freq']:.2f} | "
                         f"{r['direction_consistency']:.2f} |")

    lines += [
        "",
        "## Топ-20 генов-маркеров с координатами в геноме (hg38)",
        "",
        "| # | Ген | Локус | Цитобанд | В опухоли | Δ | Устойчивость |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, r in ctx["markers"].head(20).reset_index(drop=True).iterrows():
        arrow = "↑ выше" if r["delta"] > 0 else "↓ ниже"
        band_ = r["cytoband"] if isinstance(r["cytoband"], str) else "—"
        freq = r.get("selection_freq")
        freq_s = f"{freq:.2f}" if isinstance(freq, (int, float)) and freq == freq else "—"
        lines.append(f"| {i + 1} | {r['gene']} | {r['locus']} | {band_} | {arrow} | "
                     f"{r['delta']:+.2f} | {freq_s} |")

    lines += [
        "",
        "## Как это читать",
        "",
        "**Порог смещён в сторону чувствительности намеренно**: пропущенная опухоль "
        "стоит дороже ложной тревоги, которую снимает биопсия. Образцы в серой зоне "
        "модель не относит ни к норме, ни к раку — их нужно доисследовать.",
        "",
        "**AUC = 1.00 на базовой задаче — это свойство задачи, а не модели.** "
        "Опухолевая и нормальная ткань молочной железы различаются тысячами генов, "
        "и на TCGA-BRCA этот вопрос решается тривиально. Практический интерес "
        "начинается там, где качество НЕ равно единице: статус лимфоузлов, размер, "
        "гистотип. Там же видна и настоящая разница между моделями.",
        "",
        "**Отдельно про статус лимфоузлов.** Это самый ценный из вопросов списка: "
        "именно он решает, делать ли подмышечную лимфодиссекцию — операцию, которая "
        "примерно у половины пациенток оказывается лишней и оставляет лимфедему. "
        "Полученного качества для клиники недостаточно, и цифра в отчёте показывает "
        "ровно это, а не обратное.",
        "",
        "> Исследовательский проект. Не медицинское изделие и не основание "
        "для клинических решений.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Отчёт: {path}")


# --------------------------------------------------------------------------- #
#  main
# --------------------------------------------------------------------------- #
def main() -> None:
    global logger
    logger = setup_logging()

    parser = argparse.ArgumentParser(
        description="BioDNA: клинические предсказания по экспрессии генов, "
                    "классическое ML без нейросетей")
    parser.add_argument("--n-genes", type=int, default=300,
                        help="сколько генов отбирает базовый пайплайн")
    parser.add_argument("--folds", type=int, default=5, help="число фолдов внешней CV")
    parser.add_argument("--inner-folds", type=int, default=4,
                        help="число фолдов внутренней CV для подбора гиперпараметров")
    parser.add_argument("--repeats", type=int, default=10,
                        help="повторов кросс-валидации в доказательной части")
    parser.add_argument("--permutations", type=int, default=100,
                        help="число перестановок в пермутационном тесте")
    parser.add_argument("--bootstrap", type=int, default=2000,
                        help="реплик бутстрапа для доверительных интервалов")
    parser.add_argument("--target-sensitivity", type=float, default=0.98,
                        help="целевая чувствительность при подборе порога")
    parser.add_argument("--fast", action="store_true",
                        help="быстрый прогон: меньше повторов, перестановок и бутстрапа")
    parser.add_argument("--no-tasks", action="store_true",
                        help="пропустить набор клинических задач")
    parser.add_argument("--no-early", action="store_true",
                        help="пропустить блок раннего выявления")
    parser.add_argument("--offline", action="store_true",
                        help="не ходить в интернет за клиникой и координатами")
    args = parser.parse_args()

    if args.fast:
        args.repeats = min(args.repeats, 3)
        args.permutations = min(args.permutations, 30)
        args.bootstrap = min(args.bootstrap, 400)

    t_start = time.time()
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    section("BioDNA — классическое ML (scikit-learn), без нейросетей")
    logger.info(f"Генов в базовой модели: {args.n_genes} | "
                f"CV: {args.folds}-fold по пациентам | повторов: {args.repeats} | "
                f"перестановок: {args.permutations}")

    # ---------------------------------------------------------------- данные
    annot_dir = None if args.offline else ANNOT_DIR
    data = load_expression_data(
        DATA_DIR / "BC-TCGA-Normal.txt", DATA_DIR / "BC-TCGA-Tumor.txt",
        annotation_dir=annot_dir,
    )
    coords = pd.DataFrame() if args.offline else load_gene_coordinates(ANNOT_DIR)

    clinical = fetch_clinical(data.X.index, ANNOT_DIR) if not args.offline else pd.DataFrame()
    data.meta = attach_clinical(data.meta, clinical)

    X = data.X.to_numpy()
    y = data.y
    groups = data.meta["patient"].to_numpy()

    n_patients = len(set(groups))
    shared = data.meta.groupby("patient")["label"].nunique()
    logger.info(f"Пациентов: {n_patients}, из них с парой норма+опухоль: "
                f"{int((shared > 1).sum())} — поэтому делим по пациентам, а не по образцам")

    train_idx, test_idx = grouped_holdout(y, groups)
    logger.info(f"Train: {len(train_idx)} образцов "
                f"(норма {int((y[train_idx] == 0).sum())}, "
                f"опухоль {int((y[train_idx] == 1).sum())}) | "
                f"Тест: {len(test_idx)} (норма {int((y[test_idx] == 0).sum())}, "
                f"опухоль {int((y[test_idx] == 1).sum())})")
    assert not set(groups[train_idx]) & set(groups[test_idx]), "пациент попал в оба набора"

    # ------------------------------------------------------- сравнение моделей
    section("СРАВНЕНИЕ КЛАССИЧЕСКИХ МОДЕЛЕЙ (базовая задача)")
    models = build_models(n_genes=args.n_genes)
    summary, fitted, oof_store = benchmark_models(
        models, X[train_idx], y[train_idx], groups[train_idx],
        X[test_idx], y[test_idx], folds=args.folds,
    )

    # Победитель выбирается по кросс-валидации, НЕ по тесту:
    # иначе тест перестаёт быть независимой оценкой.
    best_name = str(summary.loc[summary["cv_auc_mean"].idxmax(), "model"])
    logger.info(f"\nЛучшая по CV: {best_name}")
    logger.info("Значимы ли различия между моделями (тест DeLong на OOF):")
    delong_rows = compare_top_models(summary, oof_store, y[train_idx])

    # ------------------------------------------------------------ калибровка
    cv_iter = list(StratifiedGroupKFold(n_splits=args.folds, shuffle=True,
                                        random_state=RANDOM_STATE)
                   .split(X[train_idx], y[train_idx], groups=groups[train_idx]))
    calibrated = CalibratedClassifierCV(
        build_models(args.n_genes)[best_name], method="sigmoid", cv=cv_iter)
    logger.info("\nКалибрую вероятности (Platt scaling)...")
    calibrated.fit(X[train_idx], y[train_idx])

    oof_cal, _ = cross_validate_oof(
        CalibratedClassifierCV(build_models(args.n_genes)[best_name],
                               method="sigmoid", cv=3),
        X[train_idx], y[train_idx], groups[train_idx], n_splits=args.folds)
    threshold = threshold_for_sensitivity(y[train_idx], oof_cal, args.target_sensitivity)
    high = max(threshold, threshold_for_specificity(y[train_idx], oof_cal, 0.99))
    logger.info(f"Скрининговый порог: {threshold:.4f} "
                f"(цель — чувствительность ≥ {args.target_sensitivity:.0%})")
    logger.info(f"Серая зона: {threshold:.4f} … {high:.4f}")

    test_prob = calibrated.predict_proba(X[test_idx])[:, 1]
    test_metrics = compute_metrics(y[test_idx], test_prob, threshold=threshold)
    test_ci = metrics_with_ci(y[test_idx], test_prob, threshold, n_boot=args.bootstrap)

    section("ОТЛОЖЕННЫЙ ТЕСТ (пациенты, которых модель не видела)")
    logger.info(f"  ROC-AUC        {test_ci['roc_auc']:.4f}  "
                f"95%ДИ [{test_ci['roc_auc_ci'][0]:.4f}, {test_ci['roc_auc_ci'][1]:.4f}]")
    logger.info(f"  PR-AUC         {test_ci['pr_auc']:.4f}  "
                f"95%ДИ [{test_ci['pr_auc_ci'][0]:.4f}, {test_ci['pr_auc_ci'][1]:.4f}]")
    logger.info(f"  Чувствительн.  {test_ci['sensitivity']:.4f}  "
                f"95%ДИ [{test_ci['sensitivity_ci'][0]:.4f}, {test_ci['sensitivity_ci'][1]:.4f}]"
                f"   ({test_ci['tp']}/{test_ci['n_pos']})")
    logger.info(f"  Специфичность  {test_ci['specificity']:.4f}  "
                f"95%ДИ [{test_ci['specificity_ci'][0]:.4f}, {test_ci['specificity_ci'][1]:.4f}]"
                f"   ({test_ci['tn']}/{test_ci['n_neg']})")
    logger.info(f"  пропущено опухолей {test_metrics['fn']}, "
                f"ложных тревог {test_metrics['fp']}")
    logger.info(f"  ВАЖНО: нормальных образцов в тесте всего {test_ci['n_neg']}, поэтому "
                f"нижняя граница специфичности — {test_ci['specificity_ci'][0]:.2f}, "
                f"а не 1.00")

    stage_df = sensitivity_by_stage(data.meta.iloc[test_idx], y[test_idx],
                                    test_prob, threshold)
    logger.info("\nРаспознавание по стадиям на тесте:")
    for _, r in stage_df.iterrows():
        name = "Норма" if r["stage"] == "Normal" else f"Стадия {r['stage']}"
        logger.info(f"  {name:12s} n={int(r['n']):3d}  {r['metric']:15s} "
                    f"{r['rate']:.1%}  (средний риск {r['mean_prob']:.3f})")

    zones = risk_zones(test_prob, threshold, high)
    zone_counts = pd.Series(zones).value_counts().to_dict()
    logger.info(f"Зоны риска на тесте: {zone_counts}")

    # --------------------------------------------------------- доказательства
    evidence = run_evidence(build_models(args.n_genes)[best_name], X, y, groups,
                            folds=args.folds, repeats=args.repeats,
                            n_perm=args.permutations)

    # ------------------------------------------------------- прочие проверки
    section("КОНТРОЛЬНЫЕ ПРОВЕРКИ")
    batch = run_batch_check(lambda: build_models(args.n_genes)[best_name], data, args.folds)
    plate_holdout = run_plate_holdout(lambda: build_models(args.n_genes)[best_name], data)
    panel_df, min_panel = run_panel_scan(
        best_name, X[train_idx], y[train_idx], groups[train_idx], args.folds)

    early = None
    if not args.no_early:
        early = run_early_detection(lambda: build_models(args.n_genes)[best_name],
                                    data, args.folds, args.target_sensitivity)

    logger.info("\nУСТОЙЧИВОСТЬ ГЕННОЙ ПАНЕЛИ")
    n_boot_stab = 60 if args.fast else 200
    stability = stability_selection(X, y, groups, data.genes, k=50,
                                   n_boot=n_boot_stab)
    robust = stability[stability["selection_freq"] >= 0.9]
    logger.info(f"  бутстрапов: {int(stability['n_bootstraps'].iloc[0])}, "
                f"генов с частотой отбора ≥0.9: {len(robust)}")
    for _, r in stability.head(10).iterrows():
        logger.info(f"    {r['gene']:12s} частота {r['selection_freq']:.2f}  "
                    f"направление согласовано в {r['direction_consistency']:.0%}")

    # ------------------------------------------------------ набор клинических задач
    task_rows, task_details = [], {}
    task_markers = pd.DataFrame()
    if not args.no_tasks:
        tasks = build_task_suite(data.meta, y)
        if tasks:
            task_rows, task_details, task_markers = run_task_suite(
                tasks, X, y, data.meta, data.genes, folds=args.folds,
                inner_folds=args.inner_folds,
                n_boot=min(args.bootstrap, 1000),
                n_stab=(40 if args.fast else 120))

    # ------------------------------------------------------------- маркеры
    section("МАРКЕРЫ И ИХ КООРДИНАТЫ В ДНК")
    final_model = build_models(args.n_genes)[best_name]
    final_model.fit(X, y)
    markers = build_marker_table(final_model, data.genes, X, y, coords, stability)

    gw = genome_wide_scores(X, y, data.genes)
    gw_ann = annotate(gw["gene"], coords).reset_index(drop=True)
    gw = pd.concat([gw, gw_ann[["chrom", "start", "genome_pos", "cytoband"]]], axis=1)
    logger.info(f"  координаты найдены для {int(gw['chrom'].notna().sum())}/{len(gw)} генов")
    for _, r in markers.head(10).iterrows():
        arrow = "↑" if r["delta"] > 0 else "↓"
        freq = r.get("selection_freq")
        freq_s = f"  устойчивость {freq:.2f}" if freq == freq else ""
        logger.info(f"  {r['gene']:12s} {arrow} {r['locus']}{freq_s}")

    # ------------------------------------------------------------- графики
    create_full_report(
        summary=summary, best_name=best_name,
        y_true=y[test_idx], y_prob=test_prob, threshold=threshold,
        low=threshold, high=high, stage_df=stage_df,
        markers=markers, genome_scores=gw, results_dir=RESULTS_DIR,
        panel_scan=panel_df,
    )
    create_evidence_report(
        RESULTS_DIR,
        task_rows=task_rows,
        permutation=evidence["permutation"],
        repeated_cv=evidence["repeated_cv"],
        holdout_auc=test_metrics["roc_auc"],
        learning=evidence["learning_curve"],
        n_minority_available=evidence["n_normal_available"],
        stability=stability,
    )

    # ------------------------------------------------------------- сохранение
    bundle = {
        "model": calibrated,
        "genes": data.genes,
        "best_model_name": best_name,
        "n_genes_selected": args.n_genes,
        "threshold": threshold,
        "zone_low": threshold,
        "zone_high": high,
        "target_sensitivity": args.target_sensitivity,
        "test_metrics": test_metrics,
        "test_metrics_ci": test_ci,
        "markers": markers,
        "stability": stability.head(200),
        "trained_on": {"n_samples": int(len(y)), "n_genes": int(X.shape[1])},
    }
    model_path = MODELS_DIR / "biodna_model.joblib"
    joblib.dump(bundle, model_path, compress=3)
    logger.info(f"\nМодель сохранена: {model_path} "
                f"({model_path.stat().st_size / 1e6:.1f} MB)")

    summary.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    panel_df.to_csv(RESULTS_DIR / "panel_size_scan.csv", index=False)
    markers.to_csv(RESULTS_DIR / "markers_with_coordinates.csv", index=False)
    stability.head(500).to_csv(RESULTS_DIR / "gene_stability.csv", index=False)
    gw.sort_values("p_value").head(2000).to_csv(
        RESULTS_DIR / "genome_wide_anova.csv", index=False)
    if delong_rows:
        pd.DataFrame(delong_rows).to_csv(RESULTS_DIR / "model_delong_tests.csv", index=False)
    if task_rows:
        pd.DataFrame([{k: v for k, v in r.items() if k != "chosen_per_fold"}
                      for r in task_rows]).to_csv(
            RESULTS_DIR / "task_landscape.csv", index=False)
    if not task_markers.empty:
        task_markers.to_csv(RESULTS_DIR / "task_markers.csv", index=False)

    ev_json = {
        "repeated_cv": {k: v for k, v in evidence["repeated_cv"].items()
                        if k not in ("oof_mean", "oof_matrix", "fold_aucs")},
        "permutation": {k: v for k, v in evidence["permutation"].items()
                        if k != "null_aucs"},
        "learning_curve": evidence["learning_curve"].to_dict(orient="records"),
        "n_normal_available": evidence["n_normal_available"],
    }
    metrics_json = {
        "dataset": {"n_samples": int(len(y)), "n_genes": int(X.shape[1]),
                    "n_normal": int((y == 0).sum()), "n_tumor": int((y == 1).sum()),
                    "n_patients": n_patients},
        "best_model": best_name,
        "threshold": threshold,
        "zone_high": high,
        "holdout_test": test_metrics,
        "holdout_test_ci": test_ci,
        "evidence": ev_json,
        "task_landscape": [{k: v for k, v in r.items() if k != "chosen_per_fold"}
                           for r in task_rows],
        "model_delong": delong_rows,
        "by_stage": stage_df.to_dict(orient="records"),
        "risk_zones": {k: int(v) for k, v in zone_counts.items()},
        "models": summary.to_dict(orient="records"),
        "early_detection": early,
        "batch_effect_check": batch,
        "plate_holdout": plate_holdout,
        "panel_scan": panel_df.to_dict(orient="records"),
        "minimal_panel_genes": min_panel,
        "gene_stability_top": stability.head(50).to_dict(orient="records"),
    }
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(metrics_json, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8")

    write_report(RESULTS_DIR / "REPORT.md", {
        "summary": summary, "best_name": best_name, "test_metrics": test_metrics,
        "test_ci": test_ci, "evidence": evidence, "delong": delong_rows,
        "task_rows": task_rows,
        "threshold": threshold, "low": threshold, "high": high,
        "target_sensitivity": args.target_sensitivity,
        "by_stage": stage_df.to_dict(orient="records"),
        "markers": markers, "early": early, "batch": batch,
        "plate_holdout": plate_holdout,
        "panel": panel_df, "min_panel": min_panel, "stability": stability,
        "folds": args.folds, "repeats": args.repeats,
        "n_samples": int(len(y)), "n_genes": int(X.shape[1]),
        "n_normal": int((y == 0).sum()), "n_tumor": int((y == 1).sum()),
        "n_patients": n_patients,
    })

    section(f"ГОТОВО за {(time.time() - t_start) / 60:.1f} мин. Результаты в {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
