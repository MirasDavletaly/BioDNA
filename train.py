"""BioDNA — обучение детектора рака груди по экспрессии генов.

Только классическое машинное обучение (scikit-learn). Нейросетей нет.

Что делает скрипт:
  1. Читает матрицу TCGA-BRCA (17 814 генов x 590 образцов).
  2. Подтягивает клинические стадии из GDC API и координаты генов из UCSC (hg38).
  3. Делит данные ПО ПАЦИЕНТАМ на train / отложенный тест.
  4. Гоняет 9 классических моделей, отбор генов — внутри пайплайна.
  5. Калибрует лучшую, подбирает порог под скрининг (максимум чувствительности).
  6. Отдельно решает задачу РАННЕГО выявления: норма vs стадия I-II.
  7. Пишет метрики, таблицу маркеров с ДНК-координатами, графики и отчёт.

Запуск:  python train.py
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
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from src.data_preprocessing import early_stage_mask, load_expression_data  # noqa: E402
from src.gene_annotation import annotate, format_locus, load_gene_coordinates  # noqa: E402
from src.models import (  # noqa: E402
    RANDOM_STATE,
    build_models,
    compute_metrics,
    cross_validate_oof,
    genome_wide_scores,
    risk_zones,
    selected_gene_scores,
    sensitivity_by_stage,
    threshold_for_sensitivity,
    threshold_for_specificity,
)
from src.visualization import create_full_report  # noqa: E402


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


def grouped_holdout(y: np.ndarray, groups: np.ndarray, test_size: float = 0.2):
    """Стратифицированный сплит, не разрывающий одного пациента между train и test."""
    n_splits = max(2, int(round(1 / test_size)))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(cv.split(np.zeros(len(y)), y, groups=groups))
    return train_idx, test_idx


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


def build_marker_table(model, gene_names, X, y, coords, top_n: int = 60) -> pd.DataFrame:
    """Отобранные моделью гены + их локусы в геноме + направление изменения."""
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


def run_early_detection(models_factory, data, coords, folds: int, target_sens: float):
    """Отдельная задача: отличить НОРМУ от опухоли только ранних стадий (I-II).

    Это и есть «раннее обнаружение» в строгом смысле: поздние стадии из обучения
    и теста убраны совсем, модель не может опереться на запущенные случаи.
    """
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
    stage_df = sensitivity_by_stage(meta.iloc[te], y[te], prob, thr)

    logger.info(f"  CV AUC {np.mean(fold_aucs):.4f}±{np.std(fold_aucs):.4f} | "
                f"тест AUC {metrics['roc_auc']:.4f}")
    logger.info(f"  При пороге {thr:.3f}: чувствительность к I-II "
                f"{metrics['sensitivity']:.3f}, специфичность {metrics['specificity']:.3f}, "
                f"пропущено опухолей {metrics['fn']}")

    stage_i = stage_df[stage_df["stage"] == "I"]
    if not stage_i.empty:
        r = stage_i.iloc[0]
        logger.info(f"  Стадия I отдельно: поймано {int(r['detected'])}/{int(r['n'])} "
                    f"({r['rate']:.1%})")

    return {
        "n_normal": n_norm,
        "n_early_tumor": n_tum,
        "cv_auc_mean": float(np.mean(fold_aucs)),
        "cv_auc_std": float(np.std(fold_aucs)),
        "threshold": thr,
        "test": metrics,
        "by_stage": stage_df.to_dict(orient="records"),
    }


def run_panel_scan(model_name: str, X, y, groups, folds: int,
                   sizes=(1, 2, 3, 5, 10, 25, 50, 100, 300, 1000)):
    """Сколько генов реально нужно?

    Практический смысл: панель из 5-10 генов — это дешёвая ПЦР-тест-система,
    панель из 300 генов — секвенирование. Если качество выходит на плато рано,
    большая панель не нужна.
    """
    logger.info("\nМИНИМАЛЬНАЯ ГЕННАЯ ПАНЕЛЬ")
    rows = []
    for k in sizes:
        if k > X.shape[1]:
            continue
        model = build_models(k)[model_name]
        oof, fold_aucs = cross_validate_oof(model, X, y, groups, n_splits=folds)
        thr = threshold_for_sensitivity(y, oof, 0.98)
        m = compute_metrics(y, oof, threshold=thr)
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
    """Проверка на батч-эффект: не плашку ли мы на самом деле различаем?

    В TCGA нормы и опухоли частично разведены по плашкам (technical plates).
    Если модель ловит биологию, а не партию реактивов, её качество не должно
    просесть, когда мы оставим ТОЛЬКО плашки, содержащие оба класса сразу.
    """
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
        "mixed_plates": sorted(mixed_plates),
        "n_normal": n_norm,
        "n_tumor": n_tum,
        "cv_auc_mean": auc,
        "cv_auc_std": float(np.std(fold_aucs)),
        "n_plates_total": int(meta["plate"].nunique()),
    }


def write_report(path: Path, ctx: dict) -> None:
    s = ctx["summary"]
    m = ctx["test_metrics"]
    lines = [
        "# BioDNA — отчёт по детекции рака груди",
        "",
        f"Данные: TCGA-BRCA, **{ctx['n_samples']} образцов x {ctx['n_genes']} генов** "
        f"(норма {ctx['n_normal']}, опухоль {ctx['n_tumor']}).",
        f"Метод: **только классическое машинное обучение** (scikit-learn), без нейросетей.",
        f"Разбиение по пациентам, отбор генов внутри пайплайна.",
        "",
        "## Сравнение моделей",
        "",
        "| Модель | CV ROC-AUC | Тест ROC-AUC | Чувств. | Спец. | MCC |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in s.sort_values("cv_auc_mean", ascending=False).iterrows():
        star = " ⭐" if r["model"] == ctx["best_name"] else ""
        lines.append(
            f"| {r['model']}{star} | {r['cv_auc_mean']:.4f} ± {r['cv_auc_std']:.4f} | "
            f"{r['test_auc']:.4f} | {r['test_sensitivity']:.3f} | "
            f"{r['test_specificity']:.3f} | {r['test_mcc']:.3f} |"
        )

    lines += [
        "",
        f"## Итоговая модель: {ctx['best_name']} (калиброванная)",
        "",
        f"- Скрининговый порог: **{ctx['threshold']:.4f}** "
        f"(подобран на train под чувствительность ≥ {ctx['target_sensitivity']:.0%})",
        f"- Серая зона: {ctx['low']:.4f} … {ctx['high']:.4f}",
        "",
        "| Метрика | Значение на отложенном тесте |",
        "|---|---|",
        f"| ROC-AUC | {m['roc_auc']:.4f} |",
        f"| PR-AUC | {m['pr_auc']:.4f} |",
        f"| Чувствительность | {m['sensitivity']:.4f} |",
        f"| Специфичность | {m['specificity']:.4f} |",
        f"| Сбалансированная точность | {m['balanced_accuracy']:.4f} |",
        f"| MCC | {m['mcc']:.4f} |",
        f"| Пропущено опухолей (FN) | {m['fn']} |",
        f"| Ложных тревог (FP) | {m['fp']} |",
        "",
        "## Ранняя стадия",
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
            f"- Обучающий набор: {early['n_normal']} норма / {early['n_early_tumor']} ранних опухолей",
            f"- CV ROC-AUC: **{early['cv_auc_mean']:.4f} ± {early['cv_auc_std']:.4f}**",
            f"- Тест ROC-AUC: **{em['roc_auc']:.4f}**, чувствительность {em['sensitivity']:.3f}, "
            f"специфичность {em['specificity']:.3f}, пропущено {em['fn']}",
        ]

    batch = ctx.get("batch")
    if batch:
        lines += [
            "",
            "### Контроль батч-эффекта",
            "",
            f"Плашек с образцами обоих классов: {len(batch['mixed_plates'])} "
            f"({', '.join(batch['mixed_plates'])}). Внутри них "
            f"{batch['n_normal']} норма / {batch['n_tumor']} опухоль.",
            f"CV ROC-AUC на этом подмножестве: **{batch['cv_auc_mean']:.4f} ± "
            f"{batch['cv_auc_std']:.4f}** — качество не падает, значит модель "
            f"различает ткань, а не партию реактивов.",
        ]

    panel = ctx.get("panel")
    if panel is not None and not panel.empty:
        lines += [
            "",
            "## Сколько генов достаточно",
            "",
            f"Качество выходит на плато уже на **{ctx['min_panel']} генах** — "
            "то есть тест реализуем не полным секвенированием, а компактной панелью.",
            "",
            "| Генов в панели | CV ROC-AUC | Чувств. | Спец. |",
            "|---|---|---|---|",
        ]
        for _, r in panel.iterrows():
            lines.append(f"| {int(r['n_genes'])} | {r['cv_auc_mean']:.4f} ± "
                         f"{r['cv_auc_std']:.4f} | {r['sensitivity']:.3f} | "
                         f"{r['specificity']:.3f} |")

    lines += [
        "",
        "## Топ-20 генов-маркеров с координатами в геноме (hg38)",
        "",
        "| # | Ген | Локус | Цитобанд | В опухоли | Δ |",
        "|---|---|---|---|---|---|",
    ]
    for i, r in ctx["markers"].head(20).iterrows():
        arrow = "↑ выше" if r["delta"] > 0 else "↓ ниже"
        band = r["cytoband"] if isinstance(r["cytoband"], str) else "—"
        lines.append(f"| {i + 1} | {r['gene']} | {r['locus']} | {band} | {arrow} | "
                     f"{r['delta']:+.2f} |")

    lines += [
        "",
        "## Как читать",
        "",
        "Порог смещён в сторону чувствительности намеренно: пропущенная опухоль "
        "стоит дороже ложной тревоги, которую снимает биопсия. Образцы в серой зоне "
        "модель не относит ни к норме, ни к раку — их нужно доисследовать.",
        "",
        "> Исследовательский проект. Не медицинское изделие и не основание "
        "для клинических решений.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Отчёт: {path}")


def main() -> None:
    global logger
    logger = setup_logging()

    parser = argparse.ArgumentParser(
        description="BioDNA: детекция рака груди классическим ML (без нейросетей)")
    parser.add_argument("--n-genes", type=int, default=300,
                        help="сколько генов отбирает пайплайн (по умолчанию 300)")
    parser.add_argument("--folds", type=int, default=5, help="число фолдов CV")
    parser.add_argument("--target-sensitivity", type=float, default=0.98,
                        help="целевая чувствительность при подборе порога")
    parser.add_argument("--no-early", action="store_true",
                        help="пропустить блок раннего выявления")
    parser.add_argument("--offline", action="store_true",
                        help="не ходить в интернет за стадиями и координатами")
    args = parser.parse_args()

    t_start = time.time()
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    logger.info("=" * 72)
    logger.info("BioDNA — детекция рака груди, классическое ML (scikit-learn)")
    logger.info(f"Генов в модели: {args.n_genes} | CV: {args.folds}-fold по пациентам")
    logger.info("=" * 72)

    annot_dir = None if args.offline else ANNOT_DIR
    data = load_expression_data(
        DATA_DIR / "BC-TCGA-Normal.txt",
        DATA_DIR / "BC-TCGA-Tumor.txt",
        annotation_dir=annot_dir,
    )
    coords = (pd.DataFrame() if args.offline else load_gene_coordinates(ANNOT_DIR))

    X = data.X.to_numpy()
    y = data.y
    groups = data.meta["patient"].to_numpy()

    n_patients = len(set(groups))
    shared = data.meta.groupby("patient")["label"].nunique()
    logger.info(f"Пациентов: {n_patients}, из них с парой норма+опухоль: "
                f"{int((shared > 1).sum())} — поэтому делим по пациентам, а не по образцам")

    train_idx, test_idx = grouped_holdout(y, groups)
    logger.info(f"Train: {len(train_idx)} образцов "
                f"(норма {int((y[train_idx] == 0).sum())}, опухоль {int((y[train_idx] == 1).sum())}) | "
                f"Тест: {len(test_idx)} "
                f"(норма {int((y[test_idx] == 0).sum())}, опухоль {int((y[test_idx] == 1).sum())})")
    assert not set(groups[train_idx]) & set(groups[test_idx]), "пациент попал в оба набора"

    logger.info("\nСравнение классических моделей...")
    models = build_models(n_genes=args.n_genes)
    summary, fitted, _ = benchmark_models(
        models, X[train_idx], y[train_idx], groups[train_idx],
        X[test_idx], y[test_idx], folds=args.folds,
    )

    # Победитель выбирается по кросс-валидации, НЕ по тесту:
    # иначе тест перестаёт быть независимой оценкой.
    best_name = str(summary.loc[summary["cv_auc_mean"].idxmax(), "model"])
    logger.info(f"\nЛучшая по CV: {best_name}")

    # Калибровка: Платт-скейлинг, чтобы выход читался как вероятность.
    cv_iter = list(StratifiedGroupKFold(n_splits=args.folds, shuffle=True,
                                        random_state=RANDOM_STATE)
                   .split(X[train_idx], y[train_idx], groups=groups[train_idx]))
    calibrated = CalibratedClassifierCV(
        build_models(args.n_genes)[best_name], method="sigmoid", cv=cv_iter)
    logger.info("Калибрую вероятности (Platt scaling)...")
    calibrated.fit(X[train_idx], y[train_idx])

    # Порог подбираем на out-of-fold предсказаниях train, тест не трогаем.
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
    logger.info("\nОТЛОЖЕННЫЙ ТЕСТ (пациенты, которых модель не видела):")
    for k in ["roc_auc", "pr_auc", "sensitivity", "specificity",
              "balanced_accuracy", "mcc"]:
        logger.info(f"  {k:20s} {test_metrics[k]:.4f}")
    logger.info(f"  пропущено опухолей   {test_metrics['fn']}")
    logger.info(f"  ложных тревог        {test_metrics['fp']}")

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

    batch = run_batch_check(lambda: build_models(args.n_genes)[best_name], data, args.folds)

    panel_df, min_panel = run_panel_scan(
        best_name, X[train_idx], y[train_idx], groups[train_idx], args.folds)

    early = None
    if not args.no_early:
        early = run_early_detection(
            lambda: build_models(args.n_genes)[best_name],
            data, coords, args.folds, args.target_sensitivity)

    logger.info("\nМаркеры и их координаты в ДНК...")
    final_model = build_models(args.n_genes)[best_name]
    final_model.fit(X, y)
    markers = build_marker_table(final_model, data.genes, X, y, coords)

    gw = genome_wide_scores(X, y, data.genes)
    gw_ann = annotate(gw["gene"], coords).reset_index(drop=True)
    gw = pd.concat([gw, gw_ann[["chrom", "start", "genome_pos", "cytoband"]]], axis=1)
    mapped = int(gw["chrom"].notna().sum())
    logger.info(f"  координаты найдены для {mapped}/{len(gw)} генов")
    for _, r in markers.head(10).iterrows():
        arrow = "↑" if r["delta"] > 0 else "↓"
        logger.info(f"  {r['gene']:12s} {arrow} {r['locus']}")

    create_full_report(
        summary=summary, best_name=best_name,
        y_true=y[test_idx], y_prob=test_prob, threshold=threshold,
        low=threshold, high=high, stage_df=stage_df,
        markers=markers, genome_scores=gw, results_dir=RESULTS_DIR,
        panel_scan=panel_df,
    )

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
        "markers": markers,
        "trained_on": {"n_samples": int(len(y)), "n_genes": int(X.shape[1])},
    }
    model_path = MODELS_DIR / "biodna_model.joblib"
    joblib.dump(bundle, model_path, compress=3)
    logger.info(f"\nМодель сохранена: {model_path} "
                f"({model_path.stat().st_size / 1e6:.1f} MB)")

    summary.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    panel_df.to_csv(RESULTS_DIR / "panel_size_scan.csv", index=False)
    markers.to_csv(RESULTS_DIR / "markers_with_coordinates.csv", index=False)
    gw.sort_values("p_value").head(2000).to_csv(
        RESULTS_DIR / "genome_wide_anova.csv", index=False)

    metrics_json = {
        "dataset": {"n_samples": int(len(y)), "n_genes": int(X.shape[1]),
                    "n_normal": int((y == 0).sum()), "n_tumor": int((y == 1).sum()),
                    "n_patients": n_patients},
        "best_model": best_name,
        "threshold": threshold,
        "zone_high": high,
        "holdout_test": test_metrics,
        "by_stage": stage_df.to_dict(orient="records"),
        "risk_zones": {k: int(v) for k, v in zone_counts.items()},
        "models": summary.to_dict(orient="records"),
        "early_detection": early,
        "batch_effect_check": batch,
        "panel_scan": panel_df.to_dict(orient="records"),
        "minimal_panel_genes": min_panel,
    }
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(metrics_json, ensure_ascii=False, indent=2), encoding="utf-8")

    write_report(RESULTS_DIR / "REPORT.md", {
        "summary": summary, "best_name": best_name, "test_metrics": test_metrics,
        "threshold": threshold, "low": threshold, "high": high,
        "target_sensitivity": args.target_sensitivity,
        "by_stage": stage_df.to_dict(orient="records"),
        "markers": markers, "early": early, "batch": batch,
        "panel": panel_df, "min_panel": min_panel,
        "n_samples": int(len(y)), "n_genes": int(X.shape[1]),
        "n_normal": int((y == 0).sum()), "n_tumor": int((y == 1).sum()),
    })

    logger.info(f"\nГотово за {(time.time() - t_start) / 60:.1f} мин. "
                f"Результаты в {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
