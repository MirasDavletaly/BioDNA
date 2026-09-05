"""BioDNA v3 - выбор модели по переносимости, а не по внутренней кросс-валидации.

Зачем понадобился третий заход. В v2 итоговая модель выбиралась по обычной
кросс-валидации, и там все четырнадцать алгоритмов дали AUC 0.9997-1.0000:
победитель определился разницей в 0.0001, то есть шумом. При этом на чужой
когорте разброс качества был вполне реальным - например, ER-положительные
опухоли ловились на 81% против 92.9% у трижды-негативных. Значит выбирать надо
не по тому, кто лучше на своих данных, а по тому, кто не разваливается на
чужих.

Протокол: leave-one-cohort-out. Обучаемся на всех источниках, кроме одного, и
проверяемся на отложенной когорте целиком. Это прямая модель ситуации «алгоритм
приехал в больницу, которой не было в обучении»: в отличие от случайного
сплита, где train и test приходят из одних и тех же лабораторий, здесь чужой
всегда именно тот, кого не видели.

Правило выбора объявлено заранее и на внешние данные не смотрит:
  1. Отбор по ХУДШЕЙ когорте (минимакс) - модель обязана работать везде,
     а не в среднем.
  2. Ничья разрешается по средней AUC.
  3. У финалистов дополнительно проверяется перенос порога.

Спектр прогрессии (норма -> неоплазия -> DCIS -> инвазия) в обучении и выборе
не участвует вообще и остаётся полностью нетронутой проверкой.

Запуск:  python train_loco.py
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
RECOUNT_DIR = BASE_DIR / "data" / "recount3"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results_loco"

sys.path.insert(0, str(BASE_DIR))

from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from src.cohorts import build_pooled_dataset, normalize  # noqa: E402
from src.evaluation import metrics_with_ci, wilson_ci  # noqa: E402
from src.viz_loco import (  # noqa: E402
    plot_comparison,
    plot_progression,
    plot_sweep,
)
from src.models import (  # noqa: E402
    RANDOM_STATE,
    build_models,
    compute_metrics,
    cross_validate_oof,
    stability_selection,
    threshold_for_sensitivity,
)

logger = logging.getLogger("biodna3")

# Когорты, каждая из которых по очереди становится отложенной. У GTEx нет
# опухолей, поэтому отдельной тестовой она быть не может и всегда лежит в
# обучении - как источник заведомо здоровой нормы.
TEST_COHORTS = ("TCGA", "FUSCC", "VARLEY")

NORMALIZATIONS = ("logtpm", "zsample", "rank")

# В переборе только те алгоритмы, что успевают отработать сотни раз. Тяжёлые
# ансамбли проверяются отдельно на финалистах.
SWEEP_MODELS = (
    "Logistic Regression (L2)",
    "Logistic Regression (L1)",
    "Elastic Net",
    "LDA (Ledoit-Wolf)",
    "PLS-DA",
    "Shrunken Centroids (PAM)",
    "SVM (RBF)",
    "k-NN (k=5)",
    "Random Forest",
)

# Конфигурация, которую выбрала v2 по внутренней кросс-валидации. Нужна как
# точка отсчёта: показать, что новый критерий действительно что-то меняет.
V2_CONFIG = {"model": "k-NN (k=5)", "normalization": "zsample", "n_genes": 50}


def setup_logging(overwrite: bool = True) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(RESULTS_DIR / "training_loco.log",
                                mode="w" if overwrite else "a", encoding="utf-8"),
        ],
        force=True,
    )


def loco_auc(data, model: str, normalization: str, n_genes: int) -> dict:
    """AUC на каждой отложенной когорте. Порог не участвует - только ранжирование."""
    X = normalize(data.X, normalization).to_numpy(dtype=np.float32)
    meta = data.meta
    y = meta["label"].to_numpy(dtype=np.int8)
    cohort = meta["cohort"].to_numpy()

    per_cohort = {}
    for held in TEST_COHORTS:
        te = cohort == held
        tr = ~te
        if len(np.unique(y[te])) < 2:
            continue
        est = build_models(n_genes)[model]
        est.fit(X[tr], y[tr])
        per_cohort[held] = float(roc_auc_score(y[te], est.predict_proba(X[te])[:, 1]))

    aucs = list(per_cohort.values())
    return {"model": model, "normalization": normalization, "n_genes": n_genes,
            "worst_auc": float(min(aucs)), "mean_auc": float(np.mean(aucs)),
            **{f"auc_{k}": v for k, v in per_cohort.items()}}


def loco_full(data, model: str, normalization: str, n_genes: int, folds: int,
              target_sens: float) -> dict:
    """Полная проверка: порог подбирается на обучающих когортах и переносится.

    Именно так модель и применялась бы на практике - настройки фиксируются до
    того, как увидены новые данные, и не подкручиваются под них.
    """
    X = normalize(data.X, normalization).to_numpy(dtype=np.float32)
    meta = data.meta
    y = meta["label"].to_numpy(dtype=np.int8)
    cohort = meta["cohort"].to_numpy()
    groups = meta["patient"].to_numpy()

    per_cohort = {}
    for held in TEST_COHORTS:
        te = cohort == held
        tr = ~te
        if len(np.unique(y[te])) < 2:
            continue

        est = build_models(n_genes)[model]
        oof, _ = cross_validate_oof(est, X[tr], y[tr], groups[tr], n_splits=folds)
        thr = threshold_for_sensitivity(y[tr], oof, target_sens)

        est.fit(X[tr], y[tr])
        prob = est.predict_proba(X[te])[:, 1]
        m = compute_metrics(y[te], prob, threshold=thr)
        m["threshold"] = thr
        m["balanced_accuracy"] = float(balanced_accuracy_score(
            y[te], (prob >= thr).astype(int)))
        per_cohort[held] = m

        logger.info(f"    {held:7s} AUC {m['roc_auc']:.4f} | чувств. "
                    f"{m['sensitivity']:.3f} | спец. {m['specificity']:.3f} | "
                    f"сбаланс. {m['balanced_accuracy']:.3f} | порог {thr:.3f}")

    bal = [m["balanced_accuracy"] for m in per_cohort.values()]
    aucs = [m["roc_auc"] for m in per_cohort.values()]
    return {"model": model, "normalization": normalization, "n_genes": n_genes,
            "worst_balanced_accuracy": float(min(bal)),
            "mean_balanced_accuracy": float(np.mean(bal)),
            "worst_auc": float(min(aucs)), "mean_auc": float(np.mean(aucs)),
            "per_cohort": per_cohort}


def sweep(data, sizes, models=SWEEP_MODELS, norms=NORMALIZATIONS) -> pd.DataFrame:
    """Перебор алгоритм x нормировка x размер панели по критерию худшей когорты."""
    rows, total = [], len(models) * len(norms) * len(sizes)
    done = 0
    for name in models:
        for norm in norms:
            for k in sizes:
                t0 = time.time()
                try:
                    rows.append(loco_auc(data, name, norm, k))
                except Exception as exc:
                    logger.warning(f"  {name}/{norm}/{k}: пропущено ({exc})")
                    continue
                done += 1
                r = rows[-1]
                logger.info(f"  [{done:3d}/{total}] {name:26s} {norm:7s} "
                            f"{k:4d} генов | худшая {r['worst_auc']:.4f} | "
                            f"средняя {r['mean_auc']:.4f} | {time.time() - t0:.0f}s")
    return pd.DataFrame(rows)


def build_final(data, cfg: dict, folds: int, target_sens: float):
    """Финальная модель на ВСЕХ когортах.

    Отложенной выборки здесь уже нет - честной оценкой качества служит LOCO,
    посчитанный до этого. Порог берётся с out-of-fold предсказаний по пациентам.
    """
    X = normalize(data.X, cfg["normalization"]).to_numpy(dtype=np.float32)
    y = data.meta["label"].to_numpy(dtype=np.int8)
    groups = data.meta["patient"].to_numpy()

    cv_iter = list(StratifiedGroupKFold(n_splits=folds, shuffle=True,
                                        random_state=RANDOM_STATE)
                   .split(X, y, groups=groups))
    calibrated = CalibratedClassifierCV(
        build_models(cfg["n_genes"])[cfg["model"]], method="sigmoid", cv=cv_iter)
    calibrated.fit(X, y)

    oof, _ = cross_validate_oof(
        CalibratedClassifierCV(build_models(cfg["n_genes"])[cfg["model"]],
                               method="sigmoid", cv=3),
        X, y, groups, n_splits=folds)
    thr = threshold_for_sensitivity(y, oof, target_sens)
    logger.info(f"  порог по out-of-fold: {thr:.4f}")
    return calibrated, thr, X, y, groups


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BioDNA v3: выбор модели по переносимости между когортами")
    parser.add_argument("--panel-sizes", type=int, nargs="+", default=[25, 50, 100, 200])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--target-sensitivity", type=float, default=0.98)
    parser.add_argument("--finalists", type=int, default=5,
                        help="сколько лучших конфигураций проверить полностью")
    parser.add_argument("--stability-boot", type=int, default=150)
    parser.add_argument("--plots-only", action="store_true",
                        help="перерисовать графики из сохранённых результатов")
    args = parser.parse_args()

    setup_logging(overwrite=not args.plots_only)
    if args.plots_only:
        redraw_plots()
        return

    t0 = time.time()
    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    logger.info("=" * 78)
    logger.info("BioDNA v3 - leave-one-cohort-out: выбор по переносимости")
    logger.info("=" * 78)

    data = build_pooled_dataset(RECOUNT_DIR)
    comp = data.meta.groupby(["cohort", "group"]).size()
    logger.info(f"Состав:\n{comp.to_string()}")

    logger.info("\n[1] ПЕРЕБОР КОНФИГУРАЦИЙ ПО ХУДШЕЙ КОГОРТЕ")
    grid = sweep(data, args.panel_sizes)
    grid = grid.sort_values(["worst_auc", "mean_auc"], ascending=False)
    grid.to_csv(RESULTS_DIR / "loco_sweep.csv", index=False)

    logger.info("\n  Топ-10 по худшей когорте:")
    for _, r in grid.head(10).iterrows():
        logger.info(f"    {r['model']:26s} {r['normalization']:7s} "
                    f"{int(r['n_genes']):4d} | худшая {r['worst_auc']:.4f} | "
                    f"средняя {r['mean_auc']:.4f}")

    logger.info(f"\n[2] ПОЛНАЯ ПРОВЕРКА {args.finalists} ФИНАЛИСТОВ (с переносом порога)")
    finalists = []
    for _, r in grid.head(args.finalists).iterrows():
        cfg = {"model": r["model"], "normalization": r["normalization"],
               "n_genes": int(r["n_genes"])}
        logger.info(f"\n  {cfg['model']} / {cfg['normalization']} / {cfg['n_genes']} генов")
        finalists.append(loco_full(data, **cfg, folds=args.folds,
                                   target_sens=args.target_sensitivity))

    logger.info("\n[3] ТОЧКА ОТСЧЁТА: конфигурация, выбранная v2 по обычной CV")
    logger.info(f"  {V2_CONFIG['model']} / {V2_CONFIG['normalization']} / "
                f"{V2_CONFIG['n_genes']} генов")
    baseline = loco_full(data, **V2_CONFIG, folds=args.folds,
                         target_sens=args.target_sensitivity)

    best = max(finalists, key=lambda r: (r["worst_balanced_accuracy"], r["mean_auc"]))
    cfg = {"model": best["model"], "normalization": best["normalization"],
           "n_genes": best["n_genes"]}
    logger.info(f"\nВЫБРАНО: {cfg['model']} / {cfg['normalization']} / "
                f"{cfg['n_genes']} генов")
    logger.info(f"  худшая когорта: сбаланс. точность "
                f"{best['worst_balanced_accuracy']:.4f}, AUC {best['worst_auc']:.4f}")
    logger.info(f"  для сравнения v2: сбаланс. "
                f"{baseline['worst_balanced_accuracy']:.4f}, "
                f"AUC {baseline['worst_auc']:.4f}")

    logger.info("\n[4] ФИНАЛЬНАЯ МОДЕЛЬ НА ВСЕХ КОГОРТАХ")
    model, thr, X, y, groups = build_final(data, cfg, args.folds,
                                           args.target_sensitivity)

    logger.info("\n[5] СПЕКТР ПРОГРЕССИИ (в обучении и выборе не участвовал)")
    from src.cohorts import load_external
    Xp, mp = load_external("PROGRESSION", RECOUNT_DIR, data.tpm_genes,
                           data.X.columns, data.annotation)
    prob = model.predict_proba(
        normalize(Xp, cfg["normalization"]).to_numpy(dtype=np.float32))[:, 1]
    mp = mp.copy()
    mp["risk"] = prob
    mp["pred"] = (prob >= thr).astype(int)

    spectrum = []
    order = ["норма", "ранняя неоплазия", "DCIS (рак на месте)", "инвазивная карцинома"]
    for sub in order:
        d = mp[mp["subgroup"] == sub]
        if not len(d):
            continue
        spectrum.append({"subgroup": sub, "n": len(d),
                         "mean_risk": float(d["risk"].mean()),
                         "median_risk": float(d["risk"].median()),
                         "flagged_rate": float(d["pred"].mean())})
        logger.info(f"  {sub:24s} n={len(d):3d}  риск {d['risk'].mean():.3f} "
                    f"(медиана {d['risk'].median():.3f})  помечено "
                    f"{d['pred'].mean():.0%}")

    logger.info("\n[6] ГЕНЫ-МАРКЕРЫ")
    stab = stability_selection(X, y, groups, list(data.X.columns),
                               k=cfg["n_genes"], n_boot=args.stability_boot)
    ann = data.genes.reindex(stab["gene"])
    markers = stab.reset_index(drop=True).copy()
    for col in ["symbol", "chrom", "start", "end", "strand"]:
        markers[col] = ann[col].to_numpy()
    markers["locus"] = [f"{c}:{int(s):,}-{int(e):,} ({st})" if isinstance(c, str) else "-"
                        for c, s, e, st in zip(markers["chrom"], markers["start"],
                                               markers["end"], markers["strand"])]
    markers = markers.head(300)
    for _, r in markers.head(10).iterrows():
        logger.info(f"  {str(r['symbol']):12s} {r['locus']:36s} "
                    f"устойчивость {r['selection_freq']:.2f}")

    out = {
        "protocol": "leave-one-cohort-out",
        "test_cohorts": list(TEST_COHORTS),
        "composition": {f"{c}/{g}": int(n) for (c, g), n in comp.items()},
        "n_samples": int(data.X.shape[0]), "n_genes": int(data.X.shape[1]),
        "sweep_top": grid.head(20).to_dict(orient="records"),
        "finalists": [{k: v for k, v in f.items() if k != "per_cohort"}
                      | {"per_cohort": {c: {kk: vv for kk, vv in m.items()}
                                        for c, m in f["per_cohort"].items()}}
                      for f in finalists],
        "v2_baseline": {k: v for k, v in baseline.items() if k != "per_cohort"}
                       | {"per_cohort": baseline["per_cohort"]},
        "chosen": cfg,
        "final_threshold": float(thr),
        "progression_spectrum": spectrum,
    }
    (RESULTS_DIR / "metrics_loco.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    markers.to_csv(RESULTS_DIR / "markers_loco.csv", index=False, encoding="utf-8")
    mp.to_csv(RESULTS_DIR / "progression_predictions.csv", encoding="utf-8")

    joblib.dump({
        "model": model, "genes": list(data.X.columns),
        "gene_symbols": data.genes["symbol"].to_dict(),
        "normalization": cfg["normalization"], "best_model_name": cfg["model"],
        "n_genes_selected": cfg["n_genes"], "threshold": float(thr),
        "markers": markers, "loco": {k: v for k, v in best.items()
                                     if k != "per_cohort"},
        "source": "recount3 TCGA+GTEx+FUSCC+Varley (LOCO)",
    }, MODELS_DIR / "biodna_v3.joblib", compress=3)

    write_report(RESULTS_DIR / "REPORT_LOCO.md", out, finalists, baseline, markers)

    logger.info("\n[7] ГРАФИКИ")
    draw_plots(best, baseline, grid, cfg, out["progression_spectrum"], float(thr), mp)
    logger.info(f"\nМодель: {MODELS_DIR / 'biodna_v3.joblib'}")
    logger.info(f"Готово за {(time.time() - t0) / 60:.1f} мин. Результаты в {RESULTS_DIR}/")


def draw_plots(best, baseline, grid, cfg, spectrum, threshold, predictions) -> None:
    """Каждый график в своём try: час вычислений не должен гибнуть от опечатки."""
    for draw in [
        lambda: plot_comparison(best, baseline, RESULTS_DIR),
        lambda: plot_sweep(grid, cfg, RESULTS_DIR),
        lambda: plot_progression(spectrum, threshold, RESULTS_DIR, predictions),
    ]:
        try:
            draw()
        except Exception as exc:
            logger.warning(f"  график пропущен: {type(exc).__name__}: {exc}")


def redraw_plots() -> None:
    """Перерисовать графики из metrics_loco.json без переобучения."""
    src = RESULTS_DIR / "metrics_loco.json"
    if not src.exists():
        raise SystemExit(f"Нет {src} - сначала выполните полный прогон")
    out = json.loads(src.read_text(encoding="utf-8"))

    finalists = out["finalists"]
    best = max(finalists, key=lambda r: (r["worst_balanced_accuracy"], r["mean_auc"]))
    grid_path = RESULTS_DIR / "loco_sweep.csv"
    grid = pd.read_csv(grid_path) if grid_path.exists() else pd.DataFrame()

    pred_path = RESULTS_DIR / "progression_predictions.csv"
    preds = pd.read_csv(pred_path) if pred_path.exists() else None

    draw_plots(best, out["v2_baseline"], grid, out["chosen"],
               out["progression_spectrum"], out["final_threshold"], preds)


def write_report(path: Path, out: dict, finalists, baseline, markers) -> None:
    cfg = out["chosen"]
    best = max(finalists, key=lambda r: (r["worst_balanced_accuracy"], r["mean_auc"]))

    L = [
        "# BioDNA v3 - выбор модели по переносимости между когортами",
        "",
        "В v2 итоговая модель выбиралась обычной кросс-валидацией, и все "
        "четырнадцать алгоритмов дали там AUC 0.9997-1.0000. Победитель "
        "определился разницей 0.0001, то есть шумом. На чужих когортах разброс "
        "при этом был вполне реальный. Значит критерий выбора был не тот.",
        "",
        "## Протокол",
        "",
        "**Leave-one-cohort-out**: обучаемся на всех источниках, кроме одного, "
        "проверяемся на отложенной когорте целиком. Прямая модель ситуации "
        "«алгоритм приехал в больницу, которой не было в обучении».",
        "",
        f"Всего **{out['n_samples']} образцов x {out['n_genes']} генов** "
        f"из четырёх источников:",
        "",
        "| Когорта / группа | N |",
        "|---|---|",
    ]
    for k, v in out["composition"].items():
        L.append(f"| {k} | {v} |")

    L += [
        "",
        "У GTEx нет опухолей, поэтому отдельной тестовой когортой он быть не "
        "может и всегда остаётся в обучении как источник заведомо здоровой "
        "нормы. Отложенными по очереди становятся TCGA, FUSCC и Varley.",
        "",
        "Правило выбора объявлено заранее: **по худшей когорте**, ничья - по "
        "средней AUC. Модель обязана работать везде, а не в среднем. Спектр "
        "прогрессии в обучении и выборе не участвует вообще.",
        "",
        "## Что выбралось",
        "",
        f"**{cfg['model']}**, нормировка `{cfg['normalization']}`, панель "
        f"**{cfg['n_genes']} генов**.",
        "",
        "| Конфигурация | Худшая когорта | Средняя AUC |",
        "|---|---|---|",
        f"| **v3: {cfg['model']}, {cfg['normalization']}, {cfg['n_genes']}** | "
        f"сбаланс. **{best['worst_balanced_accuracy']:.4f}**, AUC "
        f"{best['worst_auc']:.4f} | {best['mean_auc']:.4f} |",
        f"| v2: {V2_CONFIG['model']}, {V2_CONFIG['normalization']}, "
        f"{V2_CONFIG['n_genes']} | сбаланс. "
        f"{baseline['worst_balanced_accuracy']:.4f}, AUC "
        f"{baseline['worst_auc']:.4f} | {baseline['mean_auc']:.4f} |",
        "",
        "### По каждой отложенной когорте",
        "",
        "| Когорта | ROC-AUC | Чувствительность | Специфичность | Сбаланс. точность |",
        "|---|---|---|---|---|",
    ]
    for c, m in best["per_cohort"].items():
        L.append(f"| {c} | {m['roc_auc']:.4f} | {m['sensitivity']:.3f} | "
                 f"{m['specificity']:.3f} | {m['balanced_accuracy']:.3f} |")

    L += ["", "Для сравнения, та же таблица для конфигурации v2:", "",
          "| Когорта | ROC-AUC | Чувствительность | Специфичность | Сбаланс. точность |",
          "|---|---|---|---|---|"]
    for c, m in baseline["per_cohort"].items():
        L.append(f"| {c} | {m['roc_auc']:.4f} | {m['sensitivity']:.3f} | "
                 f"{m['specificity']:.3f} | {m['balanced_accuracy']:.3f} |")

    L += ["", "## Спектр прогрессии", "",
          "Когорта SRP023262 не участвовала ни в обучении, ни в выборе "
          "конфигурации, ни в подборе порога.", "",
          "| Состояние | N | Средний риск | Медиана | Помечено как рак |",
          "|---|---|---|---|---|"]
    for r in out["progression_spectrum"]:
        L.append(f"| {r['subgroup']} | {r['n']} | {r['mean_risk']:.3f} | "
                 f"{r['median_risk']:.3f} | {r['flagged_rate']:.0%} |")

    L += ["", "## Гены-маркеры", "",
          "| Ген | Локус (hg38) | Устойчивость |", "|---|---|---|"]
    for _, r in markers.head(20).iterrows():
        L.append(f"| {r['symbol']} | {r['locus']} | {r['selection_freq']:.2f} |")

    L += [
        "",
        "## Оговорки",
        "",
        "Финальная модель обучена на всех четырёх когортах, поэтому отдельной "
        "отложенной выборки у неё уже нет. Честной оценкой служит LOCO выше: "
        "он посчитан до того, как когорты были объединены.",
        "",
        "Четыре источника - это всё ещё мало. Худшая когорта задаёт нижнюю "
        "оценку качества, но с тремя точками эта оценка сама по себе шумная.",
        "",
        "> Исследовательский проект. Не медицинское изделие и не основание для "
        "клинических решений.",
        "",
    ]
    path.write_text("\n".join(L), encoding="utf-8")
    logger.info(f"Отчёт: {path}")


if __name__ == "__main__":
    main()
