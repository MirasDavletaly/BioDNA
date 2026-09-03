"""Предсказание моделью v2 (обучена на GTEx-здоровых + TCGA, recount3).

Модель ждёт на входе покрытия генов в формате recount3 (gene_sums): строки —
идентификаторы Ensembl с версией, столбцы — образцы. Скрипт сам приводит их к
log2(TPM+1) по тому же набору генов и с той же нормировкой, что при обучении,
иначе шкалы разъедутся и предсказание будет бессмысленным.

Примеры:
    python predict_v2.py --self-test
    python predict_v2.py --sample data/recount3/sra.gene_sums.SRP157974.G026.gz
    python predict_v2.py --sample my_counts.tsv --explain --out risk.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from src.cohorts import (  # noqa: E402
    RECOUNT_ANNOTATION_NAME,
    load_annotation,
    normalize,
    to_log_tpm,
)

DEFAULT_MODEL = BASE_DIR / "models" / "biodna_v2.joblib"
RECOUNT_DIR = BASE_DIR / "data" / "recount3"

VERDICT = {0: "НОРМА — профиль соответствует ткани без опухоли",
           1: "ОПУХОЛЬ — профиль соответствует опухолевой ткани"}


def load_bundle(path: str | Path = DEFAULT_MODEL) -> dict:
    # joblib.load выполняет pickle: грузим только файл, созданный
    # train_cohorts.py локально в этом же проекте.
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"Модель не найдена: {path}\n"
                         f"Сначала запустите: python train_cohorts.py")
    return joblib.load(path)


def read_counts(path: str | Path) -> pd.DataFrame:
    """Читает gene_sums recount3 или совместимый TSV -> samples x genes."""
    header = pd.read_csv(path, sep="\t", comment="#", nrows=0, compression="infer")
    # dtype задаём поимённо: колонка gene_id строковая, float32 к ней неприменим.
    dtypes = {c: np.float32 for c in header.columns[1:]}
    df = pd.read_csv(path, sep="\t", comment="#", index_col=0, dtype=dtypes,
                     compression="infer")
    return df.T


def prepare(counts: pd.DataFrame, bundle: dict, annotation: pd.DataFrame) -> np.ndarray:
    """Покрытия -> log2(TPM+1) -> нормировка обучения -> матрица для модели."""
    # Знаменатель TPM должен совпадать с обучением: все protein-coding гены.
    tpm_genes = list(annotation.index[annotation["gene_type"] == "protein_coding"])

    aligned = counts.reindex(columns=tpm_genes, fill_value=0.0)
    missing = int((aligned.sum(axis=0) == 0).sum())
    if missing > len(tpm_genes) * 0.5:
        print(f"[!] {missing} из {len(tpm_genes)} генов отсутствуют или нулевые — "
              f"похоже, формат входа не recount3; результат ненадёжен")

    X = to_log_tpm(aligned, annotation.loc[tpm_genes, "bp_length"])
    X = X[list(bundle["genes"])]
    return normalize(X, bundle["normalization"]).to_numpy(dtype=np.float32)


def predict(bundle: dict, counts: pd.DataFrame, annotation: pd.DataFrame) -> pd.DataFrame:
    X = prepare(counts, bundle, annotation)
    prob = bundle["model"].predict_proba(X)[:, 1]
    pred = (prob >= bundle["threshold"]).astype(int)
    return pd.DataFrame({"sample": counts.index, "risk": prob,
                         "verdict": np.where(pred == 1, "TUMOR", "NORMAL")}
                        ).set_index("sample")


def explain(bundle: dict, top_n: int = 15) -> None:
    """Из чего состоит панель модели: гены, координаты, устойчивость."""
    markers = bundle.get("markers")
    if markers is None or not len(markers):
        print("В модели нет таблицы маркеров")
        return
    print(f"\nПанель модели ({bundle['n_genes_selected']} генов, показаны первые {top_n}):")
    print(f"  {'Ген':<12} {'локус (hg38)':<40} {'устойч.':>8}  в опухоли")
    for _, m in markers.head(top_n).iterrows():
        arrow = "выше" if m["delta"] > 0 else "ниже"
        print(f"  {str(m['symbol']):<12} {str(m['locus']):<40} "
              f"{m['selection_freq']:>8.2f}  {arrow}")
    print("\n  устойч. — доля пересборок выборки, в которых ген попал в панель")


def self_test(bundle: dict, annotation: pd.DataFrame) -> None:
    """Прогон по внешней когорте FUSCC — её модель при обучении не видела."""
    path = RECOUNT_DIR / "sra.gene_sums.SRP157974.G026.gz"
    meta_path = RECOUNT_DIR / "sra.sra.SRP157974.MD.gz"
    if not path.exists() or not meta_path.exists():
        print("Файлы внешней когорты не скачаны — запустите train_cohorts.py")
        return

    from src.cohorts import _read_meta

    meta = _read_meta(meta_path)

    def attr(s, key):
        if not isinstance(s, str):
            return None
        for part in s.split("|"):
            if part.startswith(key + ";;"):
                return part.split(";;", 1)[1]
        return None

    tissue = meta["sample_attributes"].map(lambda s: attr(s, "tissue"))
    truth = pd.Series(
        np.where(tissue == "Primary breast tumor tissue", "TUMOR", "NORMAL"),
        index=meta["external_id"].to_numpy())

    res = predict(bundle, read_counts(path), annotation)
    res["truth"] = truth.reindex(res.index)
    res = res.dropna(subset=["truth"])

    print("\nВнешняя когорта FUSCC (Шанхай) — модель её не видела при обучении:")
    for label in ["TUMOR", "NORMAL"]:
        sub = res[res["truth"] == label]
        if len(sub):
            print(f"  {label:7s} n={len(sub):4d}  верно {(sub['verdict'] == label).mean():.1%}"
                  f"  средний риск {sub['risk'].mean():.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BioDNA v2: оценка риска по покрытиям генов (формат recount3)")
    parser.add_argument("--sample", type=str, help="gene_sums .gz или TSV с покрытиями")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--explain", action="store_true", help="показать панель генов")
    parser.add_argument("--self-test", action="store_true",
                        help="прогон по внешней когорте FUSCC")
    parser.add_argument("--out", type=str, help="сохранить результат в CSV")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    bundle = load_bundle(args.model)
    annotation = load_annotation(RECOUNT_DIR / RECOUNT_ANNOTATION_NAME)

    print(f"Модель: {bundle['best_model_name']}, панель {bundle['n_genes_selected']} "
          f"генов, нормировка {bundle['normalization']}")
    print(f"Порог: {bundle['threshold']:.4f} | обучена на: {bundle['source']}")

    if args.explain:
        explain(bundle)

    if args.self_test or not args.sample:
        self_test(bundle, annotation)
        if not args.sample:
            return

    counts = read_counts(args.sample)
    print(f"\nОбразцов в файле: {len(counts)}")
    res = predict(bundle, counts, annotation)

    print(f"\n{'Образец':<40} {'Риск':>7}  Вердикт")
    for sid, r in res.head(args.limit).iterrows():
        print(f"{str(sid)[:40]:<40} {r['risk']:>7.4f}  {r['verdict']}")
    if len(res) > args.limit:
        print(f"... ещё {len(res) - args.limit} образцов")
    print(f"\nИтого: {res['verdict'].value_counts().to_dict()}")

    if args.out:
        res.to_csv(args.out, encoding="utf-8")
        print(f"\nСохранено: {args.out}")


if __name__ == "__main__":
    main()
