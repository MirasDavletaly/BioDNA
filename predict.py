"""BioDNA — предсказание по обученной модели.

Раньше здесь была демка, печатавшая «ДНК-последовательность» из экспрессии.
Теперь скрипт делает то, что заявлено: берёт сохранённую модель и выдаёт
оценку риска рака для реальных образцов.

Примеры:
    python predict.py --sample data/BC-TCGA-Tumor.txt --limit 5
    python predict.py --sample my_patient.tsv --explain
    python predict.py --self-test
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

DEFAULT_MODEL = BASE_DIR / "models" / "biodna_model.joblib"

ZONE_TEXT = {
    "low": "НИЗКИЙ РИСК — картина экспрессии соответствует норме",
    "borderline": "СЕРАЯ ЗОНА — однозначного вывода нет, нужно доисследование",
    "high": "ВЫСОКИЙ РИСК — профиль соответствует опухолевой ткани",
}


def load_bundle(path: str | Path = DEFAULT_MODEL) -> dict:
    # joblib.load выполняет pickle: грузим только файл, созданный train.py
    # локально в этом же проекте, сторонние .joblib сюда подавать нельзя.
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"Модель не найдена: {path}\nСначала запустите: python train.py")
    return joblib.load(path)


def read_samples(path: str | Path) -> pd.DataFrame:
    """Читает TSV в формате TCGA (строки = гены, столбцы = образцы).

    Если файл оказался транспонированным (образцы в строках), разворачивает сам.
    """
    df = pd.read_csv(path, sep="\t", index_col=0)
    df.index = df.index.astype(str).str.strip()
    # У матрицы «гены x образцы» индекс — символы генов, они не начинаются с TCGA-.
    looks_transposed = df.index.astype(str).str.startswith("TCGA-").mean() > 0.5
    return df if looks_transposed else df.T


def align_to_model(samples: pd.DataFrame, genes: list[str]) -> tuple[np.ndarray, int]:
    """Приводит произвольную таблицу к тому набору генов, на котором обучалась модель."""
    aligned = samples.reindex(columns=genes)
    missing = int(aligned.isna().all(axis=0).sum())
    return aligned.to_numpy(dtype=float), missing


def predict(bundle: dict, samples: pd.DataFrame) -> pd.DataFrame:
    X, missing = align_to_model(samples, bundle["genes"])
    if missing:
        print(f"[!] {missing} из {len(bundle['genes'])} генов модели отсутствуют "
              f"в файле — заполняются медианой обучающей выборки")

    prob = bundle["model"].predict_proba(X)[:, 1]
    low, high = bundle["zone_low"], bundle["zone_high"]

    zone = np.where(prob < low, "low", np.where(prob >= high, "high", "borderline"))
    return pd.DataFrame({
        "sample": samples.index,
        "risk": prob,
        "verdict": np.where(prob >= bundle["threshold"], "TUMOR", "NORMAL"),
        "zone": zone,
    }).set_index("sample")


def explain(bundle: dict, samples: pd.DataFrame, sample_id: str, top_n: int = 10) -> None:
    """Показывает, какие гены-маркеры отклонены у образца и где они лежат в ДНК."""
    markers = bundle["markers"]
    row = samples.loc[sample_id]

    print(f"\nВклад маркеров для {sample_id}:")
    header = f"  {'Ген':<12} {'локус (hg38)':<42} {'значение':>9} {'норма':>8}"
    print(f"{header} {'опухоль':>9}  тренд")
    shown = 0
    for _, m in markers.iterrows():
        if m["gene"] not in row.index or shown >= top_n:
            continue
        val = float(row[m["gene"]])
        # Ближе к какому центроиду лежит образец по этому гену.
        closer = "опухоль" if abs(val - m["mean_tumor"]) < abs(val - m["mean_normal"]) else "норма"
        locus = str(m["locus"])
        print(f"  {m['gene']:<12} {locus:<42} {val:>9.3f} "
              f"{m['mean_normal']:>8.3f} {m['mean_tumor']:>9.3f}  → {closer}")
        shown += 1


def self_test(bundle: dict) -> None:
    """Прогон по реальным файлам проекта: сколько норм и опухолей узнаётся."""
    print("Самопроверка на данных проекта (модель их частично видела при обучении):\n")
    for name, path, expected in [
        ("Норма", BASE_DIR / "data" / "BC-TCGA-Normal.txt", "NORMAL"),
        ("Опухоль", BASE_DIR / "data" / "BC-TCGA-Tumor.txt", "TUMOR"),
    ]:
        if not path.exists():
            print(f"  {name}: файл не найден — пропуск")
            continue
        res = predict(bundle, read_samples(path))
        hit = (res["verdict"] == expected).mean()
        print(f"  {name:8s} n={len(res):3d}  верно {hit:.1%}  "
              f"средний риск {res['risk'].mean():.3f}")
        print(f"           зоны: {res['zone'].value_counts().to_dict()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BioDNA: оценка риска рака груди по профилю экспрессии генов")
    parser.add_argument("--sample", type=str, help="TSV с экспрессией (гены x образцы)")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--limit", type=int, default=20, help="сколько образцов показать")
    parser.add_argument("--explain", action="store_true",
                        help="разобрать первый образец по генам-маркерам")
    parser.add_argument("--self-test", action="store_true",
                        help="прогнать модель по обоим файлам проекта")
    parser.add_argument("--out", type=str, help="сохранить результат в CSV")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    bundle = load_bundle(args.model)
    print(f"Модель: {bundle['best_model_name']} (калиброванная), "
          f"{bundle['n_genes_selected']} генов из {len(bundle['genes'])}")
    print(f"Порог: {bundle['threshold']:.4f} | серая зона до {bundle['zone_high']:.4f}")

    if args.self_test or not args.sample:
        self_test(bundle)
        if not args.sample:
            return

    samples = read_samples(args.sample)
    print(f"\nОбразцов в файле: {len(samples)}")
    result = predict(bundle, samples)

    print(f"\n{'Образец':<32} {'Риск':>7}  Вердикт   Зона")
    for sid, r in result.head(args.limit).iterrows():
        print(f"{str(sid)[:32]:<32} {r['risk']:>7.4f}  {r['verdict']:<8}  "
              f"{ZONE_TEXT[r['zone']]}")
    if len(result) > args.limit:
        print(f"... ещё {len(result) - args.limit} образцов")

    counts = result["verdict"].value_counts().to_dict()
    print(f"\nИтого: {counts}")

    if args.explain and len(result):
        explain(bundle, samples, str(result.index[0]))

    if args.out:
        result.to_csv(args.out, encoding="utf-8")
        print(f"\nСохранено: {args.out}")


if __name__ == "__main__":
    main()
