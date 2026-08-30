"""Загрузка и подготовка матрицы экспрессии TCGA-BRCA.

Важно: здесь НЕ делается ни отбор признаков, ни стандартизация.
Всё это живёт внутри sklearn-Pipeline и обучается только на train-фолде —
иначе информация о тесте протекает в отбор генов и метрики становятся
завышенными (классическая ошибка: SelectKBest на всей выборке -> AUC 1.000).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExpressionData:
    """Матрица экспрессии samples x genes плюс метаданные образцов."""
    X: pd.DataFrame          # index = баркод образца, columns = символы генов
    y: np.ndarray            # 0 = норма, 1 = опухоль
    genes: list[str]
    meta: pd.DataFrame       # barcode, patient, stage, stage_group, sample_type

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_genes(self) -> int:
        return self.X.shape[1]


def _read_matrix(path: str | Path) -> pd.DataFrame:
    """Файл TCGA: строки = гены, столбцы = образцы. Транспонируем в samples x genes."""
    df = pd.read_csv(path, sep="\t", index_col=0)
    df.index = df.index.astype(str).str.strip()
    df = df[~df.index.duplicated(keep="first")]
    return df.T.astype("float32")


def load_expression_data(
    normal_path: str | Path,
    tumor_path: str | Path,
    max_nan_fraction: float = 0.2,
    annotation_dir: str | Path | None = None,
) -> ExpressionData:
    """Читает норму и опухоль, склеивает, чистит, добавляет клинические стадии."""
    logger.info("Загрузка данных экспрессии генов...")

    normal = _read_matrix(normal_path)
    tumor = _read_matrix(tumor_path)
    logger.info(f"  Norma : {normal.shape[0]} образцов x {normal.shape[1]} генов")
    logger.info(f"  Tumor : {tumor.shape[0]} образцов x {tumor.shape[1]} генов")

    common = normal.columns.intersection(tumor.columns)
    if len(common) < normal.shape[1]:
        logger.info(f"  Общих генов: {len(common)}")

    X = pd.concat([normal[common], tumor[common]], axis=0)
    y = np.array([0] * len(normal) + [1] * len(tumor), dtype=np.int8)

    # Гены с большой долей пропусков и константы бесполезны и мешают отбору.
    nan_frac = X.isna().mean(axis=0)
    keep = nan_frac <= max_nan_fraction
    dropped_nan = int((~keep).sum())

    variance = X.loc[:, keep].var(axis=0, skipna=True)
    keep_var = variance > 1e-8
    dropped_const = int((~keep_var).sum())

    X = X.loc[:, keep][keep_var.index[keep_var]]
    if dropped_nan or dropped_const:
        logger.info(f"  Отброшено генов: {dropped_nan} по пропускам, "
                    f"{dropped_const} константных")

    meta = _build_meta(X.index, y, annotation_dir)

    logger.info(f"Итого: {X.shape[0]} образцов x {X.shape[1]} генов "
                f"(норма {int((y == 0).sum())}, опухоль {int((y == 1).sum())})")

    return ExpressionData(X=X, y=y, genes=list(X.columns), meta=meta)


def _build_meta(barcodes, y: np.ndarray, annotation_dir) -> pd.DataFrame:
    from src.clinical import fetch_stages, is_tumor_barcode, plate_id, sample_type_code

    if annotation_dir is not None:
        meta = fetch_stages(barcodes, annotation_dir)
    else:
        meta = pd.DataFrame(index=pd.Index(barcodes, name="barcode"))
        meta["patient"] = [b for b in barcodes]
        meta["stage"] = "Unknown"
        meta["stage_group"] = "unknown"

    meta["label"] = y
    meta["sample_type"] = [sample_type_code(b) for b in barcodes]
    meta["plate"] = [plate_id(b) for b in barcodes]

    # Норма не имеет стадии, даже если у пациента есть диагноз.
    meta.loc[meta["label"] == 0, ["stage", "stage_group"]] = ["Normal", "normal"]

    mismatch = sum(
        1 for b, lab in zip(barcodes, y)
        if is_tumor_barcode(b) is not None and int(is_tumor_barcode(b)) != int(lab)
    )
    if mismatch:
        logger.warning(f"  Баркод не совпадает с меткой у {mismatch} образцов")

    counts = meta.loc[meta["label"] == 1, "stage"].value_counts().to_dict()
    if counts:
        logger.info(f"  Стадии опухолей: {counts}")
    return meta


def early_stage_mask(meta: pd.DataFrame, include_unknown: bool = False) -> np.ndarray:
    """Маска образцов для задачи РАННЕГО выявления: норма + опухоли стадии I-II."""
    groups = meta["stage_group"].to_numpy()
    mask = (groups == "normal") | (groups == "early")
    if include_unknown:
        mask |= groups == "unknown"
    return mask


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    base = Path(__file__).parent.parent
    data = load_expression_data(
        base / "data" / "BC-TCGA-Normal.txt",
        base / "data" / "BC-TCGA-Tumor.txt",
        annotation_dir=base / "data" / "annotation",
    )
    print(data.X.shape, np.bincount(data.y))
    print(data.meta.head())
