"""Набор клинических задач, а не одна.

Почему так
----------
Исходная постановка «норма vs опухоль» на TCGA-BRCA решается идеально:
девять разных классических моделей дают ROC-AUC = 1.000, и разница между
логистической регрессией и градиентным бустингом — ноль. Улучшать там
нечего: задача насыщена, потолок достигнут, и любые «улучшения модели»
были бы шумом в четвёртом знаке.

Поэтому проект расширен на вопросы, которые врач реально задаёт ПОСЛЕ
того, как опухоль уже найдена, и на которых качество ещё далеко от
потолка — то есть где улучшение модели видно и измеримо:

  histology  протоковый рак или дольковый   — разное течение и тактика;
  size       T1 или T2+                     — размер решает объём операции;
  nodal      N0 или N+                      — поражение лимфоузлов решает,
                                              нужна ли подмышечная лимфодиссекция;
  stage      I-II или III-IV                — интегральная запущенность.

И одна задача-пустышка как НЕГАТИВНЫЙ КОНТРОЛЬ:

  vital      жив или умер                   — по одной лишь экспрессии
                                              первичной опухоли, без времени
                                              наблюдения, сигнала быть не должно.

Негативный контроль здесь не для красоты. Единственный честный способ
доказать, что AUC = 1.000 на основной задаче не артефакт утечки, — показать,
что ТОТ ЖЕ пайплайн на той же выборке даёт ~0.5 там, где предсказывать нечего.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """Одна бинарная задача: маска образцов, метки и человеческое описание."""
    key: str
    title: str
    question: str
    neg_label: str
    pos_label: str
    why: str
    labels: np.ndarray                 # float, NaN = образец не участвует
    tumor_only: bool = True
    is_control: bool = False
    expected: str = ""                 # чего мы ждём заранее, до обучения
    _mask: np.ndarray | None = field(default=None, repr=False)

    def mask(self, y: np.ndarray) -> np.ndarray:
        """Какие образцы входят в задачу: метка известна (+ только опухоли)."""
        m = ~pd.isna(self.labels)
        if self.tumor_only:
            m &= (y == 1)
        return m

    def n_pos(self, y) -> int:
        m = self.mask(y)
        return int(np.nansum(self.labels[m] == 1))

    def n_neg(self, y) -> int:
        m = self.mask(y)
        return int(np.nansum(self.labels[m] == 0))


# --------------------------------------------------------------------------- #
#  Парсеры клинических полей TCGA
# --------------------------------------------------------------------------- #
def parse_nodal(value) -> float:
    """N0 / N0 (i-) / N0 (i+) -> 0 ; N1..N3 -> 1 ; NX и пропуск -> NaN.

    N0 (i+) — единичные опухолевые клетки в узле; по AJCC это всё ещё N0,
    поэтому кладём в отрицательный класс, а не в положительный.
    """
    if not isinstance(value, str):
        return np.nan
    v = value.strip()
    if v.startswith("NX"):
        return np.nan
    if v.startswith("N0"):
        return 0.0
    return 1.0 if re.match(r"^N[123]", v) else np.nan


def parse_size(value) -> float:
    """T1* -> 0 (до 2 см) ; T2/T3/T4 -> 1 ; TX и пропуск -> NaN."""
    if not isinstance(value, str):
        return np.nan
    v = value.strip()
    if v.startswith("TX"):
        return np.nan
    if v.startswith("T1"):
        return 0.0
    return 1.0 if re.match(r"^T[234]", v) else np.nan


def parse_histology(value) -> float:
    """Протоковый (NOS) -> 0 ; дольковый -> 1 ; смешанные и редкие -> NaN.

    Смешанные формы намеренно выброшены: это не третий класс, а промежуточные
    случаи, из-за которых бинарная метка перестаёт быть определённой.
    """
    if not isinstance(value, str):
        return np.nan
    if "and lobular" in value or "mixed with other" in value:
        return np.nan
    if "Lobular" in value:
        return 1.0
    return 0.0 if value.startswith("Infiltrating duct carcinoma") else np.nan


def parse_vital(value) -> float:
    if value == "Alive":
        return 0.0
    return 1.0 if value == "Dead" else np.nan


# --------------------------------------------------------------------------- #
#  Сборка набора
# --------------------------------------------------------------------------- #
def attach_clinical(meta: pd.DataFrame, clinical: pd.DataFrame) -> pd.DataFrame:
    """Приклеивает расширенную клинику по пациенту к таблице образцов."""
    meta = meta.copy()
    cols = ["T", "N", "M", "primary_diagnosis", "morphology", "vital", "age", "race"]
    if clinical is None or clinical.empty:
        for c in cols:
            meta[c] = None
        return meta
    for c in cols:
        if c not in clinical.columns:
            clinical[c] = None
    return meta.join(clinical[cols], on="patient")


def build_task_suite(meta: pd.DataFrame, y: np.ndarray, min_class: int = 25) -> list[Task]:
    """Все задачи, для которых в данных хватает образцов обоих классов."""
    n = len(y)

    tumor_labels = np.where(y == 1, 1.0, 0.0)
    stage_labels = np.where(meta["stage_group"].to_numpy() == "early", 0.0,
                            np.where(meta["stage_group"].to_numpy() == "late", 1.0, np.nan))

    candidates = [
        Task(
            key="tumor_vs_normal",
            title="Опухоль или норма",
            question="Является ли образец ткани опухолевым?",
            neg_label="норма", pos_label="опухоль",
            why="базовая задача проекта: детекция рака по экспрессии генов",
            labels=tumor_labels, tumor_only=False,
            expected="задача насыщена — ждём AUC около 1.00 у любой модели",
        ),
        Task(
            key="histology",
            title="Гистотип: протоковый vs дольковый",
            question="Инфильтративный протоковый рак или дольковый?",
            neg_label="протоковый", pos_label="дольковый",
            why="дольковый рак хуже виден на маммографии, чаще двусторонний "
                "и требует другой тактики визуализации",
            labels=meta["primary_diagnosis"].map(parse_histology).to_numpy(dtype=float),
            expected="сильный сигнал: дольковый рак теряет CDH1 (E-кадгерин) — "
                     "ждём AUC около 0.90 и CDH1 в топе маркеров",
        ),
        Task(
            key="size",
            title="Размер опухоли: T1 vs T2+",
            question="Опухоль до 2 см или крупнее?",
            neg_label="T1 (≤2 см)", pos_label="T2+ (>2 см)",
            why="размер входит в стадию и определяет объём операции",
            labels=meta["T"].map(parse_size).to_numpy(dtype=float),
            expected="умеренный сигнал: размер — характеристика скорее "
                     "физическая, чем транскрипционная",
        ),
        Task(
            key="nodal",
            title="Лимфоузлы: N0 vs N+",
            question="Есть ли метастазы в подмышечных лимфоузлах?",
            neg_label="N0 (чисто)", pos_label="N+ (поражены)",
            why="САМЫЙ ценный из вопросов: именно статус лимфоузлов решает, "
                "делать ли подмышечную лимфодиссекцию — операцию, которая "
                "у половины пациенток оказывается лишней и даёт лимфедему",
            labels=meta["N"].map(parse_nodal).to_numpy(dtype=float),
            expected="слабый сигнал: метастазирование зависит не только от "
                     "профиля первичной опухоли — ждём AUC 0.62-0.70",
        ),
        Task(
            key="stage",
            title="Стадия: ранняя vs поздняя",
            question="Стадия I-II или III-IV?",
            neg_label="I-II", pos_label="III-IV",
            why="интегральная оценка запущенности процесса",
            labels=stage_labels,
            expected="слабый сигнал, во многом наследует статус лимфоузлов",
        ),
        Task(
            key="vital",
            title="Витальный статус (НЕГАТИВНЫЙ КОНТРОЛЬ)",
            question="Жива пациентка на момент последнего наблюдения или нет?",
            neg_label="жива", pos_label="умерла",
            why="контроль методики: по экспрессии первичной опухоли БЕЗ учёта "
                "времени наблюдения и цензурирования предсказывать исход нечем",
            labels=meta["vital"].map(parse_vital).to_numpy(dtype=float),
            is_control=True,
            expected="ждём AUC около 0.50 — и это правильный ответ. "
                     "Если пайплайн выдаст здесь высокое качество, значит он течёт",
        ),
    ]

    suite = []
    for t in candidates:
        m = t.mask(y)
        lab = t.labels[m]
        n_pos, n_neg = int((lab == 1).sum()), int((lab == 0).sum())
        if n_pos < min_class or n_neg < min_class:
            logger.info(f"  задача «{t.title}» пропущена: {n_neg}/{n_pos} образцов "
                        f"(нужно ≥{min_class} в каждом классе)")
            continue
        suite.append(t)
        tag = "  [негативный контроль]" if t.is_control else ""
        logger.info(f"  {t.title:44s} n={int(m.sum()):4d} "
                    f"({t.neg_label} {n_neg} / {t.pos_label} {n_pos}){tag}")

    if len(suite) < len(candidates):
        logger.info(f"  итого задач: {len(suite)} из {len(candidates)}")
    return suite
