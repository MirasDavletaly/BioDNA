"""Клинические стадии TCGA через публичный GDC API.

В файлах экспрессии стадии нет — есть только баркод образца
(TCGA-A8-A06T-01A-11R-A00Z-07). Первые три поля — идентификатор пациента,
по нему GDC отдаёт ajcc_pathologic_stage. Это единственный способ
честно оценить, ловит ли модель РАННЮЮ стадию (I-II), а не только запущенную.

Результат кэшируется в data/annotation/tcga_stages.csv.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

GDC_CASES_URL = "https://api.gdc.cancer.gov/cases"

# Коды типа образца в баркоде TCGA (14-15 символы).
TUMOR_CODES = {"01", "02", "03", "04", "05", "06", "07", "08", "09"}
NORMAL_CODES = {"10", "11", "12", "13", "14"}

STAGE_RE = re.compile(r"stage\s+(IV|III|II|I|X)", re.IGNORECASE)


def patient_id(barcode: str) -> str:
    """TCGA-A8-A06T-01A-11R-A00Z-07 -> TCGA-A8-A06T"""
    return "-".join(str(barcode).split("-")[:3])


def plate_id(barcode: str) -> str:
    """TCGA-A8-A06T-01A-11R-A00Z-07 -> 'A00Z' (плашка = технический батч)."""
    parts = str(barcode).split("-")
    return parts[5] if len(parts) > 5 else "?"


def sample_type_code(barcode: str) -> str:
    """Возвращает двузначный код типа образца ('01' опухоль / '11' норма)."""
    parts = str(barcode).split("-")
    return parts[3][:2] if len(parts) > 3 and len(parts[3]) >= 2 else ""


def is_tumor_barcode(barcode: str) -> bool | None:
    code = sample_type_code(barcode)
    if code in TUMOR_CODES:
        return True
    if code in NORMAL_CODES:
        return False
    return None


def simplify_stage(raw: str | float | None) -> str:
    """'Stage IIIA' -> 'III'; мусор и отсутствие -> 'Unknown'."""
    if not isinstance(raw, str):
        return "Unknown"
    m = STAGE_RE.search(raw)
    if not m:
        return "Unknown"
    stage = m.group(1).upper()
    return "Unknown" if stage == "X" else stage


def _query_gdc(patients: list[str], timeout: int = 90) -> dict[str, str]:
    payload = {
        "filters": {"op": "in", "content": {"field": "submitter_id", "value": patients}},
        "fields": "submitter_id,diagnoses.ajcc_pathologic_stage",
        "format": "JSON",
        "size": str(len(patients) + 10),
    }
    req = urllib.request.Request(
        GDC_CASES_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "BioDNA/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())

    out: dict[str, str] = {}
    for hit in body.get("data", {}).get("hits", []):
        stages = [d.get("ajcc_pathologic_stage") for d in hit.get("diagnoses", []) or []]
        stages = [s for s in stages if isinstance(s, str)]
        if stages:
            out[hit["submitter_id"]] = stages[0]
    return out


def fetch_stages(barcodes, cache_dir: str | Path, batch_size: int = 150) -> pd.DataFrame:
    """Таблица barcode -> patient / stage_raw / stage / stage_group.

    stage_group: 'early' (I-II), 'late' (III-IV), 'unknown'.
    Без сети возвращает таблицу со стадиями 'Unknown' — пайплайн не падает.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / "tcga_stages.csv"

    barcodes = list(barcodes)
    patients = sorted({patient_id(b) for b in barcodes})

    known: dict[str, str] = {}
    if cache.exists():
        prev = pd.read_csv(cache)
        known = dict(zip(prev["patient"], prev["stage_raw"].fillna("")))
        logger.info(f"Стадии из кэша: {len(known)} пациентов")

    missing = [p for p in patients if p not in known]
    if missing:
        logger.info(f"Запрашиваю стадии в GDC API: {len(missing)} пациентов...")
        try:
            for i in range(0, len(missing), batch_size):
                batch = missing[i:i + batch_size]
                known.update(_query_gdc(batch))
                for p in batch:
                    known.setdefault(p, "")
                logger.info(f"  {min(i + batch_size, len(missing))}/{len(missing)}")
            pd.DataFrame({"patient": list(known), "stage_raw": list(known.values())}) \
                .to_csv(cache, index=False)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            logger.warning(f"GDC API недоступен ({exc}) — стадии будут 'Unknown'")

    rows = []
    for b in barcodes:
        p = patient_id(b)
        raw = known.get(p, "")
        stage = simplify_stage(raw)
        rows.append({
            "barcode": b,
            "patient": p,
            "stage_raw": raw or None,
            "stage": stage,
            "stage_group": {"I": "early", "II": "early",
                            "III": "late", "IV": "late"}.get(stage, "unknown"),
        })
    return pd.DataFrame(rows).set_index("barcode")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    base = Path(__file__).parent.parent
    header = pd.read_csv(base / "data" / "BC-TCGA-Tumor.txt", sep="\t", nrows=0)
    df = fetch_stages(header.columns[1:], base / "data" / "annotation")
    print(df["stage"].value_counts())
    print(df["stage_group"].value_counts())
