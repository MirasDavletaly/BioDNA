"""Датасет из recount3: здоровые люди (GTEx) + больные раком (TCGA).

Зачем нужен второй датасет. В исходных файлах проекта «норма» — это ткань,
взятая рядом с опухолью У ТЕХ ЖЕ онкобольных. Такая ткань уже несёт следы
болезни (field effect), и модель, обученная на ней, ничего не говорит о
здоровом человеке. Здесь «норма» — это грудная ткань доноров GTEx, у которых
рака не было вообще.

Главная опасность такого объединения: TCGA и GTEx — разные консорциумы,
разные центры, разные протоколы. Если склеить их «как есть», модель выучит
«кто секвенировал образец», а не «есть ли рак», и снова нарисует AUC 1.0.
Защита — recount3: обе когорты пересчитаны ОДНИМ пайплайном Monorail на одной
аннотации GENCODE v26. Остаточную разницу когорт всё равно нужно измерять
отдельно, этим занимается train_cohorts.py.

Источник: https://rna.recount.bio/ (recount3, human, release 1.0)
"""

from __future__ import annotations

import gzip
import logging
import re
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RECOUNT3 = ("https://recount-opendata.s3.amazonaws.com/recount3/release/"
            "human/data_sources")
ANNOTATION_URL = ("https://recount-opendata.s3.amazonaws.com/recount3/release/"
                  "human/annotations/gene_sums/human.gene_sums.G026.gtf.gz")
RECOUNT_ANNOTATION_NAME = "human.gene_sums.G026.gtf.gz"

FILES = {
    "gtex_counts": (f"{RECOUNT3}/gtex/gene_sums/ST/BREAST/gtex.gene_sums.BREAST.G026.gz",
                    "gtex.gene_sums.BREAST.G026.gz"),
    "gtex_meta": (f"{RECOUNT3}/gtex/metadata/ST/BREAST/gtex.gtex.BREAST.MD.gz",
                  "gtex.gtex.BREAST.MD.gz"),
    "tcga_counts": (f"{RECOUNT3}/tcga/gene_sums/CA/BRCA/tcga.gene_sums.BRCA.G026.gz",
                    "tcga.gene_sums.BRCA.G026.gz"),
    "tcga_meta": (f"{RECOUNT3}/tcga/metadata/CA/BRCA/tcga.tcga.BRCA.MD.gz",
                  "tcga.tcga.BRCA.MD.gz"),
    "annotation": (ANNOTATION_URL, "human.gene_sums.G026.gtf.gz"),
}

# Третья когорта — внешняя валидация. Fudan University Shanghai Cancer Center,
# трижды-негативный рак груди: 360 опухолей и 88 парных норм. Другая страна,
# больница, популяция и молекулярный подтип; в обучении не участвует никогда.
FUSCC_PROJECT = "SRP157974"
FUSCC_COUNTS_URL = (f"{RECOUNT3}/sra/gene_sums/74/{FUSCC_PROJECT}/"
                    f"sra.gene_sums.{FUSCC_PROJECT}.G026.gz")
FUSCC_META_URL = (f"{RECOUNT3}/sra/metadata/74/{FUSCC_PROJECT}/"
                  f"sra.sra.{FUSCC_PROJECT}.MD.gz")

# Группы образцов, которыми оперирует весь дальнейший анализ.
HEALTHY = "healthy"      # GTEx, донор без рака
ADJACENT = "adjacent"    # TCGA, ткань рядом с опухолью у онкобольного
TUMOR = "tumor"          # TCGA, первичная опухоль
LESION = "lesion"        # доброкачественное/предраковое поражение
INSITU = "insitu"        # карцинома in situ, ещё не инвазия

STAGE_RE = re.compile(r"stage\s+(IV|III|II|I|X)", re.IGNORECASE)


@dataclass
class CohortData:
    X: pd.DataFrame        # samples x genes, log2(TPM+1)
    meta: pd.DataFrame     # cohort, group, label, patient, sex, age, stage
    genes: pd.DataFrame    # symbol, gene_type, chrom, start, end, bp_length
    tpm_genes: list        # набор генов, по которому нормирован TPM
    annotation: pd.DataFrame = None   # полная аннотация GENCODE

    @property
    def y(self) -> np.ndarray:
        return self.meta["label"].to_numpy(dtype=np.int8)

    def subset(self, mask) -> "CohortData":
        mask = np.asarray(mask)
        return CohortData(self.X.loc[mask], self.meta.loc[mask], self.genes,
                          self.tpm_genes, self.annotation)


def _download(url: str, dest: Path, timeout: int = 1800) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        logger.info(f"Скачиваю {dest.name} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "BioDNA/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out, length=1 << 20)
        tmp.replace(dest)
        logger.info(f"  готово: {dest.stat().st_size / 1e6:.0f} MB")
        return True
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.error(f"  не скачалось: {exc}")
        tmp.unlink(missing_ok=True)
        return False


def ensure_files(cache_dir: str | Path) -> dict[str, Path]:
    cache_dir = Path(cache_dir)
    paths = {}
    for key, (url, name) in FILES.items():
        dest = cache_dir / name
        if not _download(url, dest):
            raise SystemExit(
                f"Не удалось скачать {name}. Нужен интернет — recount3 отдаёт "
                f"~190 MB один раз, дальше всё из кэша {cache_dir}/")
        paths[key] = dest
    return paths


def load_annotation(path: Path) -> pd.DataFrame:
    """GENCODE v26: ген -> символ, тип, координаты hg38, экзонная длина.

    Колонка score в этом GTF хранит суммарную длину экзонов гена — именно её
    recount3 использует для нормировки, ею же считаем TPM.
    """
    rows = []
    with gzip.open(path, "rt", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "gene":
                continue
            attrs = f[8]
            gene_id = attrs.split('gene_id "', 1)[1].split('"', 1)[0]
            gene_name = (attrs.split('gene_name "', 1)[1].split('"', 1)[0]
                         if 'gene_name "' in attrs else gene_id)
            gene_type = (attrs.split('gene_type "', 1)[1].split('"', 1)[0]
                         if 'gene_type "' in attrs else "unknown")
            rows.append((gene_id, gene_name, gene_type, f[0],
                         int(f[3]), int(f[4]), f[6], float(f[5])))

    genes = pd.DataFrame(rows, columns=["gene_id", "symbol", "gene_type", "chrom",
                                        "start", "end", "strand", "bp_length"])
    genes = genes.set_index("gene_id")
    logger.info(f"Аннотация GENCODE v26: {len(genes)} генов, "
                f"из них protein_coding {(genes['gene_type'] == 'protein_coding').sum()}")
    return genes


def _read_counts(path: Path) -> pd.DataFrame:
    """gene_sums recount3 -> DataFrame samples x genes (сырые покрытия)."""
    logger.info(f"Читаю {path.name} (это займёт минуту)...")
    # dtype задаём поимённо: общий dtype=float32 пытается применить себя и к
    # колонке gene_id, а там строки.
    with gzip.open(path, "rt") as fh:
        header = pd.read_csv(fh, sep="\t", comment="#", nrows=0)
    dtypes = {c: np.float32 for c in header.columns[1:]}

    with gzip.open(path, "rt") as fh:
        df = pd.read_csv(fh, sep="\t", comment="#", index_col=0, dtype=dtypes)
    logger.info(f"  {df.shape[1]} образцов x {df.shape[0]} генов")
    return df.T


def _read_meta(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", errors="replace") as fh:
        return pd.read_csv(fh, sep="\t", low_memory=False)


def to_log_tpm(counts: pd.DataFrame, bp_length: pd.Series) -> pd.DataFrame:
    """Покрытия -> TPM -> log2(TPM+1). Считается ПО КАЖДОМУ ОБРАЗЦУ отдельно.

    Никакой информации из других образцов здесь не используется, поэтому
    нормировка физически не может протащить утечку между train и test.
    """
    length_kb = (bp_length.reindex(counts.columns).to_numpy(dtype=np.float64) / 1000.0)
    length_kb[length_kb <= 0] = np.nan

    rate = counts.to_numpy(dtype=np.float64) / length_kb          # покрытие на т.п.н.
    rate = np.nan_to_num(rate, nan=0.0, posinf=0.0, neginf=0.0)
    total = rate.sum(axis=1, keepdims=True)
    total[total == 0] = 1.0
    tpm = rate / total * 1e6

    return pd.DataFrame(np.log2(tpm + 1.0).astype(np.float32),
                        index=counts.index, columns=counts.columns)


# Внешние когорты: в обучении не участвуют никогда, только проверка.
# Каждая описывается полем метаданных SRA и разбором его значений в
# (группа, метка, подпись). Метка None означает «промежуточное состояние»:
# для таких групп считается не точность, а распределение риска.
EXTERNAL_COHORTS = {
    "FUSCC": {
        "project": "SRP157974",
        "title": "FUSCC, Шанхай — трижды-негативный рак",
        "field": "tissue",
        "patient_field": "isolate",
        "labels": {
            "Primary breast tumor tissue": (TUMOR, 1, "TNBC"),
            "Paired normal breast tissue": (ADJACENT, 0, "норма рядом"),
        },
    },
    "VARLEY": {
        "project": "SRP042620",
        "title": "Varley — ER+ и TNBC, плюс редукционная маммопластика",
        "field": "tissue",
        "patient_field": None,
        "labels": {
            "ER+ Breast Cancer Primary Tumor": (TUMOR, 1, "опухоль ER+"),
            "Triple Negative Breast Cancer Primary Tumor": (TUMOR, 1, "опухоль TNBC"),
            "Uninvolved Breast Tissue Adjacent to ER+ Primary Tumor":
                (ADJACENT, 0, "норма рядом (ER+)"),
            "Uninvolved Breast Tissue Adjacent to TNBC Primary Tumor":
                (ADJACENT, 0, "норма рядом (TNBC)"),
            "Reduction Mammoplasty - No known cancer":
                (HEALTHY, 0, "редукционная маммопластика"),
            # Клеточные линии не ткань, в проверку не берём.
        },
    },
    "PROGRESSION": {
        "project": "SRP023262",
        "title": "Спектр прогрессии: норма → ранняя неоплазия → DCIS → инвазия",
        "field": "source_name",
        "patient_field": "patient number",
        "labels": {
            "normal breast": (HEALTHY, 0, "норма"),
            "early neoplasia": (LESION, None, "ранняя неоплазия"),
            "ductal carcinoma in situ": (INSITU, None, "DCIS (рак на месте)"),
            "invasive ductal carcinoma": (TUMOR, 1, "инвазивная карцинома"),
        },
    },
}

# Порядок групп на графике прогрессии — от нормы к инвазии.
PROGRESSION_ORDER = ["норма", "ранняя неоплазия", "DCIS (рак на месте)",
                     "инвазивная карцинома"]


def _sra_attr(s, key: str):
    """Разбирает поле sample_attributes вида 'ключ;;значение|ключ;;значение'."""
    if not isinstance(s, str):
        return None
    for part in s.split("|"):
        if part.lower().startswith(key.lower() + ";;"):
            return part.split(";;", 1)[1]
    return None


def load_external(name: str, cache_dir: str | Path, tpm_genes, model_genes,
                  annotation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Загружает внешнюю когорту из recount3 и приводит её к шкале обучения.

    Ключевая тонкость: TPM считается по ТОМУ ЖЕ набору генов, что и для
    обучающих когорт. Знаменатель нормировки обязан совпадать, иначе шкалы
    разъедутся и предсказание потеряет смысл.
    """
    spec = EXTERNAL_COHORTS[name]
    project = spec["project"]
    cache_dir = Path(cache_dir)

    counts_path = cache_dir / f"sra.gene_sums.{project}.G026.gz"
    meta_path = cache_dir / f"sra.sra.{project}.MD.gz"
    suffix = project[-2:]
    for url, dest in [
        (f"{RECOUNT3}/sra/gene_sums/{suffix}/{project}/sra.gene_sums.{project}.G026.gz",
         counts_path),
        (f"{RECOUNT3}/sra/metadata/{suffix}/{project}/sra.sra.{project}.MD.gz",
         meta_path),
    ]:
        if not _download(url, dest):
            raise SystemExit(f"Не скачался внешний набор {dest.name}")

    raw = _read_meta(meta_path)
    values = raw["sample_attributes"].map(lambda s: _sra_attr(s, spec["field"]))
    keep = values.isin(spec["labels"].keys()).to_numpy()
    if not keep.any():
        raise SystemExit(f"{name}: ни одно значение поля {spec['field']} не опознано")

    decoded = [spec["labels"][v] for v in values[keep]]
    if spec["patient_field"]:
        patient = raw.loc[keep, "sample_attributes"].map(
            lambda s: _sra_attr(s, spec["patient_field"])).to_numpy()
    else:
        patient = raw.loc[keep, "external_id"].to_numpy()

    meta = pd.DataFrame({
        "sample": raw.loc[keep, "external_id"].to_numpy(),
        "cohort": name,
        "group": [d[0] for d in decoded],
        "label": [d[1] for d in decoded],
        "subgroup": [d[2] for d in decoded],
        "patient": patient,
    }).set_index("sample")

    counts = _read_counts(counts_path)
    counts = counts.loc[counts.index.intersection(meta.index)]

    tpm_genes = list(tpm_genes)
    counts = counts.reindex(columns=tpm_genes, fill_value=0.0)
    X = to_log_tpm(counts, annotation.loc[tpm_genes, "bp_length"])[list(model_genes)]

    meta = meta.loc[X.index]
    counts_by = meta["subgroup"].value_counts().to_dict()
    logger.info(f"Внешняя когорта {name} ({project}): {len(meta)} образцов, "
                f"{meta['patient'].nunique()} пациентов, состав {counts_by}")
    return X, meta


def load_external_fuscc(cache_dir, tpm_genes, model_genes, annotation):
    """Обратная совместимость: FUSCC через общий загрузчик."""
    return load_external("FUSCC", cache_dir, tpm_genes, model_genes, annotation)


def normalize(X: pd.DataFrame, method: str = "logtpm") -> pd.DataFrame:
    """Нормировка ВНУТРИ образца — единственный вид, безопасный между когортами.

    logtpm  — как есть, log2(TPM+1).
    zsample — z-оценка по генам внутри образца: убирает сдвиг и масштаб профиля.
    rank    — ранг гена внутри образца, шкала [0,1]: полностью безразмерная
              величина, переживает смену платформы и глубины секвенирования.

    Все три считаются по одной строке матрицы и не смотрят на другие образцы,
    поэтому не могут перенести информацию из теста в обучение.
    """
    if method == "logtpm":
        return X

    values = X.to_numpy(dtype=np.float32)
    if method == "zsample":
        mu = values.mean(axis=1, keepdims=True)
        sd = values.std(axis=1, keepdims=True)
        sd[sd == 0] = 1.0
        out = (values - mu) / sd
    elif method == "rank":
        order = np.argsort(values, axis=1, kind="stable")
        ranks = np.empty_like(order, dtype=np.float32)
        idx = np.arange(values.shape[1], dtype=np.float32)
        np.put_along_axis(ranks, order, idx, axis=1)
        out = ranks / max(values.shape[1] - 1, 1)
    else:
        raise ValueError(f"неизвестная нормировка: {method}")

    return pd.DataFrame(out.astype(np.float32), index=X.index, columns=X.columns)


def simplify_stage(raw) -> str:
    if not isinstance(raw, str):
        return "Unknown"
    m = STAGE_RE.search(raw)
    if not m:
        return "Unknown"
    s = m.group(1).upper()
    return "Unknown" if s == "X" else s


def build_dataset(
    cache_dir: str | Path,
    female_only: bool = True,
    gene_types: tuple[str, ...] = ("protein_coding",),
    min_expressed_fraction: float = 0.2,
    min_log_tpm: float = 1.0,
) -> CohortData:
    """Собирает единую матрицу: GTEx-здоровые + TCGA-норма-рядом + TCGA-опухоли.

    female_only: грудная ткань мужчин и женщин различается принципиально, а в
    GTEx мужчин большинство (302 из 482) — если их не убрать, модель выучит пол.
    """
    paths = ensure_files(cache_dir)
    genes = load_annotation(paths["annotation"])

    gtex_meta = _read_meta(paths["gtex_meta"])
    tcga_meta = _read_meta(paths["tcga_meta"])

    # --- GTEx: здоровые доноры -------------------------------------------------
    g = gtex_meta[["external_id", "SUBJID", "SEX", "AGE", "SMRIN", "SMTSD",
                   "SMCENTER", "SMNABTCH"]].copy()
    g = g[g["SMTSD"] == "Breast - Mammary Tissue"]
    if female_only:
        before = len(g)
        g = g[g["SEX"] == 2]
        logger.info(f"GTEx: оставлено {len(g)} женских образцов из {before}")
    gtex_rows = pd.DataFrame({
        "sample": g["external_id"].to_numpy(),
        "cohort": "GTEX",
        "group": HEALTHY,
        "label": 0,
        "patient": g["SUBJID"].to_numpy(),
        "sex": np.where(g["SEX"].to_numpy() == 2, "female", "male"),
        "age": g["AGE"].to_numpy(),
        "stage": "Healthy",
        "batch": g["SMNABTCH"].to_numpy(),
        "rin": g["SMRIN"].to_numpy(),
    })

    # --- TCGA: опухоли и прилежащая норма -------------------------------------
    t = tcga_meta[["external_id", "tcga_barcode", "cgc_sample_sample_type",
                   "gdc_cases.demographic.gender", "cgc_case_pathologic_stage",
                   "gdc_cases.demographic.year_of_birth"]].copy()
    t.columns = ["external_id", "barcode", "sample_type", "gender", "stage_raw", "yob"]
    t = t[t["sample_type"].isin(["Primary Tumor", "Solid Tissue Normal"])]
    if female_only:
        before = len(t)
        t = t[t["gender"] == "female"]
        logger.info(f"TCGA: оставлено {len(t)} женских образцов из {before}")

    is_tumor = (t["sample_type"] == "Primary Tumor").to_numpy()
    tcga_rows = pd.DataFrame({
        "sample": t["external_id"].to_numpy(),
        "cohort": "TCGA",
        "group": np.where(is_tumor, TUMOR, ADJACENT),
        "label": is_tumor.astype(int),
        "patient": ["-".join(str(b).split("-")[:3]) for b in t["barcode"]],
        "sex": "female",
        "age": np.nan,
        "stage": [simplify_stage(s) if tu else "AdjacentNormal"
                  for s, tu in zip(t["stage_raw"], is_tumor)],
        "batch": [str(b).split("-")[5] if len(str(b).split("-")) > 5 else "?"
                  for b in t["barcode"]],
        "rin": np.nan,
    })
    tcga_rows["barcode"] = t["barcode"].to_numpy()

    meta = pd.concat([gtex_rows, tcga_rows], ignore_index=True).set_index("sample")

    # --- матрицы экспрессии ----------------------------------------------------
    gtex_counts = _read_counts(paths["gtex_counts"])
    tcga_counts = _read_counts(paths["tcga_counts"])

    keep_genes = genes.index
    if gene_types:
        keep_genes = genes.index[genes["gene_type"].isin(gene_types)]
    common = gtex_counts.columns.intersection(tcga_counts.columns).intersection(keep_genes)
    logger.info(f"Общих генов после фильтра по типу: {len(common)}")

    gtex_counts = gtex_counts.loc[gtex_counts.index.intersection(gtex_rows["sample"]), common]
    tcga_counts = tcga_counts.loc[tcga_counts.index.intersection(tcga_rows["sample"]), common]

    bp = genes.loc[common, "bp_length"]
    X = pd.concat([to_log_tpm(gtex_counts, bp), to_log_tpm(tcga_counts, bp)], axis=0)

    meta = meta.loc[meta.index.intersection(X.index)]
    X = X.loc[meta.index]

    # Гены, молчащие почти везде, только добавляют шум и различия платформ.
    expressed = (X > min_log_tpm).mean(axis=0)
    keep = expressed >= min_expressed_fraction
    logger.info(f"Экспрессируется в >={min_expressed_fraction:.0%} образцов: "
                f"{int(keep.sum())} генов из {len(keep)}")
    X = X.loc[:, keep[keep].index]

    counts = meta.groupby(["cohort", "group"]).size().to_dict()
    logger.info(f"Итоговый набор: {X.shape[0]} образцов x {X.shape[1]} генов")
    logger.info(f"  состав: {counts}")

    return CohortData(X=X, meta=meta, genes=genes.loc[X.columns],
                      tpm_genes=list(common), annotation=genes)


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    data = build_dataset(Path(__file__).parent.parent / "data" / "recount3")
    print(data.X.shape)
    print(data.meta.groupby(["cohort", "group"]).size())
    print(data.meta[data.meta["group"] == TUMOR]["stage"].value_counts())
