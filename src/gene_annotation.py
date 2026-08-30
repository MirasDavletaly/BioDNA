"""Аннотация генов геномными координатами (hg38).

Источники — публичные дампы UCSC Genome Browser:
  * refFlat.txt.gz  — координаты транскриптов (ген -> chrom, txStart, txEnd, strand)
  * cytoBand.txt.gz — цитогенетические полосы (chr17:43,000,000 -> 17q21.31)

Файлы скачиваются один раз и кэшируются в data/annotation/.
Если интернета нет, модуль деградирует мягко: координаты будут NaN,
а весь остальной пайплайн продолжит работать.
"""

from __future__ import annotations

import gzip
import logging
import shutil
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

UCSC_BASE = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database"
REFFLAT_URL = f"{UCSC_BASE}/refFlat.txt.gz"
CYTOBAND_URL = f"{UCSC_BASE}/cytoBand.txt.gz"
GENE_INFO_URL = ("https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/"
                 "Mammalia/Homo_sapiens.gene_info.gz")

MAIN_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
CHROM_ORDER = {c: i for i, c in enumerate(MAIN_CHROMS)}

# Длины хромосом hg38 — нужны для «манхэттен»-графика по координатам генома.
CHROM_SIZES = {
    "chr1": 248956422, "chr2": 242193529, "chr3": 198295559, "chr4": 190214555,
    "chr5": 181538259, "chr6": 170805979, "chr7": 159345973, "chr8": 145138636,
    "chr9": 138394717, "chr10": 133797422, "chr11": 135086622, "chr12": 133275309,
    "chr13": 114364328, "chr14": 107043718, "chr15": 101991189, "chr16": 90338345,
    "chr17": 83257441, "chr18": 80373285, "chr19": 58617616, "chr20": 64444167,
    "chr21": 46709983, "chr22": 50818468, "chrX": 156040895, "chrY": 57227415,
}


def _download(url: str, dest: Path, timeout: int = 120) -> bool:
    """Скачивает url в dest. Возвращает False, если сеть недоступна."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        logger.info(f"Скачиваю аннотацию: {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "BioDNA/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
        tmp.replace(dest)
        logger.info(f"  сохранено: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return True
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.warning(f"  не удалось скачать {url}: {exc}")
        tmp.unlink(missing_ok=True)
        return False


def _load_refflat(cache_dir: Path) -> pd.DataFrame | None:
    """ген -> chrom / start / end / strand (объединение всех транскриптов гена)."""
    gz = cache_dir / "refFlat.txt.gz"
    if not _download(REFFLAT_URL, gz):
        return None

    cols = ["geneName", "name", "chrom", "strand", "txStart", "txEnd",
            "cdsStart", "cdsEnd", "exonCount", "exonStarts", "exonEnds"]
    with gzip.open(gz, "rt") as fh:
        df = pd.read_csv(fh, sep="\t", names=cols, usecols=[0, 2, 3, 4, 5],
                         dtype={"geneName": str, "chrom": str, "strand": str})

    df = df[df["chrom"].isin(MAIN_CHROMS)]

    # У гена бывает несколько транскриптов и записей на альт-контигах:
    # берём самую частую хромосому, затем полный охват по координатам.
    main_chrom = df.groupby("geneName")["chrom"].agg(lambda s: s.value_counts().index[0])
    df = df.merge(main_chrom.rename("mainChrom"), on="geneName")
    df = df[df["chrom"] == df["mainChrom"]]

    genes = df.groupby("geneName").agg(
        chrom=("chrom", "first"),
        start=("txStart", "min"),
        end=("txEnd", "max"),
        strand=("strand", "first"),
    )
    genes["length"] = genes["end"] - genes["start"]
    return genes


def _load_cytobands(cache_dir: Path) -> pd.DataFrame | None:
    gz = cache_dir / "cytoBand.txt.gz"
    if not _download(CYTOBAND_URL, gz):
        return None
    with gzip.open(gz, "rt") as fh:
        bands = pd.read_csv(fh, sep="\t", names=["chrom", "start", "end", "band", "stain"])
    return bands[bands["chrom"].isin(MAIN_CHROMS)].reset_index(drop=True)


def _assign_bands(genes: pd.DataFrame, bands: pd.DataFrame) -> pd.Series:
    """Для каждого гена находит цитобанд по его старт-координате (17 -> 17q21.31)."""
    out = pd.Series(index=genes.index, dtype=object)
    for chrom, chunk in genes.groupby("chrom"):
        cb = bands[bands["chrom"] == chrom].sort_values("start")
        if cb.empty:
            continue
        idx = np.searchsorted(cb["start"].to_numpy(), chunk["start"].to_numpy(), side="right") - 1
        idx = np.clip(idx, 0, len(cb) - 1)
        label = chrom.replace("chr", "") + cb["band"].to_numpy()[idx]
        out.loc[chunk.index] = label
    return out



def _load_aliases(cache_dir: Path, known_symbols: set[str]) -> dict[str, str]:
    """Устаревший символ -> современный (WISP1 -> CCN4, PPAPDC1A -> PLPP4).

    В файлах TCGA 2010-х годов много старых имён генов, которых уже нет в
    свежей аннотации UCSC. NCBI gene_info хранит синонимы — по ним и мапим.
    """
    gz = cache_dir / "Homo_sapiens.gene_info.gz"
    if not _download(GENE_INFO_URL, gz):
        return {}

    with gzip.open(gz, "rt", encoding="utf-8", errors="replace") as fh:
        info = pd.read_csv(fh, sep="	", usecols=["Symbol", "Synonyms"],
                           dtype=str, na_values=["-"])

    info = info.dropna(subset=["Symbol"])
    info = info[info["Symbol"].isin(known_symbols)]

    aliases: dict[str, str] = {}
    for symbol, syns in zip(info["Symbol"], info["Synonyms"]):
        if not isinstance(syns, str):
            continue
        for alias in syns.split("|"):
            alias = alias.strip()
            # Не перетираем настоящий символ и не плодим неоднозначности.
            if alias and alias not in known_symbols:
                aliases.setdefault(alias, symbol)

    logger.info(f"Синонимов генов загружено: {len(aliases)}")
    return aliases


def load_gene_coordinates(cache_dir: str | Path) -> pd.DataFrame:
    """Таблица координат всех известных генов: index=символ гена.

    Колонки: chrom, start, end, strand, length, cytoband, chrom_num, genome_pos.
    При отсутствии сети возвращает пустой DataFrame нужной формы.
    """
    cache_dir = Path(cache_dir)
    empty = pd.DataFrame(
        columns=["chrom", "start", "end", "strand", "length",
                 "cytoband", "chrom_num", "genome_pos"]
    )

    genes = _load_refflat(cache_dir)
    if genes is None or genes.empty:
        logger.warning("Координаты генов недоступны — работаем без геномной привязки")
        return empty

    bands = _load_cytobands(cache_dir)
    genes["cytoband"] = _assign_bands(genes, bands) if bands is not None else np.nan

    genes["chrom_num"] = genes["chrom"].map(CHROM_ORDER)

    # Кумулятивная координата: позиция в «развёрнутом» геноме, для манхэттен-графика.
    offsets, running = {}, 0
    for c in MAIN_CHROMS:
        offsets[c] = running
        running += CHROM_SIZES[c]
    genes["genome_pos"] = genes["chrom"].map(offsets) + genes["start"]

    genes = genes.sort_values(["chrom_num", "start"])
    genes.attrs["aliases"] = _load_aliases(cache_dir, set(genes.index))
    logger.info(f"Аннотировано координат: {len(genes)} генов (hg38)")
    return genes


def annotate(gene_names, coords: pd.DataFrame) -> pd.DataFrame:
    """Приклеивает координаты к произвольному списку генов (сохраняя порядок)."""
    names = list(gene_names)
    idx = pd.Index(names, name="gene")
    if coords.empty:
        return pd.DataFrame(index=idx, columns=coords.columns)

    aliases = coords.attrs.get("aliases", {})
    lookup = [g if g in coords.index else aliases.get(g, g) for g in names]
    out = coords.reindex(lookup)
    out.index = idx
    return out


def format_locus(row) -> str:
    """chr17:43,044,294-43,170,245 (17q21.31, -)"""
    if row is None or pd.isna(row.get("chrom")):
        return "—"
    band = row.get("cytoband")
    band_txt = f", {band}" if isinstance(band, str) else ""
    return (f"{row['chrom']}:{int(row['start']):,}-{int(row['end']):,}"
            f" ({row['strand']}{band_txt})")


def chromosome_offsets() -> dict[str, int]:
    offsets, running = {}, 0
    for c in MAIN_CHROMS:
        offsets[c] = running
        running += CHROM_SIZES[c]
    return offsets


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    coords = load_gene_coordinates(Path(__file__).parent.parent / "data" / "annotation")
    for g in ["BRCA1", "TP53", "ERBB2", "ESR1", "MKI67"]:
        if g in coords.index:
            print(f"{g:8s} {format_locus(coords.loc[g])}")
