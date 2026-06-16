import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
# Каждому уровню экспрессии соответствует нуклеотид
EXPRESSION_TO_NUC = {0: "A", 1: "C", 2: "G", 3: "T"}


def load_expression_data(
    normal_path: str,
    tumor_path: str,
    n_top_genes: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    logger.info("Загрузка данных экспрессии генов...")

    df_normal = pd.read_csv(normal_path, sep="\t", index_col=0)
    df_tumor  = pd.read_csv(tumor_path,  sep="\t", index_col=0)

    logger.info(f"  Normal: {df_normal.shape[1]} образцов, {df_normal.shape[0]} генов")
    logger.info(f"  Tumor : {df_tumor.shape[1]} образцов, {df_tumor.shape[0]} генов")

    # Транспонируем: samples × genes
    df_normal = df_normal.T
    df_tumor  = df_tumor.T

    # Оставляем общие гены
    common_genes = df_normal.columns.intersection(df_tumor.columns)
    df_normal = df_normal[common_genes]
    df_tumor  = df_tumor[common_genes]

    # Объединяем
    X_all = pd.concat([df_normal, df_tumor], axis=0)
    y_all = np.array([0] * len(df_normal) + [1] * len(df_tumor))

    # Заполняем NaN медианой по гену
    X_all = X_all.fillna(X_all.median())

    logger.info(f"Итого: {X_all.shape[0]} образцов, {X_all.shape[1]} генов")

    # Отбор топ-N генов по дисперсии (быстрый фильтр перед ANOVA)
    gene_var = X_all.var(axis=0)
    top_var_genes = gene_var.nlargest(min(n_top_genes * 5, len(common_genes))).index
    X_filtered = X_all[top_var_genes]

    # ANOVA: выбираем гены с наибольшей разделимостью классов
    selector = SelectKBest(f_classif, k=min(n_top_genes, len(top_var_genes)))
    selector.fit(X_filtered.values, y_all)
    selected_mask = selector.get_support()
    selected_genes = list(top_var_genes[selected_mask])

    X_selected = X_all[selected_genes].values

    logger.info(f"Отобрано {len(selected_genes)} информативных генов (ANOVA F-test)")

    return X_selected, y_all, selected_genes


def expression_to_dna_sequence(
    expression_vector: np.ndarray,
    kmer_size: int = 6,
) -> str:
    q1, q2, q3 = np.percentile(expression_vector, [25, 50, 75])

    def to_nuc(val: float) -> str:
        if val <= q1:
            return "A"
        elif val <= q2:
            return "C"
        elif val <= q3:
            return "G"
        else:
            return "T"

    dna_seq = "".join(to_nuc(v) for v in expression_vector)
    return dna_seq


def sequence_to_kmers(sequence: str, k: int = 6) -> str:
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    return " ".join(kmers)


def prepare_sequences(
    X: np.ndarray,
    kmer_size: int = 6,
) -> List[str]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    sequences = []
    for i, sample in enumerate(X_scaled):
        dna_seq = expression_to_dna_sequence(sample, kmer_size)
        kmer_seq = sequence_to_kmers(dna_seq, k=kmer_size)
        sequences.append(kmer_seq)

        if (i + 1) % 100 == 0:
            logger.info(f"  Обработано {i+1}/{len(X_scaled)} образцов")

    logger.info(f"Сгенерировано {len(sequences)} последовательностей")
    logger.info(f"   Пример: {sequences[0][:80]}...")

    return sequences


if __name__ == "__main__":
    BASE = Path(__file__).parent.parent

    X, y, genes = load_expression_data(
        normal_path=str(BASE / "data" / "BC-TCGA-Normal.txt"),
        tumor_path=str(BASE / "data" / "BC-TCGA-Tumor.txt"),
        n_top_genes=512,
    )
    seqs = prepare_sequences(X, kmer_size=6)
    print(f"\nShape X: {X.shape}")
    print(f"Labels: {np.bincount(y)}")
    print(f"Seq length: {len(seqs[0].split())} k-mers")