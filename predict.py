import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / "src"))

from src.data_preprocessing import expression_to_dna_sequence, sequence_to_kmers


def predict_sample(expression_values: np.ndarray, model, tokenizer_or_vocab,
                   mode: str = "lite", max_len: int = 256) -> dict:
    import torch
    from sklearn.preprocessing import StandardScaler

    dna_seq  = expression_to_dna_sequence(expression_values)
    kmer_seq = sequence_to_kmers(dna_seq, k=6)

    if mode == "dnabert":
        encoding = tokenizer_or_vocab(
            kmer_seq, max_length=512,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            logits = model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
            )
    else:
        vocab = tokenizer_or_vocab
        ids = [vocab.get(k, 1) for k in kmer_seq.split()][:max_len]
        ids += [0] * (max_len - len(ids))
        input_ids = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            logits = model(input_ids=input_ids)

    probs = torch.softmax(logits, dim=-1).squeeze().numpy()
    prediction = int(np.argmax(probs))

    return {
        "P(Normal)": float(probs[0]),
        "P(Tumor)":  float(probs[1]),
        "Prediction": "TUMOR 🔴" if prediction == 1 else "NORMAL 🟢",
        "Confidence": float(max(probs)),
        "dna_sequence_preview": dna_seq[:50] + "...",
    }


def demo_predict():
    print("ДЕМО: Предсказание рака по профилю экспрессии генов")

    np.random.seed(42)
    n_genes = 512

    print("\nСинтетические образцы:")

    for sample_name, shift in [("Нормальный", 0.0), ("Опухолевый", 1.5)]:
        expression = np.random.normal(loc=shift, scale=0.5, size=n_genes)
        dna_seq    = expression_to_dna_sequence(expression)
        kmer_seq   = sequence_to_kmers(dna_seq, k=6)

        n_kmers = len(kmer_seq.split())
        nt_counts = {nt: dna_seq.count(nt) for nt in "ACGT"}
        gc_content = (nt_counts["C"] + nt_counts["G"]) / len(dna_seq) * 100

        print(f"\n  [{sample_name}]")
        print(f"  Экспрессия (первые 5 генов): {expression[:5].round(3)}")
        print(f"  DNA последовательность: {dna_seq[:60]}...")
        print(f"  Нуклеотидный состав: A={nt_counts['A']} C={nt_counts['C']} "
              f"G={nt_counts['G']} T={nt_counts['T']}")
        print(f"  GC-содержание: {gc_content:.1f}%")
        print(f"  k-меры (первые 5): {' '.join(kmer_seq.split()[:5])}...")
        print(f"  Всего k-меров: {n_kmers}")

    print("💡 Для реального предсказания запустите:")
    print("   python train.py --mode baseline   (обучение)")
    print("   python predict.py --demo          (этот скрипт)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Запустить демо")
    parser.add_argument("--sample_file", type=str, help="TSV файл с экспрессией")
    parser.add_argument("--model_path", type=str, default="models/best_model.pt")
    parser.add_argument("--mode", choices=["dnabert", "lite"], default="lite")
    args = parser.parse_args()

    if args.demo or not args.sample_file:
        demo_predict()
    else:
        print(f"📂 Загрузка образца: {args.sample_file}")
        df = pd.read_csv(args.sample_file, sep="\t", index_col=0)
        expression = df.values.flatten()
        print(f"   Генов: {len(expression)}")
        print("Загрузка сохранённой модели требует предварительного обучения.")
        print("   Запустите: python train.py --mode lite")


if __name__ == "__main__":
    main()