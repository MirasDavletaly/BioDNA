# 🧬 Breast Cancer Detection с DNABERT

Проект предсказания рака молочной железы на основе RNA-seq данных с применением
предобученной модели **DNABERT-2** и классических ML-алгоритмов.

## 📁 Структура проекта

```
cancer_dnabert/
├── data/
│   ├── BC-TCGA-Normal.txt     # RNA-seq нормальные образцы (61 пациент)
│   └── BC-TCGA-Tumor.txt      # RNA-seq опухолевые образцы (529 пациентов)
│
├── src/
│   ├── data_preprocessing.py  # Загрузка, ANOVA-отбор генов, DNA-конвертация
│   ├── dnabert_model.py        # DNABERT-2 классификатор + BiLSTM baseline
│   └── visualization.py       # ROC, confusion matrix, важность генов
│
├── models/                    # Сохранённые веса (создаётся при обучении)
├── results/                   # Графики и метрики (создаётся при обучении)
│
├── train.py                   # Главный скрипт обучения
├── predict.py                 # Инференс на новых образцах
├── requirements.txt
└── README.md
```

## 🚀 Быстрый старт

### 1. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 2. Подготовка данных
Скопируйте файлы в папку `data/`:
- `BC-TCGA-Normal.txt` — нормальные образцы
- `BC-TCGA-Tumor.txt`  — опухолевые образцы

### 3. Обучение

**Вариант A: Sklearn baseline (быстро, без GPU)**
```bash
python train.py --mode baseline --n_genes 512
```

**Вариант B: BiLSTM + k-мер эмбеддинги (средняя скорость)**
```bash
python train.py --mode lite --n_genes 512 --epochs 15
```

**Вариант C: DNABERT-2 fine-tuning (медленно, нужен GPU + интернет)**
```bash
python train.py --mode dnabert --n_genes 512 --epochs 5 --freeze
```

### 4. Предсказание
```bash
python predict.py --demo                                  # Демо-режим
python predict.py --sample_file my_rna_sample.txt --mode lite
```

## 🧪 Датасет: BC-TCGA (The Cancer Genome Atlas)

| Параметр | Значение |
|---|---|
| Тип данных | RNA-seq экспрессия генов |
| Кол-во генов | 17,814 |
| Normal образцы | 61 |
| Tumor образцы | 529 |
| Тип рака | Рак молочной железы (Breast Cancer) |

## 🔬 Методология

### Ключевая идея: Expression → DNA Sequence
DNABERT обучен на реальных геномных последовательностях ДНК.
Мы адаптируем RNA-seq данные для этой модели:

```
Шаг 1: Отбор генов
  17,814 генов → ANOVA F-test → топ 512 наиболее дифференциально экспрессированных

Шаг 2: Квантилирование → нуклеотиды
  Уровень экспрессии  → Нуклеотид
  ─────────────────────────────────
  Q1 (низкий)  → A
  Q2            → C
  Q3            → G
  Q4 (высокий) → T

Шаг 3: k-меры для токенизации
  "ACGTACGTACGT..." → "ACGTAC CGTACG GTACGT ..."
                       ↑ 6-меры через пробел

Шаг 4: DNABERT-2 Encoder
  k-меры → CLS token (768-dim) → Classifier Head → P(Tumor)
```

### Архитектура DNABERT классификатора

```
Input k-mers
    ↓
[DNABERT-2 Encoder] (117M параметров, предобучен на геноме человека)
    ↓ CLS token pooling
[Dropout 0.3]
[Linear: 768 → 256]
[GELU]
[LayerNorm]
[Dropout 0.2]
[Linear: 256 → 2]
    ↓
Softmax → P(Normal), P(Tumor)
```

## 📊 Ожидаемые результаты

| Модель | AUC-ROC | Accuracy | F1 |
|--------|---------|----------|-----|
| Logistic Regression | ~0.97 | ~0.95 | ~0.97 |
| Random Forest | ~0.98 | ~0.96 | ~0.98 |
| Gradient Boosting | ~0.98 | ~0.97 | ~0.98 |
| BiLSTM + k-мер | ~0.97 | ~0.95 | ~0.97 |
| DNABERT-2 fine-tuned | ~0.99 | ~0.98 | ~0.99 |

## ⚙️ Параметры командной строки

```
train.py:
  --mode        dnabert | lite | baseline
  --n_genes     кол-во генов (default: 512)
  --epochs      эпох обучения (default: 10)
  --batch_size  размер батча (default: 16)
  --kmer_size   размер k-мера [3|6] (default: 6)
  --max_len     макс токенов для BERT (default: 512)
  --freeze      заморозить backbone DNABERT
```

## 📚 Литература

- Ji et al., "DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome", *Bioinformatics*, 2021
- Zhou et al., "DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome", *arXiv*, 2023
- TCGA Research Network: https://www.cancer.gov/tcga