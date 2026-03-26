# tok-to-fr-en

Projet de génération de datasets JSONL et fine-tuning LoRA pour enseigner le toki pona à des débutants francophones.

## Statut actuel (26 mars 2026)

Trois runs curriculum complétés. Meilleur adapter : **`qwen25-1.5b-tokipona-unsloth-final`** (perplexité test **1.1897**).

| Run | Données | Perplexité test |
|-----|---------|----------------|
| A (A0,A1) | 6 000 ex | 1.2032 |
| B (A0,A1,A2) | 8 000 ex | 1.2032 |
| C/final (all) | 5 000 ex | **1.1897** |

Le flux recommandé du dépôt est le flux pédagogique avec Unsloth :

1. Générer un dataset avec `generate_pedagogical_dataset.py`
2. Valider le schéma avec `validate_dataset.py`
3. Splitter sans fuite avec `split_pedagogy_jsonl.py`
4. Fine-tuner avec `train_qwen25_unsloth.py` (Unsloth + FlashAttention)
5. Évaluer avec `eval_adapter.py` sur `pedagogy_dataset_test.jsonl` (test figé)
6. Tester interactivement avec `chat_model.py`

Prochaine étape : micro-plan v3 — voir `PLAN_MONTER_D_UN_CRAN_SAME_LLM.md`.

## Fichiers importants

- `generate_pedagogical_dataset.py`: génération du dataset pédagogique multi-tours
- `split_pedagogy_jsonl.py`: split train/val/test groupé par paires fr/tok (évite la fuite)
- `train_qwen25_unsloth.py`: entraînement Unsloth LoRA Qwen2.5-1.5B-Instruct **(script principal)**
- `eval_adapter.py`: évaluation quantitative (perplexité) sur test set figé
- `chat_model.py`: chat interactif avec le modèle entraîné
- `validate_dataset.py`: validation JSONL contre `schema.json`
- `pedagogy_dataset_test.jsonl`: test set figé (508 exemples) — **ne jamais régénérer**

## Obtenir les CSV depuis Tatoeba

Les deux fichiers nécessaires sont:

- `sentences.csv`
- `links.csv`

### Option A: wget

```bash
mkdir -p tatoeba_tmp && cd tatoeba_tmp

wget https://downloads.tatoeba.org/exports/sentences.tar.bz2
wget https://downloads.tatoeba.org/exports/links.tar.bz2

tar -xjf sentences.tar.bz2
tar -xjf links.tar.bz2

mv sentences.csv ../sentences.csv
mv links.csv ../links.csv

cd ..
rm -rf tatoeba_tmp
```

### Option B: curl

```bash
mkdir -p tatoeba_tmp && cd tatoeba_tmp

curl -LO https://downloads.tatoeba.org/exports/sentences.tar.bz2
curl -LO https://downloads.tatoeba.org/exports/links.tar.bz2

tar -xjf sentences.tar.bz2
tar -xjf links.tar.bz2

mv sentences.csv ../sentences.csv
mv links.csv ../links.csv

cd ..
rm -rf tatoeba_tmp
```

Vérification:

```bash
ls -lh sentences.csv links.csv
```

## Démarrage rapide

### 0) Installer l'environnement Unsloth + FlashAttention

Activer l'environnement conda (exemple: `tok-to-fr-en`) puis installer les dépendances:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate tok-to-fr-en

pip install -r requirements-unsloth.txt
```

Compiler ensuite FlashAttention pour l'architecture GPU locale.
Pour une RTX 4070, utiliser `sm_89`:

```bash
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX"
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/include"
export C_INCLUDE_PATH="$CONDA_PREFIX/include"
export FLASH_ATTN_CUDA_ARCHS='89'
export NVCC_THREADS=4
export MAX_JOBS=4

pip install flash-attn --no-build-isolation
```

Vérifier l'installation:

```bash
python -c "import flash_attn; print(flash_attn.__version__)"
```

### 1) Génération dataset pédagogique

```bash
python generate_pedagogical_dataset.py \
  --sentences sentences.csv \
  --links links.csv \
  --output pedagogy_dataset.jsonl \
  --depth 3 \
  --max-samples 5000
```

### 2) Validation

```bash
python validate_dataset.py --jsonl pedagogy_dataset.jsonl --schema schema.json
```

### 3) Split sans fuite

```bash
python split_pedagogy_jsonl.py pedagogy_dataset.jsonl
```

### 4) Fine-tuning Unsloth LoRA

```bash
python train_qwen25_unsloth.py \
  --train-file pedagogy_dataset_train.jsonl \
  --val-file pedagogy_dataset_val.jsonl \
  --output-dir qwen25-1.5b-tokipona-unsloth-v3 \
  --epochs 3 \
  --max-length 384 \
  --batch-size 2 \
  --grad-accum 8 \
  --save-steps 200 \
  --early-stopping-patience 3
```

Pour continuer depuis le meilleur adapter existant :

```bash
python train_qwen25_unsloth.py \
  --train-file pedagogy_dataset_train.jsonl \
  --val-file pedagogy_dataset_val.jsonl \
  --output-dir qwen25-1.5b-tokipona-unsloth-v3 \
  --epochs 3 \
  --max-length 384 \
  --batch-size 2 \
  --grad-accum 8 \
  --save-steps 200 \
  --early-stopping-patience 3 \
  --resume-from-checkpoint qwen25-1.5b-tokipona-unsloth-final
```

Option VRAM réduite (< 8 GB) : `--batch-size 1 --grad-accum 16`

### 5) Évaluation quantitative

```bash
python eval_adapter.py \
  --adapter qwen25-1.5b-tokipona-unsloth-v3 \
  --test-file pedagogy_dataset_test.jsonl \
  --output eval_test_unsloth_v3_metrics.json
```

### 6) Test interactif

```bash
python chat_model.py --adapter qwen25-1.5b-tokipona-unsloth-v3
```

## Leçons générées

Le générateur pédagogique produit les types suivants:

- `guided_dialogue`
- `pattern_drill`
- `error_correction`
- `review_recap`
- `translation_with_explanation`
- `session_opening` (injecté pour gérer les messages d'ouverture spontanés)

Note: `translation_with_explanation` est maintenant plus contraint (copie exacte + confirmation finale) pour réduire les variations au benchmark.

## Documentation

- `DEMARRAGE_RAPIDE.md`: procédure courte et opérationnelle
- `GUIDE_JSONL.md`: guide détaillé des scripts et formats
- `INDEX.md`: index de référence et checklists
- `RESUME_PROJET.md`: résumé technique
- `SYNTHESE_FINALE.md`: synthèse projet et état d'avancement
