# Démarrage Rapide

Ce guide est la version courte, à jour, pour relancer un cycle complet sans ambiguïté.

## Pré-requis

- Python environnement prêt (`tok-to-fr-en` conseillé)
- `sentences.csv` et `links.csv` présents à la racine
- GPU NVIDIA recommandé pour l'entraînement

### Pré-requis Unsloth + FlashAttention (recommandé)

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate tok-to-fr-en

pip install -r requirements-unsloth.txt
```

Compiler FlashAttention pour l'architecture GPU locale.
Pour RTX 4070 (Ada), utiliser `sm_89`:

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

Vérification:

```bash
python -c "import flash_attn; print(flash_attn.__version__)"
```

## Télécharger les CSV Tatoeba

Si vous n'avez pas encore les CSV:

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

Vérifier la présence des fichiers:

```bash
ls -lh sentences.csv links.csv
```

## Workflow en 6 étapes

### 1) Générer le dataset pédagogique

```bash
python generate_pedagogical_dataset.py \
  --sentences sentences.csv \
  --links links.csv \
  --output pedagogy_dataset_v3.jsonl \
  --depth 3 \
  --max-samples 6000 \
  --level all \
  --seed 100
```

### 2) Valider le JSONL contre le schéma

```bash
python validate_dataset.py --jsonl pedagogy_dataset_v3.jsonl --schema schema.json
```

### 3) Split train/val/test sans fuite

```bash
python split_pedagogy_jsonl.py pedagogy_dataset_v3.jsonl
```

Fichiers produits :

- `pedagogy_dataset_v3_train.jsonl`
- `pedagogy_dataset_v3_val.jsonl`
- `pedagogy_dataset_v3_test.jsonl`

### 4) Lancer le fine-tuning Unsloth LoRA

```bash
python train_qwen25_unsloth.py \
  --train-file pedagogy_dataset_v3_train.jsonl \
  --val-file pedagogy_dataset_v3_val.jsonl \
  --output-dir qwen25-1.5b-tokipona-unsloth-v3 \
  --epochs 3 \
  --max-length 384 \
  --batch-size 2 \
  --grad-accum 8 \
  --save-steps 200 \
  --early-stopping-patience 3 \
  --resume-from-checkpoint qwen25-1.5b-tokipona-unsloth-final
```

Si VRAM limitée (< 8 GB) : remplacer `--batch-size 2` par `--batch-size 1 --grad-accum 16`.

### 5) Évaluer sur le test set figé

```bash
python eval_adapter.py \
  --adapter qwen25-1.5b-tokipona-unsloth-v3 \
  --test-file pedagogy_dataset_test.jsonl \
  --output eval_test_unsloth_v3_metrics.json
```

Référence actuelle : perplexité **1.1897** (adapter final). Le run v3 doit rester ≤ **1.2254** (gate +3%).

### 6) Tester en chat (smoke test 5 prompts fixes)

```bash
python chat_model.py --adapter qwen25-1.5b-tokipona-unsloth-v3
```

Prompts à tester dans l'ordre (voir `PLAN_MONTER_D_UN_CRAN_SAME_LLM.md` pour le protocole complet) :
1. `bonjour, je veux apprendre le toki pona`
2. `comment dit-on "je mange" en toki pona ?`
3. `j'ai dit "mi tawa li tomo", c'est correct ?` **(vali dation correction agrammaticale)**
4. `traduis : "tu es bon"`
5. `récapitule ce qu'on a vu`

## Points à connaître

- Le split pédagogique (`split_pedagogy_jsonl.py`) est obligatoire pour éviter la fuite entre exemples proches.
- `pedagogy_dataset_test.jsonl` est **figé** : ne jamais le régénérer, tous les runs sont comparés sur ce même test.
- Le script principal est `train_qwen25_unsloth.py` (Unsloth + FlashAttention).
- L'adapter de référence est `qwen25-1.5b-tokipona-unsloth-final` (perplexité 1.1897).

## Vérifications rapides

```bash
wc -l pedagogy_dataset_v3.jsonl
wc -l pedagogy_dataset_v3_train.jsonl pedagogy_dataset_v3_val.jsonl pedagogy_dataset_v3_test.jsonl
```

## Si vous reprenez un run interrompu

```bash
python train_qwen25_unsloth.py \
  --train-file pedagogy_dataset_v3_train.jsonl \
  --val-file pedagogy_dataset_v3_val.jsonl \
  --output-dir qwen25-1.5b-tokipona-unsloth-v3 \
  --resume-from-checkpoint qwen25-1.5b-tokipona-unsloth-v3/checkpoint-XXXX
```
