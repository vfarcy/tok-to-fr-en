# Démarrage Rapide

Ce guide est la version courte, à jour, pour relancer un cycle complet sans ambiguïté.

## Pré-requis

- Python environnement prêt (`tok-to-fr-en` conseillé)
- `sentences.csv` et `links.csv` présents à la racine
- GPU NVIDIA recommandé pour l'entraînement

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
  --output pedagogy_dataset.jsonl \
  --depth 3 \
  --max-samples 5000
```

### 2) Valider le JSONL contre le schéma

```bash
python validate_dataset.py --jsonl pedagogy_dataset.jsonl --schema schema.json
```

### 3) Split train/val/test sans fuite

```bash
python split_pedagogy_jsonl.py pedagogy_dataset.jsonl
```

Fichiers produits:

- `pedagogy_dataset_train.jsonl`
- `pedagogy_dataset_val.jsonl`
- `pedagogy_dataset_test.jsonl`

### 4) Lancer le fine-tuning LoRA

```bash
python train_qwen25_lora.py \
  --train-file pedagogy_dataset_train.jsonl \
  --val-file pedagogy_dataset_val.jsonl \
  --output-dir qwen25-1.5b-tokipona-lora \
  --epochs 3 \
  --batch-size 1 \
  --grad-accum 16 \
  --max-length 512 \
  --save-steps 50 \
  --early-stopping-patience 3
```

Option 4-bit (si VRAM limite): ajouter `--load-in-4bit`.

### 5) Tester en chat

```bash
python chat_model.py --adapter qwen25-1.5b-tokipona-lora
```

### 6) Évaluer sur le test set

Recommandé: benchmark génération vs réponse attendue sur `pedagogy_dataset_test.jsonl`.

## Points à connaître

- Le split pédagogique (`split_pedagogy_jsonl.py`) est obligatoire pour éviter la fuite entre exemples proches.
- Le générateur pédagogique inclut maintenant des exemples `session_opening` pour gérer les messages du type "je veux apprendre le toki pona".
- `translation_with_explanation` a été renforcé pour mieux stabiliser la forme exacte de traduction.

## Vérifications rapides

```bash
wc -l pedagogy_dataset.jsonl
wc -l pedagogy_dataset_train.jsonl pedagogy_dataset_val.jsonl pedagogy_dataset_test.jsonl
```

## Si vous reprenez un run interrompu

```bash
python train_qwen25_lora.py \
  --train-file pedagogy_dataset_train.jsonl \
  --val-file pedagogy_dataset_val.jsonl \
  --output-dir qwen25-1.5b-tokipona-lora \
  --resume-from-checkpoint qwen25-1.5b-tokipona-lora/checkpoint-XXXX
```
