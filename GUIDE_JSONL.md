# Guide JSONL

Guide de référence du dépôt pour générer, valider, splitter et entraîner des datasets JSONL.

## 1. Deux formats de dataset

### A. Dataset de traduction brute

- Fichier type: `training_data.jsonl`
- Format: `{"prompt": "...", "completion": "..."}`
- Scripts: `generate_jsonl.py`, `generate_jsonl_advanced.py`, `split_jsonl.py`
- Usage: tâches de traduction simple

### B. Dataset pédagogique conversationnel (recommandé)

- Fichier type: `pedagogy_dataset.jsonl`
- Format: enregistrements multi-tours avec `messages[]`, `lesson`, `pedagogy`, `quality`
- Scripts: `generate_pedagogical_dataset.py`, `validate_dataset.py`, `split_pedagogy_jsonl.py`, `train_qwen25_lora.py`
- Usage: modèle tuteur de toki pona pour débutants francophones

## 2. Workflow recommandé

### Étape 1: Génération

```bash
python generate_pedagogical_dataset.py \
  --sentences sentences.csv \
  --links links.csv \
  --output pedagogy_dataset.jsonl \
  --depth 3 \
  --max-samples 5000
```

Options utiles:

- `--max-source-sentences`: limiter le scan pour itérations rapides
- `--min-words-fr`, `--max-words-fr`: borne phrases FR
- `--min-words-tok`, `--max-words-tok`: borne phrases TOK
- `--level`: filtrage CECRL (`A0,A1`, etc.)

### Étape 2: Validation schéma

```bash
python validate_dataset.py --jsonl pedagogy_dataset.jsonl --schema schema.json
```

### Étape 3: Split sans fuite

```bash
python split_pedagogy_jsonl.py pedagogy_dataset.jsonl
```

Ce script groupe les exemples par paire fr/tok reconstruite pour éviter la fuite entre train/val/test.

### Étape 4: Fine-tuning LoRA

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

Option VRAM réduite: `--load-in-4bit`.

### Étape 5: Test et benchmark

- Test interactif: `python chat_model.py --adapter qwen25-1.5b-tokipona-lora`
- Évaluation recommandée: benchmark sur `pedagogy_dataset_test.jsonl`

## 3. Types de leçons pédagogiques

Le générateur produit les types suivants:

1. `guided_dialogue`
2. `pattern_drill`
3. `error_correction`
4. `review_recap`
5. `translation_with_explanation`
6. `session_opening` (exemples injectés)

Notes importantes:

- `session_opening` couvre les démarrages spontanés (ex: "je veux apprendre le toki pona").
- `translation_with_explanation` a été renforcé avec recopie exacte et validation finale pour réduire les variations non souhaitées.

## 4. Reprise d'entraînement

```bash
python train_qwen25_lora.py \
  --train-file pedagogy_dataset_train.jsonl \
  --val-file pedagogy_dataset_val.jsonl \
  --output-dir qwen25-1.5b-tokipona-lora \
  --resume-from-checkpoint qwen25-1.5b-tokipona-lora/checkpoint-XXXX
```

## 5. Dépannage rapide

### Erreur mémoire GPU

- Réduire `--batch-size`
- Augmenter `--grad-accum`
- Activer `--load-in-4bit`

### Validation schéma échoue

- Relancer `validate_dataset.py`
- Corriger les lignes indiquées dans la sortie

### Split non cohérent

- Utiliser `split_pedagogy_jsonl.py` pour le dataset pédagogique
- Garder `split_jsonl.py` pour `training_data.jsonl`

## 6. Commandes de vérification

```bash
wc -l pedagogy_dataset.jsonl
wc -l pedagogy_dataset_train.jsonl pedagogy_dataset_val.jsonl pedagogy_dataset_test.jsonl
python analyze_jsonl.py pedagogy_dataset_train.jsonl --samples 5
```
