# tok-to-fr-en

Projet de génération de datasets JSONL et fine-tuning LoRA pour enseigner le toki pona à des débutants francophones.

## Statut actuel

Le flux recommandé du dépôt est le flux pédagogique:

1. Générer un dataset conversationnel avec `generate_pedagogical_dataset.py`
2. Valider le schéma avec `validate_dataset.py`
3. Splitter sans fuite avec `split_pedagogy_jsonl.py`
4. Fine-tuner localement avec `train_qwen25_lora.py`
5. Tester avec `chat_model.py` et benchmarker sur `pedagogy_dataset_test.jsonl`

Le flux `training_data.jsonl` (paires `prompt/completion`) reste disponible pour des besoins de traduction brute, mais il n'est plus le flux principal.

## Fichiers importants

- `generate_pedagogical_dataset.py`: génération du dataset pédagogique multi-tours
- `split_pedagogy_jsonl.py`: split train/val/test groupé par paires fr/tok (évite la fuite)
- `train_qwen25_lora.py`: entraînement LoRA Qwen2.5-1.5B-Instruct
- `chat_model.py`: chat interactif avec le modèle entraîné
- `validate_dataset.py`: validation JSONL contre `schema.json`
- `pedagogy_dataset.jsonl`: dataset pédagogique source
- `pedagogy_dataset_train.jsonl`, `pedagogy_dataset_val.jsonl`, `pedagogy_dataset_test.jsonl`: splits

## Démarrage rapide

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

### 4) Fine-tuning LoRA

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

Option VRAM réduite:

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
  --early-stopping-patience 3 \
  --load-in-4bit
```

### 5) Test interactif

```bash
python chat_model.py --adapter qwen25-1.5b-tokipona-lora
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
