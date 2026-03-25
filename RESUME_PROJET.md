# Résumé Projet

## Objectif

Construire et entraîner un modèle pédagogique de toki pona pour débutants francophones à partir de données Tatoeba, avec un pipeline reproductible de génération JSONL, split sans fuite et fine-tuning LoRA.

## Architecture actuelle

### 1) Génération pédagogique

Entrées:

- `sentences.csv`
- `links.csv`

Sortie:

- `pedagogy_dataset.jsonl`

Le script `generate_pedagogical_dataset.py`:

- reconstruit des paires fr/tok via graphe de liens
- applique des filtres de qualité
- génère des dialogues pédagogiques multi-tours
- produit des métadonnées pédagogiques et qualité

### 2) Validation et split

- validation: `validate_dataset.py --jsonl pedagogy_dataset.jsonl --schema schema.json`
- split sans fuite: `split_pedagogy_jsonl.py pedagogy_dataset.jsonl`

### 3) Fine-tuning local

- script: `train_qwen25_lora.py`
- modèle de base: `Qwen/Qwen2.5-1.5B-Instruct`
- sortie: dossier adapter LoRA (`qwen25-1.5b-tokipona-lora`)

## Évolutions importantes intégrées

1. Conservation checkpoints améliorée (`save_total_limit=3`) pour limiter le risque de perdre le meilleur.
2. Prompt système de `chat_model.py` aligné avec le prompt système FR du dataset.
3. Ajout d'exemples `session_opening` pour mieux gérer les débuts de conversation spontanés.
4. Renforcement de `translation_with_explanation` avec recopie et validation exacte.

## Types de leçons générées

- `guided_dialogue`
- `pattern_drill`
- `error_correction`
- `review_recap`
- `translation_with_explanation`
- `session_opening`

## État d'évaluation (modèle actuel avant nouveau retrain)

Benchmark complet observé sur `pedagogy_dataset_test.jsonl`:

- exact match global: ~83%
- similarité moyenne: ~99%
- excellent sur 4 formats fermés
- principal écart: `translation_with_explanation` (variantes de traduction)

Ce constat a motivé la mise à jour du générateur pour contraindre davantage ce type.

## Recommandation opérationnelle

Toujours relancer un cycle complet après changement de générateur:

1. régénérer dataset
2. revalider schéma
3. resplit sans fuite
4. relancer fine-tuning
5. rebench sur test set
