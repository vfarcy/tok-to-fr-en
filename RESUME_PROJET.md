# Résumé Projet

**Date :** 26 mars 2026

## Objectif

Construire et entraîner un modèle pédagogique de toki pona pour débutants francophones à partir de données Tatoeba, avec un pipeline reproductible de génération JSONL, split sans fuite et fine-tuning LoRA via Unsloth.

## Architecture actuelle

### 1) Génération pédagogique

Entrées : `sentences.csv`, `links.csv`  
Script : `generate_pedagogical_dataset.py`  
Sortie : dataset JSONL pédagogique multi-tours

Le script :
- reconstruit des paires fr/tok via graphe de liens Tatoeba
- applique des filtres de qualité sur les phrases
- génère des dialogues pédagogiques multi-tours
- produit des métadonnées (`lesson_type`, `level`, `skills`)

### 2) Validation et split

```bash
python validate_dataset.py --jsonl <dataset>.jsonl --schema schema.json
python split_pedagogy_jsonl.py <dataset>.jsonl
```

Le split `split_pedagogy_jsonl.py` regroupe les exemples par paires fr/tok pour éviter toute fuite entre train et test.

### 3) Fine-tuning Unsloth LoRA

Script principal : `train_qwen25_unsloth.py`  
Modèle de base : `Qwen/Qwen2.5-1.5B-Instruct`  
Configuration GPU testée : RTX 4070 (11.6 GB VRAM), CUDA 12.8, Torch 2.11.0, FlashAttention 2

Paramètres recommandés :
```bash
python train_qwen25_unsloth.py \
  --train-file <dataset>_train.jsonl \
  --val-file <dataset>_val.jsonl \
  --output-dir <adapter_dir> \
  --epochs 3 --max-length 384 --batch-size 2 --grad-accum 8 \
  --save-steps 200 --early-stopping-patience 3
```

### 4) Évaluation

Script : `eval_adapter.py`  
Test set figé : `pedagogy_dataset_test.jsonl` (508 exemples)

```bash
python eval_adapter.py \
  --adapter <adapter_dir> \
  --test-file pedagogy_dataset_test.jsonl \
  --output eval_test_unsloth_<run>_metrics.json
```

### 5) Test interactif

```bash
python chat_model.py --adapter <adapter_dir>
```

## Résultats des runs curriculum

| Run | Dataset | Adapter | Perplexité |
|-----|---------|---------|-----------|
| A | A0,A1 — 6 000 ex | `qwen25-1.5b-tokipona-unsloth-A01` | 1.2032 |
| B | A0,A1,A2 — 8 000 ex | `qwen25-1.5b-tokipona-unsloth-A012` | 1.2032 |
| C (final) | all levels — 5 000 ex | `qwen25-1.5b-tokipona-unsloth-final` | **1.1897** |

Meilleur adapter : `qwen25-1.5b-tokipona-unsloth-final`

## Types de leçons générées

- `guided_dialogue` — dialogue guidé débutant
- `pattern_drill` — réflexe sur un patron syntaxique
- `error_correction` — correction explicite d'une erreur fréquente
- `review_recap` — récapitulatif de session
- `translation_with_explanation` — traduction avec explication de structure
- `session_opening` — accueil et orientation débutant en début de conversation

## Défaut connu (v3)

Le modèle final peut proposer `"mi tawa li tomo"` comme correction dans certains cas — structure agrammaticale en toki pona.  
Correctif prévu : augmentation des exemples `error_correction` sur structures de lieu/mouvement dans le dataset v3.

## État d'évaluation

Métriques enregistrées dans :
- `eval_test_unsloth_A01_metrics.json`
- `eval_test_unsloth_A012_metrics.json`
- `eval_test_unsloth_final_metrics.json`

## Recommandation opérationnelle

Suivre le micro-plan v3 dans `PLAN_MONTER_D_UN_CRAN_SAME_LLM.md` :
1. Régénérer dataset v3 (seed 100, all levels, 6 000 ex, ≥25% error_correction)
2. Renforcer le prompt système sur les corrections
3. Passer le smoke test 5 prompts fixes avant tout nouveau run
