# Synthèse Finale

**Date :** 26 mars 2026  
**Branche :** `experiment/curriculum-v2-plan`

## Résumé exécutif

Trois runs curriculum ont été complétés sur Qwen2.5-1.5B-Instruct + Unsloth LoRA.  
Le meilleur adapter est `qwen25-1.5b-tokipona-unsloth-final` avec une perplexité test de **1.1897**.  
Un défaut de correction grammaticale (`error_correction` sur phrases avec complément de lieu) a été identifié en smoke test et documente le micro-plan v3.

## Résultats des runs

| Run | Adapter | Perplexité | avg_loss | Exemples |
|-----|---------|-----------|----------|----------|
| A (A0,A1) | `qwen25-1.5b-tokipona-unsloth-A01` | 1.2032 | — | 508 |
| B (A0,A1,A2) | `qwen25-1.5b-tokipona-unsloth-A012` | 1.2032 | — | 508 |
| C/final (all) | `qwen25-1.5b-tokipona-unsloth-final` | **1.1897** | 0.1737 | 508 |

Le run B n'a pas apporté de gain vs A : curriculum sur A0→A1→A2 n'est pas une amélioration en soi. C'est le passage à tous les niveaux (run C) qui produit une légère amélioration de perplexité.

## Pipeline en production

```bash
# Génération
python generate_pedagogical_dataset.py \
  --sentences sentences.csv --links links.csv \
  --output pedagogy_dataset_v3.jsonl \
  --depth 3 --max-samples 6000 --level all --seed 100

# Validation + split
python validate_dataset.py --jsonl pedagogy_dataset_v3.jsonl --schema schema.json
python split_pedagogy_jsonl.py pedagogy_dataset_v3.jsonl

# Train
python train_qwen25_unsloth.py \
  --train-file pedagogy_dataset_v3_train.jsonl \
  --val-file pedagogy_dataset_v3_val.jsonl \
  --output-dir qwen25-1.5b-tokipona-unsloth-v3 \
  --epochs 3 --max-length 384 --batch-size 2 --grad-accum 8 \
  --save-steps 200 --early-stopping-patience 3 \
  --resume-from-checkpoint qwen25-1.5b-tokipona-unsloth-final

# Eval
python eval_adapter.py \
  --adapter qwen25-1.5b-tokipona-unsloth-v3 \
  --test-file pedagogy_dataset_test.jsonl \
  --output eval_test_unsloth_v3_metrics.json

# Chat
python chat_model.py --adapter qwen25-1.5b-tokipona-unsloth-v3
```

## Défaut connu et prochaine action

**Défaut :** le modèle final peut proposer `"mi tawa li tomo"` comme correction — agrammatical en toki pona.  
**Cause probable :** sous-représentation des exemples `error_correction` sur structures de lieu/mouvement.  
**Prochaine action :** appliquer le micro-plan v3 (voir `PLAN_MONTER_D_UN_CRAN_SAME_LLM.md`).

## Ce qui est stable et versionné

- `schema.json` : validé, supporte `session_opening` (`OPEN_XXXX`, `L_OPEN`, skill `accueil`)
- `eval_test_unsloth_A01_metrics.json`
- `eval_test_unsloth_A012_metrics.json`
- `eval_test_unsloth_final_metrics.json`
- Tous les scripts (`generate_pedagogical_dataset.py`, `validate_dataset.py`, `split_pedagogy_jsonl.py`, `train_qwen25_unsloth.py`, `eval_adapter.py`, `chat_model.py`)

## Ce qui n'est pas versionné (artefacts lourds)

- Dossiers adapter (`qwen25-1.5b-tokipona-unsloth-*/`)
- Caches de compilation (`unsloth_compiled_cache/`)
