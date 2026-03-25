# Plan de progression (meme LLM, CSV actuels)

Ce document detaille une strategie concrete pour "monter d'un cran" sans changer de modele de base.

Hypothese:
- on garde le meme base model (Qwen2.5-1.5B-Instruct)
- on exploite uniquement les donnees disponibles (sentences.csv, links.csv)
- on fait progresser surtout la qualite des donnees et la sequence d'entrainement

## 1) Objectif

Ameliorer:
1. exactitude des traductions toki pona <-> francais
2. stabilite des formulations attendues
3. qualite pedagogique en dialogue

Sans changer:
1. architecture du LLM
2. stack principale (generate/split/train/chat)

## 2) Principe general

Le gain vient surtout de:
1. meilleur curriculum de donnees
2. entrainement en phases
3. evaluation stricte et iterative

Plutot qu'un seul run "tout melange", on fait 3 runs successifs.

## 3) Pipeline en 3 runs

## Run A - Fondations debutant (A0+A1)

But:
- renforcer les structures les plus frequentes et simples
- reduire le bruit avance au debut

### A.1 Generer dataset cible (A0+A1)

```bash
python generate_pedagogical_dataset.py \
  --sentences sentences.csv \
  --links links.csv \
  --output pedagogy_dataset_A01.jsonl \
  --depth 3 \
  --max-samples 6000 \
  --level A0,A1 \
  --seed 42
```

### A.2 Valider + split sans fuite

```bash
python validate_dataset.py --jsonl pedagogy_dataset_A01.jsonl --schema schema.json
python split_pedagogy_jsonl.py pedagogy_dataset_A01.jsonl
```

### A.3 Entrainement

```bash
python train_qwen25_lora.py \
  --train-file pedagogy_dataset_A01_train.jsonl \
  --val-file pedagogy_dataset_A01_val.jsonl \
  --output-dir qwen25-1.5b-tokipona-lora-A01 \
  --epochs 2 \
  --batch-size 1 \
  --grad-accum 16 \
  --max-length 512 \
  --save-steps 50 \
  --early-stopping-patience 3
```

## Run B - Extension (A0+A1+A2)

But:
- etendre la couverture linguistique
- garder la base stable acquise en Run A

### B.1 Generer dataset plus large

```bash
python generate_pedagogical_dataset.py \
  --sentences sentences.csv \
  --links links.csv \
  --output pedagogy_dataset_A012.jsonl \
  --depth 3 \
  --max-samples 8000 \
  --level A0,A1,A2 \
  --seed 43
```

### B.2 Valider + split

```bash
python validate_dataset.py --jsonl pedagogy_dataset_A012.jsonl --schema schema.json
python split_pedagogy_jsonl.py pedagogy_dataset_A012.jsonl
```

### B.3 Reprise depuis Run A (curriculum)

```bash
python train_qwen25_lora.py \
  --train-file pedagogy_dataset_A012_train.jsonl \
  --val-file pedagogy_dataset_A012_val.jsonl \
  --output-dir qwen25-1.5b-tokipona-lora-A012 \
  --epochs 2 \
  --batch-size 1 \
  --grad-accum 16 \
  --max-length 512 \
  --save-steps 50 \
  --early-stopping-patience 3 \
  --resume-from-checkpoint qwen25-1.5b-tokipona-lora-A01/checkpoint-XXXX
```

Note:
- remplacer checkpoint-XXXX par le meilleur checkpoint du Run A.
- si reprise impossible proprement, lancer Run B from scratch (moins ideal, mais acceptable).

## Run C - Stabilisation finale pedagogique

But:
- consolider les comportements attendus (especially translation_with_explanation)
- finaliser un adapter deployable

### C.1 Regenerer dataset de prod final

```bash
python generate_pedagogical_dataset.py \
  --sentences sentences.csv \
  --links links.csv \
  --output pedagogy_dataset_final.jsonl \
  --depth 3 \
  --max-samples 5000 \
  --level all \
  --seed 44
```

### C.2 Valider + split

```bash
python validate_dataset.py --jsonl pedagogy_dataset_final.jsonl --schema schema.json
python split_pedagogy_jsonl.py pedagogy_dataset_final.jsonl
```

### C.3 Entrainement final

```bash
python train_qwen25_lora.py \
  --train-file pedagogy_dataset_final_train.jsonl \
  --val-file pedagogy_dataset_final_val.jsonl \
  --output-dir qwen25-1.5b-tokipona-lora-final \
  --epochs 3 \
  --batch-size 1 \
  --grad-accum 16 \
  --max-length 512 \
  --save-steps 50 \
  --early-stopping-patience 3
```

Option VRAM limite:
- ajouter `--load-in-4bit`

## 4) Recommandations data (avec CSV actuels)

Pour augmenter la competence sans changer de LLM:
1. garder un fort poids A0/A1 (coeur pedagogique)
2. conserver A2 pour generalisation
3. limiter B1 (faible volume, bruit potentiel)
4. conserver les exemples session_opening
5. conserver la version renforcee de translation_with_explanation

## 5) Evaluation apres chaque run

Minimum apres chaque run:
1. test interactif:

```bash
python chat_model.py --adapter qwen25-1.5b-tokipona-lora-final
```

2. benchmark sur test set pedagogique
- mesurer exact match + similarite
- mesurer par type de lecon

Criteres de progression attendus:
1. baisse eval_loss stable
2. hausse/stabilite eval_mean_token_accuracy
3. reduction des erreurs sur translation_with_explanation

## 6) Decision gate (quand s'arreter)

Arreter et garder le meilleur checkpoint si:
1. eval_loss remonte 2-3 evaluations de suite
2. gain marginal tres faible pendant une longue phase
3. qualite chat commence a se degrader qualitativement

## 7) Risques et mitigations

Risque 1: overfit format
- Mitigation: garder diversite de phrases et de types

Risque 2: variabilite trop libre en traduction
- Mitigation: cible stricte + recopie + validation finale (deja en place)

Risque 3: fuite de donnees
- Mitigation: split_pedagogy_jsonl.py uniquement pour le pedagogique

## 8) Checklist execution

1. [ ] Run A genere / valide / split / train
2. [ ] Bench A enregistre
3. [ ] Run B genere / valide / split / train
4. [ ] Bench B enregistre
5. [ ] Run C genere / valide / split / train
6. [ ] Bench C enregistre
7. [ ] Choix final du meilleur adapter

## 9) Resultat attendu

Sans changer de LLM, cette approche doit apporter:
1. meilleure stabilite des reponses
2. meilleure exactitude pedagogique
3. meilleure competence pratique en toki pona pour l'usage cible debutant francophone
