# Plan v3 — Améliorer la qualité pédagogique (même LLM, même stack)

**Date de mise à jour :** 26 mars 2026  
**Branche active :** `experiment/curriculum-v2-plan`

## Statut des runs curriculum (COMPLÉTÉS)

| Run | Données | Adapter produit | Perplexité test | Statut |
|-----|---------|-----------------|-----------------|--------|
| A | A0,A1 (6 000 ex) | `qwen25-1.5b-tokipona-unsloth-A01` | 1.2032 | ✅ done |
| B | A0,A1,A2 (8 000 ex) | `qwen25-1.5b-tokipona-unsloth-A012` | 1.2032 | ✅ done — aucun gain vs A |
| C (final) | all levels (5 000 ex) | `qwen25-1.5b-tokipona-unsloth-final` | **1.1897** | ✅ done — meilleur |

Meilleur adapter en production : **`qwen25-1.5b-tokipona-unsloth-final`**

Fichiers de métriques de référence :
- `eval_test_unsloth_A01_metrics.json`
- `eval_test_unsloth_A012_metrics.json`
- `eval_test_unsloth_final_metrics.json`

---

## Défaut identifié en smoke test

Le run final produit des réponses pédagogiques correctes dans la majorité des cas.  
Un défaut de correction grammaticale a été observé : correction proposée `"mi tawa li tomo"` — incorrect en toki pona (un seul prédicat attendu par phrase simple).

Ce défaut n'est pas capturé par la perplexité. Il se manifeste dans les `error_correction` sur des phrases avec deux prédicats ou un complément de lieu.

---

## Micro-plan v3 — 3 changements ciblés

### Changement 1 — Dataset : renforcer les exemples de correction

Dans `generate_pedagogical_dataset.py`, augmenter la proportion d'exemples `error_correction` qui couvrent :
1. phrase avec deux prédicats (ex: `mi tawa mi moku`) → seul `mi moku` ou `mi tawa` est correct seul
2. phrases avec complément de lieu sur un verbe de mouvement (ex: `mi tawa tomo` est correct, `mi tawa li tomo` ne l'est pas)
3. structure SOV mal formée

Commande de régénération avec seed différent pour éviter les doublons :
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

Puis valider et re-splitter :
```bash
python validate_dataset.py --jsonl pedagogy_dataset_v3.jsonl --schema schema.json
python split_pedagogy_jsonl.py pedagogy_dataset_v3.jsonl
```

Contrôle : vérifier que les `error_correction` représentent au moins 25 % du split train :
```bash
python -c "
import json
lines = [json.loads(l) for l in open('pedagogy_dataset_v3_train.jsonl')]
ec = [l for l in lines if l.get('metadata', {}).get('lesson_type') == 'error_correction']
print(f'error_correction: {len(ec)}/{len(lines)} = {100*len(ec)/len(lines):.1f}%')
"
```

### Changement 2 — Prompt : contraindre la correction explicite

Dans le script d'évaluation ou dans le prompt système de `chat_model.py`, ajouter une instruction explicite pour que le modèle :
1. cite la phrase fautive entre guillemets avant de corriger
2. donne la forme corrigée suivie d'une courte explication grammaticale
3. ne propose jamais une variante agrammaticale comme alternative

Exemple de prompt système renforcé pour l'évaluation :
```
Tu es un professeur de toki pona pour débutants francophones.
Quand tu corriges une erreur, tu dois :
1. citer la phrase incorrecte : « ... »
2. donner la correction : la forme correcte est « ... »
3. expliquer brièvement pourquoi en une phrase simple
Ne propose jamais une forme qui n'est pas grammaticalement correcte en toki pona.
```

### Changement 3 — Smoke test fixe 5 prompts (protocole pass/fail)

Avant tout nouveau raining, le smoke test suivant doit passer sur les 5 prompts ci-dessous.  
Un prompt est PASS si la réponse ne contient aucune structure agrammaticale.

```bash
python chat_model.py --adapter qwen25-1.5b-tokipona-unsloth-final
```

Prompts fixes (à coller dans cet ordre) :

| # | Prompt | Critère PASS |
|---|--------|--------------|
| 1 | `bonjour, je veux apprendre le toki pona` | accueil pédagogique, pas de code |
| 2 | `comment dit-on "je mange" en toki pona ?` | réponse = "mi moku" (ou variante correcte avec explication) |
| 3 | `j'ai dit "mi tawa li tomo", c'est correct ?` | correction explicite, mentionne que "li" après "mi" est incorrect ici |
| 4 | `traduis : "tu es bon"` | réponse = "sina pona" |
| 5 | `récapitule ce qu'on a vu` | liste courte des points abordés, ton pédagogique |

Critère global : 5/5 PASS requis. Toute régression sur le prompt 3 bloque le passage au run suivant.

---

## Règles de base (non négociables)

1. **Test set figé** : `pedagogy_dataset_test.jsonl` ne sera jamais régénéré.
2. **Gate de promotion** : perplexité test ≤ meilleur run précédent + 3 %.
3. **Smoke test 5/5** requis avant de déclarer un adapter en production.
4. **Aucune comparaison de runs sur des test sets différents.**

---

## Pipeline complet de référence (v3)

```bash
# 1. Génération
python generate_pedagogical_dataset.py \
  --sentences sentences.csv \
  --links links.csv \
  --output pedagogy_dataset_v3.jsonl \
  --depth 3 --max-samples 6000 --level all --seed 100

# 2. Validation + split
python validate_dataset.py --jsonl pedagogy_dataset_v3.jsonl --schema schema.json
python split_pedagogy_jsonl.py pedagogy_dataset_v3.jsonl

# 3. Train (depuis final comme point de départ)
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

# 4. Evaluation quantitative
python eval_adapter.py \
  --adapter qwen25-1.5b-tokipona-unsloth-v3 \
  --test-file pedagogy_dataset_test.jsonl \
  --output eval_test_unsloth_v3_metrics.json

# 5. Smoke test (5 prompts fixes, voir ci-dessus)
python chat_model.py --adapter qwen25-1.5b-tokipona-unsloth-v3

# 6. Gate : comparer avec final
python -c "
import json
a = json.load(open('eval_test_unsloth_final_metrics.json'))
b = json.load(open('eval_test_unsloth_v3_metrics.json'))
diff = (b['perplexity'] - a['perplexity']) / a['perplexity'] * 100
gate = b['perplexity'] <= a['perplexity'] * 1.03
print(f'final={a[\"perplexity\"]:.6f}  v3={b[\"perplexity\"]:.6f}  delta={diff:+.2f}%  gate_pass={gate}')
"
```

---

## Checklist v3

- [ ] Régénération dataset v3 (seed 100, all levels, 6 000 ex)
- [ ] Validation schéma OK
- [ ] Proportion error_correction ≥ 25 % vérifiée
- [ ] Train v3 lancé depuis final
- [ ] Eval quantitative enregistrée (`eval_test_unsloth_v3_metrics.json`)
- [ ] Gate perplexité : v3 ≤ 1.1897 * 1.03 = 1.2254
- [ ] Smoke test 5/5 PASS (en particulier prompt 3 — correction "mi tawa li tomo")
- [ ] Adapter v3 déclaré en production si tous les gates passent

---

## Gestion des artefacts

Versionner :
- scripts
- docs
- fichiers JSON de métriques (`eval_test_unsloth_*_metrics.json`)

Ne pas versionner :
- dossiers d'adapters et checkpoints
- caches de compilation (`unsloth_compiled_cache/`)
