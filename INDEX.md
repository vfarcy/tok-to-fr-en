# Index

Référence rapide des fichiers et flux du projet.

## Données

- `training_data.jsonl`: dataset de traduction brute (`prompt/completion`)
- `training_data_train.jsonl`, `training_data_val.jsonl`, `training_data_test.jsonl`: split du dataset brut
- `pedagogy_dataset.jsonl`: dataset pédagogique multi-tours (source)
- `pedagogy_dataset_train.jsonl`, `pedagogy_dataset_val.jsonl`, `pedagogy_dataset_test.jsonl`: split pédagogique sans fuite
- `schema.json`: schéma de validation du dataset pédagogique

## Scripts génération

- `generate_jsonl.py`: génération simple du dataset brut
- `generate_jsonl_advanced.py`: génération avancée (paramètres CLI)
- `generate_pedagogical_dataset.py`: génération du dataset pédagogique

## Scripts qualité / transformation

- `validate_dataset.py`: validation JSONL vs `schema.json`
- `split_jsonl.py`: split simple d'un JSONL ligne à ligne
- `split_pedagogy_jsonl.py`: split groupé fr/tok sans fuite
- `analyze_jsonl.py`: stats et échantillons

## Scripts entraînement / test

- `train_qwen25_unsloth.py`: entraînement Unsloth LoRA Qwen2.5-1.5B **(script principal)**
- `train_qwen25_lora.py`: entraînement LoRA standard (conservé pour référence)
- `eval_adapter.py`: évaluation quantitative sur test set figé (perplexité)
- `chat_model.py`: chat interactif sur adapter fine-tuné

## Datasets curriculum (branch experiment/curriculum-v2-plan)

- `pedagogy_dataset_A01.jsonl` + splits — run A (A0,A1, perplexité 1.2032)
- `pedagogy_dataset_A012.jsonl` + splits — run B (A0,A1,A2, perplexité 1.2032)
- `pedagogy_dataset_final.jsonl` + splits — run C/final (all, perplexité 1.1897)
- `eval_test_unsloth_A01_metrics.json`: métriques run A
- `eval_test_unsloth_A012_metrics.json`: métriques run B
- `eval_test_unsloth_final_metrics.json`: métriques run C
- `pedagogy_dataset_test.jsonl`: **test set figé (508 exemples)**

## Documentation

- `README.MD`: vue générale et commandes à jour
- `DEMARRAGE_RAPIDE.md`: exécution rapide du pipeline
- `GUIDE_JSONL.md`: détails techniques et options
- `RESUME_PROJET.md`: résumé technique + résultats runs
- `SYNTHESE_FINALE.md`: synthèse opérationnelle + défaut connu
- `PLAN_MONTER_D_UN_CRAN_SAME_LLM.md`: micro-plan v3 + checklist

## Workflow recommandé

1. `generate_pedagogical_dataset.py`
2. `validate_dataset.py`
3. `split_pedagogy_jsonl.py`
4. `train_qwen25_unsloth.py`
5. `eval_adapter.py` sur `pedagogy_dataset_test.jsonl`
6. `chat_model.py` — smoke test 5 prompts fixes

## Checklist opérationnelle (v3)

- [ ] Générer `pedagogy_dataset_v3.jsonl` (seed 100, all, 6 000 ex)
- [ ] Valider le schéma
- [ ] Vérifier que error_correction ≥ 25 % du train
- [ ] Splitter sans fuite
- [ ] Lancer train v3 depuis final
- [ ] Évaluer sur `pedagogy_dataset_test.jsonl`
- [ ] Gate perplexité : v3 ≤ 1.2254
- [ ] Smoke test 5/5 PASS
