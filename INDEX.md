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

- `train_qwen25_lora.py`: entraînement LoRA local Qwen2.5-1.5B
- `chat_model.py`: chat interactif sur adapter fine-tuné

## Documentation

- `README.MD`: vue générale
- `DEMARRAGE_RAPIDE.md`: exécution rapide du pipeline
- `GUIDE_JSONL.md`: détails techniques et options
- `RESUME_PROJET.md`: résumé technique courant
- `SYNTHESE_FINALE.md`: synthèse opérationnelle

## Workflow recommandé

1. `generate_pedagogical_dataset.py`
2. `validate_dataset.py`
3. `split_pedagogy_jsonl.py`
4. `train_qwen25_lora.py`
5. `chat_model.py` + benchmark test set

## Checklist opérationnelle

- [ ] Générer `pedagogy_dataset.jsonl`
- [ ] Valider le schéma
- [ ] Splitter sans fuite
- [ ] Lancer le fine-tuning LoRA
- [ ] Tester en chat
- [ ] Évaluer sur `pedagogy_dataset_test.jsonl`

## Notes de version (docs)

- `session_opening` ajouté au dataset pédagogique
- `translation_with_explanation` renforcé (copie exacte + validation)
- Pipeline local LoRA priorisé sur les exemples OpenAI historiques
