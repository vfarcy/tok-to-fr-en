# 🚀 DÉMARRAGE RAPIDE

## En 5 minutes ⏱️

### 1. Vérifier le dataset (1 min)
```bash
# Afficher la première paire
head -n 1 training_data.jsonl

# Compter les paires
wc -l training_data.jsonl
```

Résultat attendu: **87,576 paires**

### 2. Analyser le dataset (2 min)
```bash
python analyze_jsonl.py training_data.jsonl --samples 10
```

Vous verrez:
- Statistiques: 87,576 paires, 7.3 MB
- Exemples: 10 paires aléatoires
- Validation: ✓ Aucune erreur

### 3. Utiliser avec OpenAI API (1 min)
```bash
# Installation
pip install openai

# Fine-tuning (voir integration_guide.py)
# Ou utiliser directement le CLI OpenAI:
openai api fine_tunes.create \
  -t training_data_train.jsonl \
  -v training_data_val.jsonl \
  -m gpt-3.5-turbo
```

### 4. Utiliser avec HuggingFace (1 min)
```bash
# Installation
pip install transformers torch datasets

# Code simple (voir integration_guide.py)
from datasets import load_dataset
dataset = load_dataset("json", data_files="training_data.jsonl")
# ... continuer avec Transformers Trainer
```

---

## Ce que vous avez ✅

| Item | Détail |
|------|--------|
| **Dataset principal** | training_data.jsonl (87,576 paires) |
| **Train/Val/Test** | Split 80/10/10 automatique |
| **Format** | JSONL standard (prêt pour fine-tuning) |
| **Qualité** | Paires réelles de Tatoeba + chaînes indirectes |
| **Bidirectionnel** | tok→fra ET fra→tok (50/50) |
| **Scripts** | 5 scripts Python pour tout faire |
| **Documentation** | 4 guides complets |

---

## Exemples de paires 📝

```json
{"prompt": "mi wile e pan.", "completion": "Je veux du pain."}
{"prompt": "o!", "completion": "Sois mon invité !"}
{"prompt": "sewi o!", "completion": "Bon Dieu !"}
{"prompt": "Merci.", "completion": "mi pana e pona."}
```

---

## Commandes essentielles 💻

```bash
# 1. Valider le dataset
python integration_guide.py

# 2. Analyser en détail
python analyze_jsonl.py training_data.jsonl --samples 20 --check-dupes

# 3. Régénérer pour autre paire
python generate_jsonl_advanced.py --source eng --target fra -o eng_fra.jsonl

# 4. Splitter différemment
python split_jsonl.py training_data.jsonl --train 0.7 --val 0.2 --test 0.1
```

---

## Prochaines étapes 📋

### Option 1: OpenAI (Recommandé)
1. ✅ Dataset prêt: `training_data_train.jsonl`
2. → Installer: `pip install openai`
3. → Copier le code de `integration_guide.py` (section 1)
4. → Lancer le fine-tuning
5. → Attendre ~30 min à 2h

### Option 2: HuggingFace
1. ✅ Dataset prêt: `training_data.jsonl`
2. → Installer: `pip install transformers torch`
3. → Copier le code de `integration_guide.py` (section 2)
4. → Fine-tune sur votre GPU
5. → Attendre ~2-6h selon matériel

### Option 3: CLI OpenAI (Plus simple!)
```bash
openai api fine_tunes.create \
  -t training_data_train.jsonl \
  -v training_data_val.jsonl \
  -m gpt-3.5-turbo
```

---

## FAQ Rapide ❓

**Q: Le dataset est-il directement utilisable?**
✅ OUI! Format JSONL standard, pas besoin de preprocessing

**Q: Dois-je utiliser les splits?**
✅ Oui, training_data_train.jsonl pour entraînement, val pour validation

**Q: Comment savoir si ça marche?**
📊 Comparer tok→fra vs fra→tok performance
   Tester sur test.jsonl
   Calculer BLEU/METEOR scores

**Q: Puis-je ajouter plus de données?**
✅ Oui! `python generate_jsonl_advanced.py` pour générer plus
   `split_jsonl.py` pour combiner et splitter

**Q: Quelle est la taille minimale/maximale?**
- Minimum: Même 10 paires peuvent améliorer un peu
- Maximum: Pas de limite, plus est mieux (jusqu'à un point)
- Optimal: Pour tok↔fra, 87,576 est très bon!

---

## Fichiers importants 📂

```
📦 Ton projet
├── training_data.jsonl              ← Dataset complet (87,576 paires)
├── training_data_train.jsonl        ← Pour entraîner 80%
├── training_data_val.jsonl          ← Pour valider 10%
├── training_data_test.jsonl         ← Pour tester 10%
├── generate_jsonl.py                ← Régénérer le dataset
├── analyze_jsonl.py                 ← Analyser/valider
├── split_jsonl.py                   ← Splitter autrement
├── integration_guide.py              ← Exemples fine-tuning
├── SYNTHESE_FINALE.md               ← Vue d'ensemble
├── GUIDE_JSONL.md                   ← Guide complet
└── INDEX.md                         ← Référence
```

---

## Validation en 30 secondes ⚡

```python
import json

# Vérifier rapidement
with open("training_data.jsonl") as f:
    lines = f.readlines()
    
print(f"Total paires: {len(lines)}")
print(f"Première paire:")
print(json.loads(lines[0]))
```

Sortie attendue:
```
Total paires: 87576
Première paire:
{'prompt': '...', 'completion': '...'}
```

---

## Dernier conseil 💡

**Ne pas overcomplicate!**

Le dataset est prêt. Vous pouvez immédiatement:
1. Copier `training_data_train.jsonl`
2. L'uploader à OpenAI/HuggingFace
3. Lancer le fine-tuning
4. Attendre les résultats

**C'est tout!** 🎉

---

## Contact / Questions

Voir `GUIDE_JSONL.md` pour:
- Troubleshooting détaillé
- Options avancées
- Cas d'usage spécifiques

Voir `INDEX.md` pour:
- Liste complète des fichiers
- Références rapides
- Commandes utiles

---

**Bonne chance avec ton modèle de traduction tok↔fra!** 🚀

Créé: Mars 2026
Status: ✅ Production-ready
