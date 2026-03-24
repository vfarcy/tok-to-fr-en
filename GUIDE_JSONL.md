# Générateur JSONL Bidirectionnel pour Fine-tuning

## 📋 Vue d'ensemble

Ce projet génère un dataset JSONL de haute qualité pour fine-tuner des modèles de traduction. À partir des fichiers Tatoeba `sentences.csv` et `links.csv`, il crée des paires de traduction bidirectionnelles en exploitant **les chaînes de traduction indirectes** (ex: tok → eng → fra).

## 🎯 Caractéristiques principales

✅ **Bidirectionnel** : Génère les deux direaction (tok↔fra)  
✅ **Chaînes de traduction** : Exploite toutes les connexions (tok→eng→fra)  
✅ **Optimisé en mémoire** : Charge uniquement les langues nécessaires  
✅ **Traçabilité** : Affiche la progression et les statistiques  
✅ **Flexible** : Support de multiples paires de langues  

## 📊 Résultats pour tok ↔ fra

```
Phrases chargées:
  • tok: 74,577 phrases
  • fra: 710,037 phrases
  • eng: 2,016,191 phrases (pivot)

Paires générées:
  • tok → fra: 43,788 paires
  • fra → tok: 43,788 paires
  • Total: 87,576 paires

Fichier: training_data.jsonl (~10 MB)
```

## 🚀 Utilisation

### Option 1: Script simple (recommandé pour débuter)

```bash
python generate_jsonl.py
```

Génère `training_data.jsonl` avec la configuration par défaut (tok↔fra, profondeur=4).

### Option 2: Script avancé (configuration personnalisée)

```bash
# Configuration par défaut
python generate_jsonl_advanced.py

# Paire de langues personnalisée
python generate_jsonl_advanced.py --source eng --target fra

# Avec profondeur réducée (plus rapide, moins exhaustif)
python generate_jsonl_advanced.py --depth 2

# Configuration complète
python generate_jsonl_advanced.py \
  --source tok \
  --target fra \
  --depth 4 \
  --output mon_dataset.jsonl \
  --sentences sentences.csv \
  --links links.csv
```

### Options du script avancé

| Option | Court | Type | Défaut | Description |
|--------|-------|------|--------|-------------|
| `--source` | `-s` | str | tok | Langue source |
| `--target` | `-t` | str | fra | Langue cible |
| `--depth` | `-d` | int | 4 | Profondeur de recherche pour chaînes |
| `--output` | `-o` | str | training_data.jsonl | Fichier de sortie |
| `--sentences` | - | str | sentences.csv | Fichier d'entrée (phrases) |
| `--links` | - | str | links.csv | Fichier d'entrée (liens) |

## 📁 Format du fichier JSONL

Chaque ligne est un objet JSON avec deux champs:
- **prompt** : La phrase source
- **completion** : La phrase cible

```json
{"prompt": "sina kama lon tenpo ike.", "completion": "Tu arrives trop tard."}
{"prompt": "o!", "completion": "Sois mon invité !"}
{"prompt": "Tu arrives trop tard.", "completion": "sina kama lon tenpo ike."}
```

## 🔍 Comment fonctionnent les chaînes de traduction

Le script découvre les traductions **indirectes** via BFS jusqu'à une profondeur maximum:

```
Exemple avec profondeur=2:

toki pona → Anglais → Français
    tok        eng        fra
     ↓         ↓          ↓
 "hello"  →  "hello"  →  "bonjour"
```

**Résultat**: Une paire tok→fra créée même sans lien direct!

Cela **multiplie le nombre de paires** sans dupliquer les données sources.

## ⚙️ Détails techniques

### Algorithme

1. **Chargement sélectif** : Ne charge que les langues source et cible (+englais comme pont)
2. **Graphe de liaisons** : Construit un dictionnaire des connexions bidirectionnelles
3. **BFS pour chaînes** : Pour chaque phrase source, explore tous les chemins vers la langue cible
4. **Déduplication** : Utilise des `set()` pour éviter les paires dupliquées
5. **Export JSONL** : Format recommandé pour les APIs OpenAI et équivalentes

### Performance

- **Temps total** : ~15-20 minutes de traitement pour tok↔fra
- **Mémoire** : ~4-6 GB (beaucoup moins que charger tout le dataset)
- **Résult précis** : Traduction de phrases complètes contextualisées

## 📦 Dépendances

- Python 3.7+
- Modules standard: `csv`, `json`, `argparse`, `time`
- Aucune dépendance externe!

## 🐛 Troubleshooting

### "Aucune paire de traduction trouvée"
- Vérifier que les fichiers CSV existent et sont accessibles
- S'assurer que les codes langue (ex: 'tok', 'fra') sont corrects
- Vérifier que les fichiers contiennent effectivement des liens

### Les paires ne sont pas trouvées
- Augmenter `--depth` (par défaut 4)
- Vérifier que la langue existe: `grep 'tok' sentences.csv | head`

### Lenteur d'exécution
- Réduire `--depth` (3 ou 2)
- S'assurer que les fichiers sont sur un disque rapide (SSD)

## 📈 Cas d'usage

✅ Fine-tuning de modèles de traduction automatique  
✅ Augmentation de datasets bilingues  
✅ Création de ressources pour langues peu dotées  
✅ Évaluation de modèles de traduction  

## 📚 Source des données

[Tatoeba Project](https://tatoeba.org/) - 13+ millions de phrases multilingues alignées

## 📝 Licence

Voir la licence Tatoeba: https://tatoeba.org/en/terms_of_use

## ✨ Exemples supplémentaires

### Générer dataset eng↔fra
```bash
python generate_jsonl_advanced.py -s eng -t fra -o eng_fra_dataset.jsonl
```

### Exploration rapide avec profondeur réduite
```bash
python generate_jsonl_advanced.py -d 2 -o quick_test.jsonl
```

### Tous les langues disponibles
Lancer `python generate_jsonl.py` pour voir la liste complète affichée.

---

**Créé avec ❤️ pour les passionnés de traduction et de ML**
