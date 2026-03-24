#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script optimisé pour générer un JSONL bidirectionnel pour fine-tuning
À partir de deux fichiers CSV pour la paire tok ↔ fra:
- sentences.csv: id, language, sentence
- links.csv: id1, id2 (liens entre phrases)

Stratégie:
1. Charger UNIQUEMENT les phrases tok et fra (pas tous les 13M)
2. Construire des chaînes de traduction efficaces
3. Générer les paires bidirectionnelles
"""

import csv
import json
from collections import defaultdict, deque
import sys

# Chemins des fichiers
SENTENCES_CSV = "sentences.csv"
LINKS_CSV = "links.csv"
OUTPUT_JSONL = "training_data.jsonl"

def load_sentences_selective(filepath, target_langs=None):
    """Charge UNIQUEMENT les phrases des langues spécifiées"""
    if target_langs is None:
        target_langs = {'tok', 'fra', 'eng'}  # Langues essentielles
    
    sentences = {}
    lang_count = defaultdict(int)
    
    print(f"Chargement sélectif des phrases depuis {filepath}...")
    print(f"  Langues ciblées: {target_langs}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 3:
                try:
                    sent_id = int(row[0])
                    lang = row[1].strip()
                    text = row[2].strip()
                    
                    if lang in target_langs and text:
                        sentences[sent_id] = {'lang': lang, 'text': text}
                        lang_count[lang] += 1
                except (ValueError, IndexError):
                    continue
    
    print(f"  Total phrases chargées: {len(sentences)}")
    for lang in sorted(lang_count.keys()):
        print(f"    - {lang}: {lang_count[lang]} phrases")
    
    return sentences, lang_count

def load_links_selective(filepath, sentence_ids):
    """Charge les liens qui connectent les phrases chargées"""
    sentence_ids_set = set(sentence_ids)
    links = defaultdict(set)
    
    print(f"Chargement des liens depuis {filepath}...")
    link_count = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 2:
                try:
                    id1, id2 = int(row[0]), int(row[1])
                    # Ne garder que les liens où AU MOINS UN des deux IDs est dans nos phrases
                    if id1 in sentence_ids_set or id2 in sentence_ids_set:
                        links[id1].add(id2)
                        links[id2].add(id1)
                        link_count += 1
                except ValueError:
                    continue
    
    print(f"  Liaisons chargées: {link_count}")
    return links

def find_all_paths_to_lang(start_id, target_lang, sentences, links, max_depth=4):
    """
    Trouve TOUS les chemins de traduction vers une langue cible.
    Utilise BFS avec mémoire d'état pour explorer efficacement.
    """
    if start_id not in sentences:
        return set()
    
    results = set()
    queue = deque([(start_id, 0)])
    visited = {(start_id, sentences[start_id]['lang'])}
    
    while queue:
        current_id, depth = queue.popleft()
        
        if depth > max_depth:
            continue
        
        current_lang = sentences[current_id]['lang']
        
        for neighbor_id in links.get(current_id, set()):
            if neighbor_id not in sentences:
                continue
            
            neighbor_lang = sentences[neighbor_id]['lang']
            
            # Éviter les cycles
            if (neighbor_id, neighbor_lang) in visited:
                continue
            
            visited.add((neighbor_id, neighbor_lang))
            
            # Si c'est la langue cible
            if neighbor_lang == target_lang:
                results.add(neighbor_id)
            
            # Continuer l'exploration
            if depth < max_depth:
                queue.append((neighbor_id, depth + 1))
    
    return results

def generate_translation_pairs(sentences, links, source_lang, target_lang):
    """
    Génère les paires de traduction, incluant les chaînes indirectes.
    Utilise les chemins de traduction pour enrichir les données.
    """
    pairs = set()
    
    # Obtenir tous les IDs de la langue source
    source_ids = [
        sid for sid, data in sentences.items() 
        if data['lang'] == source_lang
    ]
    
    print(f"\nGénération des paires {source_lang} → {target_lang}...")
    print(f"  Phrases source ({source_lang}): {len(source_ids)}")
    
    total = len(source_ids)
    for i, source_id in enumerate(source_ids):
        if i % max(1, total // 20) == 0:
            percent = int((i / total) * 100)
            print(f"  Progression: {percent}% ({i}/{total})")
        
        source_text = sentences[source_id]['text']
        
        # Trouver tous les chemins vers la langue cible
        target_ids = find_all_paths_to_lang(
            source_id, target_lang, sentences, links, max_depth=4
        )
        
        for target_id in target_ids:
            target_text = sentences[target_id]['text']
            # Ignorer les doublons et les paires identiques
            if source_text != target_text:
                pairs.add((source_text, target_text))
    
    return pairs

def create_jsonl(output_file, pairs_forward, pairs_backward, source_lang, target_lang):
    """Crée un fichier JSONL avec les paires bidirectionnelles"""
    
    print(f"\nCréation du fichier JSONL: {output_file}")
    print(f"  Paires {source_lang} → {target_lang}: {len(pairs_forward)}")
    print(f"  Paires {target_lang} → {source_lang}: {len(pairs_backward)}")
    
    total_pairs = len(pairs_forward) + len(pairs_backward)
    print(f"  Total paires: {total_pairs}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Paires source → cible
        for source, target in pairs_forward:
            record = {
                "prompt": source,
                "completion": target
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        # Paires cible → source (bidirectionnel)
        for target, source in pairs_backward:
            record = {
                "prompt": target,
                "completion": source
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"✓ Fichier créé avec succès: {output_file}")
    print(f"  Paires d'entraînement: {total_pairs}")

def main():
    # Charger UNIQUEMENT tok, fra et eng
    sentences, lang_count = load_sentences_selective(SENTENCES_CSV, {'tok', 'fra', 'eng'})
    
    if 'tok' not in lang_count or 'fra' not in lang_count:
        print("Erreur: Il faut au minimum 'tok' ou 'fra' dans les données!")
        return
    
    links = load_links_selective(LINKS_CSV, sentences.keys())
    
    if not sentences or not links:
        print("Erreur: Pas assez de données chargées!")
        return
    
    # Configuration
    source_lang = 'tok'
    target_lang = 'fra'
    
    print(f"\nConfiguration:")
    print(f"  Langue source: {source_lang}")
    print(f"  Langue cible: {target_lang}")
    print(f"  Profondeur de recherche: 4 (chaînes de traduction indirectes)")
    
    # Vérifier la disponibilité des langues
    if source_lang not in lang_count:
        print(f"⚠️  Langue '{source_lang}' non disponible!")
        available = [l for l in lang_count if l != '\\N']
        print(f"  Langues disponibles: {available[:10]}...")
        return
    
    if target_lang not in lang_count:
        print(f"⚠️  Langue '{target_lang}' non disponible!")
        return
    
    # Générer les paires
    pairs_forward = generate_translation_pairs(sentences, links, source_lang, target_lang)
    pairs_backward = generate_translation_pairs(sentences, links, target_lang, source_lang)
    
    if not pairs_forward and not pairs_backward:
        print("⚠️  Aucune paire de traduction trouvée!")
        return
    
    # Créer le JSONL
    create_jsonl(OUTPUT_JSONL, pairs_forward, pairs_backward, source_lang, target_lang)

if __name__ == "__main__":
    main()
