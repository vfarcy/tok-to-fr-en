#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script d'analyse et validation du fichier JSONL de fine-tuning
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
import statistics

def analyze_jsonl(filepath, sample_size=10):
    """Analyse un fichier JSONL"""
    
    if not Path(filepath).exists():
        print(f"✗ Fichier non trouvé: {filepath}")
        return False
    
    print("="*70)
    print(f"Analyse du fichier JSONL: {filepath}")
    print("="*70)
    
    # Statistiques
    total_lines = 0
    total_bytes = 0
    prompt_lengths = []
    completion_lengths = []
    language_pairs = defaultdict(int)
    errors = []
    
    # Lire le fichier
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line.strip())
                    
                    if 'prompt' not in record or 'completion' not in record:
                        errors.append(f"Ligne {line_num}: Champs 'prompt' ou 'completion' manquants")
                        continue
                    
                    prompt = record['prompt']
                    completion = record['completion']
                    
                    total_lines += 1
                    total_bytes += len(line.encode('utf-8'))
                    
                    prompt_len = len(prompt.split())
                    completion_len = len(completion.split())
                    
                    prompt_lengths.append(prompt_len)
                    completion_lengths.append(completion_len)
                    
                    # Déterminer les langues par heuristique (simplifiée)
                    # Peut être améliorée avec langdetect
                    lang_pair = f"{prompt_len}→{completion_len} mots"
                    language_pairs[lang_pair] += 1
                    
                except json.JSONDecodeError as e:
                    errors.append(f"Ligne {line_num}: Erreur de parsing JSON - {str(e)[:50]}")
                    continue
    
    except Exception as e:
        print(f"✗ Erreur de lecture: {e}")
        return False
    
    if total_lines == 0:
        print("✗ Aucune ligne valide trouvée!")
        return False
    
    # Afficher les statistiques
    print(f"\n📊 STATISTIQUES GÉNÉRALES")
    print(f"  Paires totales: {total_lines:,}")
    print(f"  Taille du fichier: {total_bytes / (1024*1024):.1f} MB")
    print(f"  Taille moyenne par paire: {total_bytes / total_lines:.0f} bytes")
    
    if prompt_lengths:
        print(f"\n📏 LONGUEUR DES PHRASES SOURCE")
        print(f"  Moyenne: {statistics.mean(prompt_lengths):.1f} mots")
        print(f"  Médiane: {statistics.median(prompt_lengths):.1f} mots")
        print(f"  Min: {min(prompt_lengths)} mots")
        print(f"  Max: {max(prompt_lengths)} mots")
        if len(prompt_lengths) > 1:
            print(f"  Écart-type: {statistics.stdev(prompt_lengths):.1f}")
    
    if completion_lengths:
        print(f"\n📏 LONGUEUR DES PHRASES CIBLE")
        print(f"  Moyenne: {statistics.mean(completion_lengths):.1f} mots")
        print(f"  Médiane: {statistics.median(completion_lengths):.1f} mots")
        print(f"  Min: {min(completion_lengths)} mots")
        print(f"  Max: {max(completion_lengths)} mots")
        if len(completion_lengths) > 1:
            print(f"  Écart-type: {statistics.stdev(completion_lengths):.1f}")
    
    print(f"\n👀 EXEMPLES ALÉATOIRES ({min(sample_size, total_lines)} paires)")
    print("-" * 70)
    
    # Affiche quelques exemples
    import random
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    samples = random.sample(lines, min(sample_size, len(lines)))
    for i, line in enumerate(samples, 1):
        record = json.loads(line)
        prompt = record['prompt'][:60]
        completion = record['completion'][:60]
        
        # Ajouter "..." si texte tronqué
        if len(record['prompt']) > 60:
            prompt += "..."
        if len(record['completion']) > 60:
            completion += "..."
        
        print(f"\n  {i}. SOURCE: {prompt}")
        print(f"     CIBLE:  {completion}")
    
    # Erreurs
    if errors:
        print(f"\n⚠️  {len(errors)} erreurs détectées:")
        for error in errors[:5]:
            print(f"  • {error}")
        if len(errors) > 5:
            print(f"  ... et {len(errors) - 5} autres")
    else:
        print(f"\n✅ Aucune erreur détectée")
    
    # Résumé de santé
    print(f"\n{'='*70}")
    print("✓ FILE VALID FOR FINE-TUNING")
    print(f"{'='*70}\n")
    
    return True

def check_duplicates(filepath, check_pairs=False):
    """Vérifie les doublons dans le fichier"""
    print("🔍 Vérification des doublons...")
    
    prompts = defaultdict(int)
    completions = defaultdict(int)
    pairs = set()
    duplicates = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            prompt = record['prompt']
            completion = record['completion']
            
            prompts[prompt] += 1
            completions[completion] += 1
            
            pair = (prompt, completion)
            if pair in pairs:
                duplicates += 1
            pairs.add(pair)
    
    duplicate_prompts = sum(1 for v in prompts.values() if v > 1)
    duplicate_completions = sum(1 for v in completions.values() if v > 1)
    
    print(f"  Paires dupliquées: {duplicates}")
    print(f"  Prompts dupliqués: {duplicate_prompts}")
    print(f"  Complétions dupliquées: {duplicate_completions}")
    
    if duplicates == 0:
        print("  ✓ Aucun doublon détecté!")
    
    return duplicates

def main():
    """Point d'entrée principal"""
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python analyze_jsonl.py <fichier.jsonl> [--samples N] [--check-dupes]")
        print("\nExemples:")
        print("  python analyze_jsonl.py training_data.jsonl")
        print("  python analyze_jsonl.py training_data.jsonl --samples 20")
        print("  python analyze_jsonl.py training_data.jsonl --check-dupes")
        sys.exit(1)
    
    filepath = sys.argv[1]
    sample_size = 10
    check_dupes = False
    
    # Parser les arguments
    for arg in sys.argv[2:]:
        if arg.startswith('--samples'):
            sample_size = int(arg.split('=')[1]) if '=' in arg else 20
        elif arg == '--check-dupes':
            check_dupes = True
    
    # Analyser
    if analyze_jsonl(filepath, sample_size):
        if check_dupes:
            print("\n")
            check_duplicates(filepath)

if __name__ == "__main__":
    main()
