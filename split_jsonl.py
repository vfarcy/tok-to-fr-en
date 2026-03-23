#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilitaire pour splitter le fichier JSONL en ensembles train/validation/test
"""

import json
import random
import sys
from pathlib import Path

def split_jsonl(input_file, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, 
                output_dir_prefix="", shuffle=True, seed=42):
    """
    Split un fichier JSONL en trois ensembles
    
    Args:
        input_file: Chemin du fichier JSONL source
        train_ratio: Ratio d'entraînement (défaut: 0.8)
        val_ratio: Ratio de validation (défaut: 0.1)
        test_ratio: Ratio de test (défaut: 0.1)
        output_dir_prefix: Préfixe des fichiers de sortie
        shuffle: Mélanger les données (défaut: True)
        seed: Graine aléatoire (défaut: 42)
    """
    
    # Vérifier les ratios
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 0.001:
        print(f"✗ Les ratios doivent totaliser 1.0, reçu: {total}")
        return False
    
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"✗ Fichier non trouvé: {input_file}")
        return False
    
    print("="*70)
    print(f"Splitting du fichier JSONL: {input_file}")
    print(f"Ratios: train={train_ratio*100:.0f}%, val={val_ratio*100:.0f}%, test={test_ratio*100:.0f}%")
    print("="*70)
    
    # Charger toutes les lignes
    print("\n📖 Lecture du fichier...")
    lines = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    json.loads(line)
                    lines.append(line)
                except json.JSONDecodeError as e:
                    print(f"  ⚠️  Ligne {i}: JSON invalide - {str(e)[:50]}")
    except Exception as e:
        print(f"✗ Erreur de lecture: {e}")
        return False
    
    total_lines = len(lines)
    print(f"  ✓ {total_lines:,} lignes chargées")
    
    # Mélanger si demandé
    if shuffle:
        print("\n🔀 Mélange des données...")
        random.seed(seed)
        random.shuffle(lines)
    
    # Calculer les indices de split
    train_size = int(total_lines * train_ratio)
    val_size = int(total_lines * val_ratio)
    
    train_lines = lines[:train_size]
    val_lines = lines[train_size:train_size + val_size]
    test_lines = lines[train_size + val_size:]
    
    print(f"\n📊 Répartition:")
    print(f"  Train: {len(train_lines):,} ({len(train_lines)/total_lines*100:.1f}%)")
    print(f"  Val:   {len(val_lines):,} ({len(val_lines)/total_lines*100:.1f}%)")
    print(f"  Test:  {len(test_lines):,} ({len(test_lines)/total_lines*100:.1f}%)")
    
    # Écrire les fichiers
    suffix = input_path.suffix
    stem = input_path.stem
    
    train_file = f"{output_dir_prefix}{stem}_train{suffix}" if output_dir_prefix else f"{stem}_train{suffix}"
    val_file = f"{output_dir_prefix}{stem}_val{suffix}" if output_dir_prefix else f"{stem}_val{suffix}"
    test_file = f"{output_dir_prefix}{stem}_test{suffix}" if output_dir_prefix else f"{stem}_test{suffix}"
    
    print("\n💾 Écriture des fichiers...")
    
    try:
        # Train
        with open(train_file, 'w', encoding='utf-8') as f:
            f.writelines(train_lines)
        print(f"  ✓ {train_file}")
        
        # Val
        with open(val_file, 'w', encoding='utf-8') as f:
            f.writelines(val_lines)
        print(f"  ✓ {val_file}")
        
        # Test
        with open(test_file, 'w', encoding='utf-8') as f:
            f.writelines(test_lines)
        print(f"  ✓ {test_file}")
        
    except Exception as e:
        print(f"✗ Erreur d'écriture: {e}")
        return False
    
    print("\n" + "="*70)
    print("✓ Splitting complété avec succès!")
    print("="*70)
    
    return True

def merge_jsonl(input_files, output_file, shuffle=False, seed=42):
    """Merge plusieurs fichiers JSONL en un seul"""
    
    print("="*70)
    print(f"Fusion de {len(input_files)} fichiers JSONL")
    print("="*70)
    
    lines = []
    total = 0
    
    for input_file in input_files:
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                file_lines = [line for line in f if line.strip()]
                lines.extend(file_lines)
                print(f"  ✓ {input_file}: {len(file_lines):,} lignes")
                total += len(file_lines)
        except Exception as e:
            print(f"  ✗ {input_file}: {e}")
            return False
    
    if shuffle:
        print(f"\n🔀 Mélange des {total:,} lignes...")
        random.seed(seed)
        random.shuffle(lines)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print(f"\n✓ Fusion complétée: {output_file} ({total:,} lignes)")
        return True
    except Exception as e:
        print(f"✗ Erreur: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python split_jsonl.py <input.jsonl> [--train 0.8] [--val 0.1] [--test 0.1] [--shuffle] [--seed 42]")
        print("\nExemples:")
        print("  python split_jsonl.py training_data.jsonl")
        print("  python split_jsonl.py training_data.jsonl --train 0.7 --val 0.15 --test 0.15")
        print("  python split_jsonl.py training_data.jsonl --train 0.9 --val 0.05 --test 0.05 --shuffle --seed 123")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Parser les arguments
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    shuffle = True
    seed = 42
    output_prefix = ""
    
    for i, arg in enumerate(sys.argv[2:]):
        if arg == '--train' and i+2 < len(sys.argv):
            train_ratio = float(sys.argv[i+3])
        elif arg == '--val' and i+2 < len(sys.argv):
            val_ratio = float(sys.argv[i+3])
        elif arg == '--test' and i+2 < len(sys.argv):
            test_ratio = float(sys.argv[i+3])
        elif arg == '--shuffle':
            shuffle = True
        elif arg == '--no-shuffle':
            shuffle = False
        elif arg == '--seed' and i+2 < len(sys.argv):
            seed = int(sys.argv[i+3])
        elif arg == '--prefix' and i+2 < len(sys.argv):
            output_prefix = sys.argv[i+3]
    
    split_jsonl(input_file, train_ratio, val_ratio, test_ratio, 
                output_prefix, shuffle, seed)

if __name__ == "__main__":
    main()
