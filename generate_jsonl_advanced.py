#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script avancé pour générer un JSONL bidirectionnel pour fine-tuning
Avec support de multiples paires de langues et configurations personnalisables.

Utilisation:
    python generate_jsonl_advanced.py --source tok --target fra --output training_data.jsonl --depth 4
"""

import csv
import json
import argparse
from collections import defaultdict
import time

class TranslationDataGenerator:
    def __init__(self, sentences_file, links_file, source_lang, target_lang, max_depth=4):
        self.sentences_file = sentences_file
        self.links_file = links_file
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_depth = max_depth
        self.sentences = {}
        self.links = defaultdict(set)
        self.lang_count = defaultdict(int)
        
    def load_sentences_selective(self, target_langs=None):
        """Charge UNIQUEMENT les phrases des langues spécifiées"""
        if target_langs is None:
            target_langs = {self.source_lang, self.target_lang, 'eng'}
        
        print(f"[1/4] Chargement sélectif des phrases...")
        print(f"      Langues ciblées: {target_langs}")
        
        start_time = time.time()
        
        with open(self.sentences_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) >= 3:
                    try:
                        sent_id = int(row[0])
                        lang = row[1].strip()
                        text = row[2].strip()
                        
                        if lang in target_langs and text:
                            self.sentences[sent_id] = {'lang': lang, 'text': text}
                            self.lang_count[lang] += 1
                    except (ValueError, IndexError):
                        continue
        
        elapsed = time.time() - start_time
        print(f"      ✓ {len(self.sentences):,} phrases chargées en {elapsed:.1f}s")
        for lang in sorted(self.lang_count.keys()):
            print(f"        • {lang}: {self.lang_count[lang]:,} phrases")
    
    def load_links_selective(self):
        """Charge les liens pertinents"""
        print(f"\n[2/4] Chargement des liens...")
        
        start_time = time.time()
        sentence_ids_set = set(self.sentences.keys())
        link_count = 0
        
        with open(self.links_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) >= 2:
                    try:
                        id1, id2 = int(row[0]), int(row[1])
                        if id1 in sentence_ids_set or id2 in sentence_ids_set:
                            self.links[id1].add(id2)
                            self.links[id2].add(id1)
                            link_count += 1
                    except ValueError:
                        continue
        
        elapsed = time.time() - start_time
        print(f"      ✓ {link_count:,} liaisons chargées en {elapsed:.1f}s")
    
    def find_all_paths_to_lang(self, start_id, target_lang):
        """Trouve tous les chemins vers une langue via BFS"""
        if start_id not in self.sentences:
            return set()
        
        results = set()
        queue = [(start_id, 0)]
        visited = {(start_id, self.sentences[start_id]['lang'])}
        
        while queue:
            current_id, depth = queue.pop(0)
            
            if depth > self.max_depth:
                continue
            
            for neighbor_id in self.links.get(current_id, set()):
                if neighbor_id not in self.sentences:
                    continue
                
                neighbor_lang = self.sentences[neighbor_id]['lang']
                
                if (neighbor_id, neighbor_lang) in visited:
                    continue
                
                visited.add((neighbor_id, neighbor_lang))
                
                if neighbor_lang == target_lang:
                    results.add(neighbor_id)
                
                if depth < self.max_depth:
                    queue.append((neighbor_id, depth + 1))
        
        return results
    
    def generate_translation_pairs(self, source_lang, target_lang):
        """Génère les paires de traduction"""
        pairs = set()
        
        source_ids = [
            sid for sid, data in self.sentences.items() 
            if data['lang'] == source_lang
        ]
        
        print(f"\n[3/4] Génération des paires {source_lang} → {target_lang}")
        print(f"      Phrases source: {len(source_ids):,}")
        
        start_time = time.time()
        total = len(source_ids)
        
        for i, source_id in enumerate(source_ids):
            if i % max(1, total // 20) == 0:
                percent = int((i / total) * 100)
                print(f"      Progression: {percent}% ({i:,}/{total:,})")
            
            source_text = self.sentences[source_id]['text']
            
            target_ids = self.find_all_paths_to_lang(source_id, target_lang)
            
            for target_id in target_ids:
                target_text = self.sentences[target_id]['text']
                if source_text != target_text:
                    pairs.add((source_text, target_text))
        
        elapsed = time.time() - start_time
        print(f"      ✓ {len(pairs):,} paires générées en {elapsed:.1f}s")
        return pairs
    
    def create_jsonl(self, output_file):
        """Génère le fichier JSONL bidirectionnel"""
        print(f"\n[4/4] Création du fichier JSONL...")
        
        pairs_forward = self.generate_translation_pairs(self.source_lang, self.target_lang)
        pairs_backward = self.generate_translation_pairs(self.target_lang, self.source_lang)
        
        if not pairs_forward and not pairs_backward:
            print("      ✗ Aucune paire de traduction trouvée!")
            return False
        
        print(f"\n      Détails:")
        print(f"      • {self.source_lang} → {self.target_lang}: {len(pairs_forward):,} paires")
        print(f"      • {self.target_lang} → {self.source_lang}: {len(pairs_backward):,} paires")
        print(f"      • Total: {len(pairs_forward) + len(pairs_backward):,} paires")
        
        start_time = time.time()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for source, target in pairs_forward:
                record = {
                    "prompt": source,
                    "completion": target
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            for target, source in pairs_backward:
                record = {
                    "prompt": target,
                    "completion": source
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        elapsed = time.time() - start_time
        file_size_mb = __import__('os').path.getsize(output_file) / (1024 * 1024)
        
        print(f"\n      ✓ Fichier créé: {output_file}")
        print(f"        Taille: {file_size_mb:.1f} MB")
        print(f"        Temps: {elapsed:.1f}s")
        
        return True
    
    def run(self, output_file):
        """Exécute tout le pipeline"""
        print("="*60)
        print(f"Générateur JSONL pour Fine-tuning")
        print(f"Source: {self.source_lang}, Cible: {self.target_lang}")
        print(f"Profondeur chaînes: {self.max_depth}")
        print("="*60)
        
        self.load_sentences_selective({self.source_lang, self.target_lang, 'eng'})
        
        if self.source_lang not in self.lang_count:
            print(f"\n✗ Langue '{self.source_lang}' non disponible!")
            return False
        if self.target_lang not in self.lang_count:
            print(f"\n✗ Langue '{self.target_lang}' non disponible!")
            return False
        
        self.load_links_selective()
        self.create_jsonl(output_file)
        
        print("\n" + "="*60)
        print("✓ Processus complété avec succès!")
        print("="*60)
        return True

def main():
    parser = argparse.ArgumentParser(
        description="Générateur JSONL bidirectionnel pour fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python generate_jsonl_advanced.py --source tok --target fra
  python generate_jsonl_advanced.py --source eng --target fra --depth 3
  python generate_jsonl_advanced.py --source cmn --target eng --output custom.jsonl
        """
    )
    
    parser.add_argument('-s', '--source', default='tok', 
                        help='Langue source (défaut: tok)')
    parser.add_argument('-t', '--target', default='fra', 
                        help='Langue cible (défaut: fra)')
    parser.add_argument('-d', '--depth', type=int, default=4,
                        help='Profondeur de recherche pour chaînes (défaut: 4)')
    parser.add_argument('-o', '--output', default='training_data.jsonl',
                        help='Fichier de sortie (défaut: training_data.jsonl)')
    parser.add_argument('--sentences', default='sentences.csv',
                        help='Fichier d\'entrée phrases (défaut: sentences.csv)')
    parser.add_argument('--links', default='links.csv',
                        help='Fichier d\'entrée liens (défaut: links.csv)')
    
    args = parser.parse_args()
    
    generator = TranslationDataGenerator(
        args.sentences,
        args.links,
        args.source,
        args.target,
        args.depth
    )
    
    success = generator.run(args.output)
    exit(0 if success else 1)

if __name__ == "__main__":
    main()
