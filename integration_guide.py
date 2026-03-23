#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Guide d'intégration et d'utilisation du dataset JSONL
Exemples avec différentes APIs et frameworks
"""

# ============================================================================
# 1. OPENAI API - Fine-tuning GPT-3.5-turbo
# ============================================================================

"""
Installation:
  pip install openai

Configuration:
  export OPENAI_API_KEY="sk-..."

Code:
"""

def example_openai_finetuning():
    from openai import OpenAI
    
    client = OpenAI()
    
    # Uploader le fichier d'entraînement
    with open("training_data_train.jsonl", "rb") as f:
        response = client.files.create(
            file=f,
            purpose="fine-tune"
        )
        training_file_id = response.id
    
    # Uploader le fichier de validation
    with open("training_data_val.jsonl", "rb") as f:
        response = client.files.create(
            file=f,
            purpose="fine-tune"
        )
        validation_file_id = response.id
    
    # Créer le fine-tuning job
    response = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        validation_file=validation_file_id,
        model="gpt-3.5-turbo",
        hyperparameters={
            "n_epochs": 3,
            "learning_rate_multiplier": 0.1
        }
    )
    
    job_id = response.id
    print(f"Fine-tuning job créé: {job_id}")
    
    # Monitorer le job
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        print(f"Status: {job.status}")
        
        if job.status in ["succeeded", "failed"]:
            break
    
    # Utiliser le modèle fine-tuné
    if job.status == "succeeded":
        model_id = job.fine_tuned_model
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "user", "content": "mi wile e pan."}
            ]
        )
        print(f"Réponse: {response.choices[0].message.content}")


# ============================================================================
# 2. HUGGING FACE - Fine-tuning avec Transformers
# ============================================================================

"""
Installation:
  pip install transformers torch datasets

Code:
"""

def example_huggingface_finetuning():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from transformers import Trainer, TrainingArguments
    from datasets import load_dataset
    import torch
    
    # Charger le model de base (ex: Helsinki-NLP/Opus-MT-tok-fr)
    model_name = "Helsinki-NLP/Opus-MT-tok-fr"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Charger le dataset
    dataset = load_dataset(
        "json",
        data_files={
            "train": "training_data_train.jsonl",
            "validation": "training_data_val.jsonl"
        }
    )
    
    # Tokenizer le dataset
    def preprocess_function(examples):
        inputs = [ex["prompt"] for ex in examples["prompt"]]
        targets = [ex["completion"] for ex in examples["completion"]]
        
        model_inputs = tokenizer(
            inputs,
            max_length=128,
            truncation=True,
            padding="max_length"
        )
        
        labels = tokenizer(
            targets,
            max_length=128,
            truncation=True,
            padding="max_length"
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["prompt", "completion"]
    )
    
    # Configuration du training
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=5e-5,
        warmup_steps=500,
        weight_decay=0.01,
        save_total_limit=3,
        evaluation_strategy="epoch",
        logging_steps=100,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
    )
    
    # Fine-tune
    trainer.train()
    
    # Sauvegarder
    model.save_pretrained("./tok-fr-finetuned")
    tokenizer.save_pretrained("./tok-fr-finetuned")


# ============================================================================
# 3. PYTORCH LIGHTNING - Setup personnalisé
# ============================================================================

"""
Installation:
  pip install pytorch-lightning transformers torch

Code pour fine-tuning personnalisé:
"""

def example_pytorch_lightning():
    import pytorch_lightning as pl
    from torch.utils.data import DataLoader, Dataset
    import json
    import torch
    
    class TranslationDataset(Dataset):
        def __init__(self, jsonl_file, tokenizer, max_length=128):
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.data = []
            
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    record = json.loads(line)
                    self.data.append(record)
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            record = self.data[idx]
            
            source = self.tokenizer(
                record["prompt"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            target = self.tokenizer(
                record["completion"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            return {
                "input_ids": source["input_ids"].squeeze(),
                "attention_mask": source["attention_mask"].squeeze(),
                "labels": target["input_ids"].squeeze()
            }
    
    # Usage:
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/Opus-MT-tok-fr")
    train_dataset = TranslationDataset("training_data_train.jsonl", tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # ... continuer avec PyTorch Lightning


# ============================================================================
# 4. LLAMA FINE-TUNING - Avec llama-recipes
# ============================================================================

"""
Pour fine-tuning LLaMA 2:
  pip install git+https://github.com/facebookresearch/llama-recipes

Préparation:
  python -m llama_recipes.recipes.lora_finetune_single_gpu \
    --model_name meta-llama/Llama-2-7b \
    --dataset custom_dataset \
    --output_dir ./llama-tok-fr
"""


# ============================================================================
# 5. EVALUATION - Comparer la qualité
# ============================================================================

def evaluate_translation_quality():
    """
    Évaluer la qualité du fine-tuning avec BLEU, METEOR, etc.
    """
    import json
    from nltk.translate.bleu_score import sentence_bleu
    from nltk.translate.meteor_score import single_meteor_score
    
    with open("training_data_test.jsonl", "r", encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]
    
    bleu_scores = []
    meteor_scores = []
    
    # Assuquer que vous avez généré des prédictions
    for example in test_data[:100]:  # Évaluer sur 100 exemples
        reference = example["completion"]
        # prediction = model.generate(example["prompt"])  # À implémenter
        
        # Calcul BLEU
        reference_tokens = reference.split()
        # prediction_tokens = prediction.split()
        # bleu = sentence_bleu([reference_tokens], prediction_tokens)
        # bleu_scores.append(bleu)
    
    print(f"BLEU (échelle 0-1): {sum(bleu_scores)/len(bleu_scores):.3f}")
    print(f"METEOR (échelle 0-1): {sum(meteor_scores)/len(meteor_scores):.3f}")


# ============================================================================
# 6. BATCH INFERENCE - Traduire en masse
# ============================================================================

def batch_inference_openai():
    """
    Utiliser le modèle fine-tuné pour translater en batch
    """
    import json
    from openai import OpenAI
    
    client = OpenAI()
    
    # Lire les phrases à traduire
    with open("test_phrases.txt", "r", encoding='utf-8') as f:
        phrases = [line.strip() for line in f if line.strip()]
    
    # Configuration du modèle fine-tuné
    model_id = "ft:gpt-3.5-turbo:org-id::model-id"  # À remplacer par votre model_id
    
    results = []
    
    for phrase in phrases:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "user", "content": phrase}
            ],
            temperature=0.3
        )
        
        translation = response.choices[0].message.content
        results.append({
            "source": phrase,
            "translation": translation
        })
    
    # Sauvegarder les résultats
    with open("translations.jsonl", "w", encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"✓ {len(results)} phrases traduites et sauvegardées")


# ============================================================================
# 7. VALIDATION AVANT FINE-TUNING
# ============================================================================

def validate_before_finetuning():
    """
    Vérifications avant de démarrer le fine-tuning
    """
    import json
    from pathlib import Path
    
    checks = {
        "training_file_exists": Path("training_data_train.jsonl").exists(),
        "validation_file_exists": Path("training_data_val.jsonl").exists(),
        "test_file_exists": Path("training_data_test.jsonl").exists(),
    }
    
    # Vérifier le format JSONL
    try:
        with open("training_data_train.jsonl", "r", encoding='utf-8') as f:
            for i, line in enumerate(f):
                record = json.loads(line)
                if "prompt" not in record or "completion" not in record:
                    raise ValueError(f"Ligne {i}: Champs manquants")
                if i >= 100:  # Vérifier 100 lignes
                    break
        checks["jsonl_format_valid"] = True
    except Exception as e:
        checks["jsonl_format_valid"] = False
        checks["error"] = str(e)
    
    # Vérifier les sizes
    train_size = Path("training_data_train.jsonl").stat().st_size / (1024*1024)
    checks[f"train_size_mb"] = round(train_size, 1)
    
    # Afficher les résultats
    print("=" * 50)
    print("VALIDATION PRE-FINETUNING")
    print("=" * 50)
    for check, result in checks.items():
        status = "✓" if result else "✗"
        print(f"{status} {check}: {result}")
    
    all_passed = all(v for k, v in checks.items() if k != "error")
    print("\n" + ("✓ PRÊT POUR FINE-TUNING!" if all_passed else "✗ Corrections nécessaires"))


# ============================================================================
# 8. MONITORING - Tracker la performance
# ============================================================================

def monitor_finetuning():
    """
    Script pour monitorer les métriques pendant le fine-tuning
    """
    import time
    from openai import OpenAI
    
    client = OpenAI()
    job_id = "ftjob-xyz"  # À remplacer par votre job_id
    
    previous_step = 0
    
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        
        print(f"\n[{time.strftime('%H:%M:%S')}] Job: {job_id}")
        print(f"  Status: {job.status}")
        print(f"  Trained tokens: {job.trained_tokens}")
        
        if hasattr(job, "result_files"):
            print(f"  Result files: {job.result_files}")
        
        if job.status == "succeeded":
            print(f"✓ Fine-tuning complété!")
            print(f"  Model ID: {job.fine_tuned_model}")
            break
        elif job.status == "failed":
            print(f"✗ Fine-tuning échoué")
            break
        elif job.status == "cancelled":
            print(f"⊘ Fine-tuning annulé")
            break
        
        time.sleep(60)  # Vérifier chaque minute


# ============================================================================
if __name__ == "__main__":
    print("""
    ✨ Guide d'intégration du dataset tok↔fra
    
    Exemples fournis:
    1. OpenAI API (gpt-3.5-turbo)
    2. Hugging Face Transformers
    3. PyTorch Lightning (custom)
    4. LLaMA fine-tuning
    5. Évaluation (BLEU, METEOR)
    6. Batch inference
    7. Validation pré-fine-tuning
    8. Monitoring du job
    
    Consulter le код pour les implémentations détaillées!
    """)
    
    # Lancer la validation
    print("\nValidation du dataset...")
    validate_before_finetuning()
