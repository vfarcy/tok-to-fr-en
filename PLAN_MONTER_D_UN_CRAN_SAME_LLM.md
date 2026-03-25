# Plan v2 pour faire un meilleur prof (meme LLM, stack actuelle)

Objectif: ameliorer la qualite pedagogique et la fiabilite des traductions sans changer le modele de base.

Contraintes gardees:
- base model: Qwen/Qwen2.5-1.5B-Instruct
- donnees: sentences.csv + links.csv
- stack actuelle: generate/split/train/chat avec Unsloth + FlashAttention

Obsolete retire:
- plan centre sur train_qwen25_lora.py
- option --load-in-4bit (dans la stack Unsloth, on utilise --no-4bit si besoin)

## 1) Definition precise du succes

On garde un run seulement si les 3 conditions sont reunies:
1. metrique test (loss/perplexity) non regressive vs meilleur run precedent
2. qualite generationnelle stable sur les types critiques (translation_with_explanation, guided_dialogue)
3. comportement chat utile pour debutant (consignes courtes, correction claire, pas de hors-sujet)

## 2) Regle experimentale non negociable

Le test set est fige une fois pour toutes.

On reutilise strictement:
- pedagogy_dataset_test.jsonl

Interdit:
- regenerer le test set entre les runs A/B/C
- comparer des runs sur des tests differents

## 3) Pipeline en 3 runs (curriculum)

## Run A - Fondations (A0,A1)

But:
- consolider les patrons simples et frequents
- maximiser la robustesse debutant

### A.1 Generation

```bash
python generate_pedagogical_dataset.py \
  --sentences sentences.csv \
  --links links.csv \
  --output pedagogy_dataset_A01.jsonl \
  --depth 3 \
  --max-samples 6000 \
  --level A0,A1 \
  --seed 42
```

### A.2 Validation + split

```bash
python validate_dataset.py --jsonl pedagogy_dataset_A01.jsonl --schema schema.json
python split_pedagogy_jsonl.py pedagogy_dataset_A01.jsonl
```

### A.3 Train (Unsloth)

```bash
python train_qwen25_unsloth.py \
  --train-file pedagogy_dataset_A01_train.jsonl \
  --val-file pedagogy_dataset_A01_val.jsonl \
  --output-dir qwen25-1.5b-tokipona-unsloth-A01 \
  --epochs 2 \
  --max-length 384 \
  --batch-size 2 \
  --grad-accum 8 \
  --save-steps 200 \
  --early-stopping-patience 3
```

## Run B - Extension (A0,A1,A2)

But:
- etendre la couverture sans casser la base pedagogique

### B.1 Generation

```bash
python generate_pedagogical_dataset.py \
  --sentences sentences.csv \
  --links links.csv \
  --output pedagogy_dataset_A012.jsonl \
  --depth 3 \
  --max-samples 8000 \
  --level A0,A1,A2 \
  --seed 43
```

### B.2 Validation + split

```bash
python validate_dataset.py --jsonl pedagogy_dataset_A012.jsonl --schema schema.json
python split_pedagogy_jsonl.py pedagogy_dataset_A012.jsonl
```

### B.3 Train (curriculum depuis Run A)

```bash
python train_qwen25_unsloth.py \
  --train-file pedagogy_dataset_A012_train.jsonl \
  --val-file pedagogy_dataset_A012_val.jsonl \
  --output-dir qwen25-1.5b-tokipona-unsloth-A012 \
  --epochs 2 \
  --max-length 384 \
  --batch-size 2 \
  --grad-accum 8 \
  --save-steps 200 \
  --early-stopping-patience 3 \
  --resume-from-checkpoint qwen25-1.5b-tokipona-unsloth-A01/checkpoint-XXXX
```

## Run C - Stabilisation finale (all levels)

But:
- finaliser un adapter deployable
- stabiliser les reponses pedagogiques exigees

### C.1 Generation

```bash
python generate_pedagogical_dataset.py \
  --sentences sentences.csv \
  --links links.csv \
  --output pedagogy_dataset_final.jsonl \
  --depth 3 \
  --max-samples 5000 \
  --level all \
  --seed 44
```

### C.2 Validation + split

```bash
python validate_dataset.py --jsonl pedagogy_dataset_final.jsonl --schema schema.json
python split_pedagogy_jsonl.py pedagogy_dataset_final.jsonl
```

### C.3 Train final

```bash
python train_qwen25_unsloth.py \
  --train-file pedagogy_dataset_final_train.jsonl \
  --val-file pedagogy_dataset_final_val.jsonl \
  --output-dir qwen25-1.5b-tokipona-unsloth-final \
  --epochs 3 \
  --max-length 384 \
  --batch-size 2 \
  --grad-accum 8 \
  --save-steps 200 \
  --early-stopping-patience 3
```

Si VRAM limite:
- utiliser --batch-size 1 --grad-accum 16
- en dernier recours, ajouter --no-4bit

## 4) Evaluation obligatoire apres chaque run

## 4.1 Evaluation quantitative (test fige)

Commande type deja validee:

```bash
python -c "import json; d=json.load(open('eval_test_unsloth_metrics.json')); print(d)"
```

Minimum a enregistrer:
1. avg_loss
2. perplexity
3. nombre d'exemples testes

## 4.2 Evaluation generationnelle (qualite prof)

Mesurer sur un sous-ensemble fixe (ex: 100 exemples du test set):
1. exact match sur traduction attendue (quand la cible est explicite)
2. conformite pedagogique (consigne courte, correction explicite, recap utile)
3. taux de hors-sujet

## 4.3 Smoke chat en stream

```bash
python chat_model.py --adapter qwen25-1.5b-tokipona-unsloth-final
```

Verifier:
1. pas de boucle
2. reponses pedagogiques courtes
3. correction claire quand l'eleve se trompe

## 5) Criteres de decision (gates)

Promouvoir Run N vers N+1 seulement si:
1. perplexity test <= meilleur run precedent + 3%
2. exact match generationnel non regressif
3. chat smoke sans regression evidente

Arreter un run si:
1. eval_loss remonte sur 3 evaluations consecutives
2. qualite chat se degrade clairement
3. gain marginal devient negligeable

## 6) Recommandations data (priorite "meilleur prof")

1. garder un coeur A0/A1 majoritaire
2. conserver A2 pour la generalisation, sans le surponderer
3. conserver session_opening
4. conserver translation_with_explanation renforce
5. eviter d'injecter des formulations trop longues pour debutants

## 7) Gestion des artefacts et traçabilité

Ne pas versionner:
1. dossiers d'adapters/checkpoints
2. caches de compilation

Versionner:
1. scripts
2. docs
3. fichiers de resultats de benchmark legers (json de metriques)

## 8) Checklist execution (precise)

1. [ ] Run A genere / valide / split / train
2. [ ] Eval quantitative A enregistree
3. [ ] Eval generationnelle A enregistree
4. [ ] Run B genere / valide / split / train
5. [ ] Eval quantitative B enregistree
6. [ ] Eval generationnelle B enregistree
7. [ ] Run C genere / valide / split / train
8. [ ] Eval quantitative C enregistree
9. [ ] Eval generationnelle C enregistree
10. [ ] Choix final de l'adapter (preuves a l'appui)

## 9) Resultat attendu

Sans changer de LLM, ce plan doit produire:
1. un prof plus stable en conversation
2. de meilleures corrections pour debutants
3. une meilleure exactitude des traductions dans le cadre pedagogique cible

## 10) Runbook execution (Run A, copier-coller)

Ce runbook execute uniquement le Run A de bout en bout, avec controles rapides.

### 10.1 Preparation environnement

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate tok-to-fr-en

python -V
nvidia-smi
python -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
```

Attendu:
1. environnement actif: tok-to-fr-en
2. GPU visible
3. import flash_attn OK

### 10.2 Generation dataset A0/A1

```bash
python generate_pedagogical_dataset.py \
  --sentences sentences.csv \
  --links links.csv \
  --output pedagogy_dataset_A01.jsonl \
  --depth 3 \
  --max-samples 6000 \
  --level A0,A1 \
  --seed 42
```

Controle:

```bash
wc -l pedagogy_dataset_A01.jsonl
```

### 10.3 Validation schema + split

```bash
python validate_dataset.py --jsonl pedagogy_dataset_A01.jsonl --schema schema.json
python split_pedagogy_jsonl.py pedagogy_dataset_A01.jsonl
```

Controle:

```bash
wc -l pedagogy_dataset_A01_train.jsonl pedagogy_dataset_A01_val.jsonl pedagogy_dataset_A01_test.jsonl
```

### 10.4 Train Unsloth (Run A)

```bash
python train_qwen25_unsloth.py \
  --train-file pedagogy_dataset_A01_train.jsonl \
  --val-file pedagogy_dataset_A01_val.jsonl \
  --output-dir qwen25-1.5b-tokipona-unsloth-A01 \
  --epochs 2 \
  --max-length 384 \
  --batch-size 2 \
  --grad-accum 8 \
  --save-steps 200 \
  --early-stopping-patience 3
```

Si OOM:

```bash
python train_qwen25_unsloth.py \
  --train-file pedagogy_dataset_A01_train.jsonl \
  --val-file pedagogy_dataset_A01_val.jsonl \
  --output-dir qwen25-1.5b-tokipona-unsloth-A01 \
  --epochs 2 \
  --max-length 384 \
  --batch-size 1 \
  --grad-accum 16 \
  --save-steps 200 \
  --early-stopping-patience 3
```

### 10.5 Evaluation quantitative (test fige)

Commande recommandee (pipeline deja valide dans ce repo):

```bash
python - <<'PY'
import json, math
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = "Qwen/Qwen2.5-1.5B-Instruct"
adapter = "qwen25-1.5b-tokipona-unsloth-A01"
test_file = Path("pedagogy_dataset_test.jsonl")

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(
    base_model,
    trust_remote_code=True,
    device_map="auto",
    dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
)
model = PeftModel.from_pretrained(base, adapter)
model.eval()

texts = []
with test_file.open("r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        txt = tokenizer.apply_chat_template(obj["messages"], tokenize=False, add_generation_prompt=False)
        texts.append(txt)

batch_size = 2
loss_sum = 0.0
n_batches = 0
n_tokens = 0

for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    input_ids = enc["input_ids"].to(model.device)
    attn = enc["attention_mask"].to(model.device)
    labels = input_ids.clone()
    labels[attn == 0] = -100
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
    loss_sum += out.loss.detach().float().item()
    n_batches += 1
    n_tokens += int((attn == 1).sum().item())

avg_loss = loss_sum / max(1, n_batches)
ppl = math.exp(avg_loss)
result = {
    "adapter": adapter,
    "test_file": str(test_file),
    "num_examples": len(texts),
    "avg_loss": avg_loss,
    "perplexity": ppl,
    "evaluated_tokens": n_tokens,
}
Path("eval_test_unsloth_A01_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
print("saved=eval_test_unsloth_A01_metrics.json")
PY
```

### 10.6 Smoke chat stream

```bash
python chat_model.py --adapter qwen25-1.5b-tokipona-unsloth-A01
```

Checks minimaux:
1. reponse coherent debutant
2. pas de boucle immediate
3. correction explicite quand la reponse eleve est fausse

### 10.7 Artefacts a conserver

Conserver:
1. eval_test_unsloth_A01_metrics.json
2. notes qualitatives chat (court resume)

Ne pas versionner:
1. qwen25-1.5b-tokipona-unsloth-A01/

### 10.8 Condition pour passer au Run B

Passer au Run B si:
1. run termine sans erreur critique
2. metriques test enregistrees
3. smoke chat valide
