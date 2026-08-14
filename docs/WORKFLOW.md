# Workflow — Fine-tuning de Gemma E2B pour le QA santé multilingue

> Méthode retenue : **fine-tuning d'un LLM instruction-tuned** (Gemma E2B), et non les baselines TF-IDF / mT5-seq2seq du notebook de départ. Ce document remplace le plan du notebook starter par un workflow adapté à un modèle **causal (decoder-only)** comme Gemma.
>
> ✅ Checkpoint confirmé : **`google/gemma-4-E2B-it`**. Chargement testé avec succès en 4-bit sur T4 (empreinte mémoire ≈ 6,7 Go) lors de l'exécution de `notebooks/00_setup_environment.ipynb` — voir Phase 0.

Voir [PROJET.md](PROJET.md) pour le contexte du challenge et [FICHIERS.md](FICHIERS.md) pour le détail des fichiers fournis.

---

## Phase 0 — Cadrage et environnement (détail pas à pas, Google Colab)

**Environnement retenu : Google Colab, GPU gratuit (T4, 16 Go de VRAM).** Suivre les étapes ci-dessous **dans l'ordre**, à faire au début de chaque nouvelle session Colab (une session ne conserve rien d'un jour à l'autre, sauf ce qui est sur Drive/Git).

### Étape 1 — Créer un dépôt Git pour versionner le travail
Sur GitHub (ou GitLab), créer un dépôt vide pour ce projet (notebooks, scripts, rapport). Le stockage local d'une session Colab est **éphémère** : tout ce qui n'est ni sur Drive ni committé sur Git disparaît à la déconnexion.

### Étape 2 — Ouvrir Google Colab et créer le notebook
Aller sur `colab.research.google.com` → *New notebook*. Le renommer clairement (ex. `gemma_finetune_health_qa.ipynb`).

### Étape 3 — Activer le GPU T4
Menu *Exécution* (Runtime) → *Modifier le type d'exécution* → Accélérateur matériel : **T4 GPU** → Enregistrer.

### Étape 4 — Vérifier le GPU alloué
Dans la première cellule :
```python
!nvidia-smi
```
Vérifier que la sortie mentionne bien un **Tesla T4** et ~15-16 Go de mémoire. Si Colab refuse d'allouer un GPU ("usage limit exceeded"), c'est une limitation du tier gratuit — réessayer plus tard (voir contraintes en fin de phase).

### Étape 5 — Monter Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```
Autoriser l'accès quand Colab le demande. Tout ce qui doit survivre à la session (CSV, checkpoints, logs) sera écrit ici, jamais uniquement dans `/content/`.

### Étape 6 — Créer l'arborescence du projet sur Drive
```python
import os

PROJECT_DIR = '/content/drive/MyDrive/gemmafro-e2b'
for sub in ['checkpoints', 'logs', 'submissions']:
    os.makedirs(f'{PROJECT_DIR}/{sub}', exist_ok=True)

print(os.listdir(PROJECT_DIR))
```
Pas de dossier `data/` ici : les CSV viennent directement du dépôt Git cloné à l'étape suivante, pas besoin de les dupliquer sur Drive.

### Étape 7 — Cloner le dépôt du projet
Les CSV, notebooks et docs vivent dans le dépôt Git ([github.com/andilMc/gemmafro-e2b](https://github.com/andilMc/gemmafro-e2b)). Le cloner directement dans Colab :
```python
!git clone https://github.com/andilMc/gemmafro-e2b.git /content/gemmafro-e2b
DATA_DIR = '/content/gemmafro-e2b/data'
```
Les CSV sont alors accessibles via `DATA_DIR` sans copie manuelle. Un notebook prêt à l'emploi pour toute la Phase 0 (étapes 4 à 14) est disponible dans [`notebooks/00_setup_environment.ipynb`](../notebooks/00_setup_environment.ipynb).

### Étape 8 — Installer les dépendances
À refaire en tête de notebook à **chaque nouvelle session** (l'environnement Python de Colab n'est pas persistant) :
```python
!pip install -q -U transformers accelerate peft bitsandbytes trl datasets rouge-score pandas
```

### Étape 9 — Créer un compte Hugging Face et un token d'accès
Si ce n'est pas déjà fait : compte sur `huggingface.co`, puis `Settings → Access Tokens → New token` (droits lecture suffisants).

### Étape 10 — Stocker le token HF dans les Colab Secrets (pas en clair dans le notebook)
Dans la barre latérale gauche de Colab, icône clé 🔑 → *Add new secret* → nom `HF_TOKEN`, valeur = le token copié à l'étape 9 → activer *Notebook access*. Puis :
```python
from google.colab import userdata
from huggingface_hub import login

login(token=userdata.get('HF_TOKEN'))
```

### Étape 11 — Accepter la licence du modèle Gemma sur Hugging Face
Ouvrir la page du modèle [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it) sur `huggingface.co` et cliquer sur *Agree and access repository*. Sans cette étape, le chargement du modèle échouera même avec un token valide.

### Étape 12 — Test de chargement minimal du modèle en 4-bit
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_NAME = "google/gemma-4-E2B-it"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,   # fp16, pas bf16 : T4 ne le supporte pas efficacement
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)
print(f"Empreinte mémoire : {model.get_memory_footprint() / 1e9:.2f} Go")
```
Si cette cellule tourne sans erreur mémoire, l'environnement est prêt pour la Phase 3 (fine-tuning).

### Étape 13 — Test rapide de génération (valider le chat template)
```python
messages = [{"role": "user", "content": "Bonjour, peux-tu répondre en français ?"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```
Vérifie que le tokenizer/chat template fonctionne correctement avant de construire le pipeline complet.

### Étape 14 — Vérifier l'accès aux données depuis Drive
```python
import pandas as pd

train = pd.read_csv(f'{PROJECT_DIR}/data/Train.csv')
print(train.shape)
train.head(3)
```

### Contraintes spécifiques au tier gratuit de Colab — à anticiper dès maintenant

- **Limite de session** : déconnexion automatique après ~90 minutes d'inactivité, et limite dure autour de 12h de session continue. Un entraînement plus long **sera interrompu**.
- **Disponibilité du GPU non garantie** : Google peut refuser l'attribution d'un GPU en cas d'usage intensif récent ("You have exceeded your GPU usage limit... try again later").
- **Conséquence directe sur le plan d'entraînement** (voir Phase 3) : checkpoints fréquents sauvegardés sur Drive (`save_strategy="steps"`, `save_steps` petit, `save_total_limit` pour ne pas saturer le quota Drive) et logique de **reprise** (`resume_from_checkpoint`) en cas de coupure. Ne jamais viser un entraînement monolithique de plusieurs heures sans point de sauvegarde intermédiaire.

## Phase 1 — Analyse exploratoire des données (EDA)

Notebook prêt à l'emploi : [`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb) (couvre les points ci-dessous, exécutable indépendamment de `00_setup_environment.ipynb`).

- [ ] Charger `Train.csv`, `Val.csv`, `Test.csv` et vérifier les types, valeurs manquantes, doublons d'ID.
- [ ] Confirmer la distribution par langue (`subset`) — déjà connue : très déséquilibrée (`Eng_Uga` ≈ 7 600 lignes vs `Amh_Eth` ≈ 1 800). **Ce déséquilibre est le premier risque du projet** : sans correction, le modèle fine-tuné répondra mieux en anglais et en akan qu'en amharique/luganda.
- [ ] Étudier la longueur des questions/réponses (en tokens, avec le tokenizer de Gemma) par langue — les scripts non-latins (amharique) et les langues agglutinantes peuvent produire des séquences beaucoup plus longues après tokenisation.
- [ ] Vérifier la tokenisation de chaque langue avec le tokenizer Gemma : compter le nombre moyen de tokens par mot. Si le tokenizer fragmente excessivement l'akan/le luganda/l'amharique, cela indique une couverture pré-entraînement faible pour ces langues → anticiper des séquences plus longues et un entraînement plus long pour converger.
- [ ] Repérer les réponses dupliquées ou "template" (ex. réponses très courtes du type "Yes"/"No"/"Ndiyo"/"Hapana" repérées lors de l'EDA initiale) — décider si elles doivent être filtrées, sous-échantillonnées ou conservées telles quelles.

## Phase 2 — Construction du dataset d'instruction-tuning

Contrairement à un seq2seq (mT5), Gemma attend un **format de chat** avec des rôles (`user` / `model`). Étapes :

- [ ] Définir un template de prompt unique, incluant le **nom de la langue cible** (comme dans le notebook starter) pour conditionner explicitement la génération, par ex. :
  ```
  Réponds à la question de santé suivante en {langue}, de façon claire et médicalement fiable.

  Question : {input}
  ```
- [ ] Mettre en forme chaque exemple avec `tokenizer.apply_chat_template()` (rôle `user` = prompt, rôle `model` = `output`), pour respecter exactement les tokens spéciaux attendus par Gemma à l'inférence.
- [ ] Masquer la perte (`label = -100`) sur la partie prompt/instruction : le modèle ne doit être pénalisé que sur la génération de la réponse, pas sur la reformulation de la question. `trl.SFTTrainer` gère cela nativement via `DataCollatorForCompletionOnlyLM`.
- [ ] Découper `Train.csv` en train/dev interne (ex. 95/5) **stratifié par `subset`**, en gardant `Val.csv` intact comme jeu de validation final (fidèle à l'évaluation Zindi).
- [ ] Envisager un **rééquilibrage** des langues sous-représentées (sur-échantillonnage d'Amh_Eth/Swa_Ken/Eng_Ken/Eng_Gha, ou pondération de la perte par langue) pour compenser le déséquilibre observé en Phase 1.

## Phase 3 — Fine-tuning (LoRA / QLoRA)

- [ ] Charger le modèle en 4-bit (`bitsandbytes`, `BitsAndBytesConfig`) pour économiser la VRAM.
- [ ] Configurer LoRA (`peft.LoraConfig`) : cibler les projections d'attention et du MLP (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`), rang typique `r=16` à `64`, `lora_alpha = 2×r`, `dropout ≈ 0.05`.
- [ ] Entraîner avec `trl.SFTTrainer` (ou `transformers.Trainer` + collator maison) :
  - `learning_rate` ≈ 1e-4 à 2e-4 (LoRA tolère un LR plus élevé qu'un fine-tuning complet),
  - 2-4 epochs (surveiller l'overfitting vu la taille limitée du jeu),
  - `per_device_train_batch_size` petit (1-4) + `gradient_accumulation_steps` pour simuler un batch plus grand — nécessaire sur un T4 16 Go,
  - `fp16=True` (pas `bf16`, non supporté efficacement sur T4),
  - `gradient_checkpointing=True` pour réduire l'empreinte mémoire au prix d'un entraînement un peu plus lent (compromis pertinent sur GPU gratuit).
- [ ] Logger la loss train/eval (Weights & Biases ou simplement les logs du `Trainer`) et **sauvegarder les checkpoints directement sur Google Drive** (`output_dir` pointant vers `/content/drive/MyDrive/...`), avec `save_steps` rapprochés vu le risque de déconnexion Colab évoqué en Phase 0.
- [ ] Garder le meilleur checkpoint selon la loss de validation (`load_best_model_at_end=True`).
- [ ] En cas de déconnexion en cours d'entraînement : relancer la session, remonter Drive, recharger le modèle 4-bit + adaptateur LoRA depuis le dernier checkpoint sauvegardé, et reprendre (`trainer.train(resume_from_checkpoint=...)`) plutôt que de repartir de zéro.

## Phase 4 — Évaluation

- [ ] Réutiliser (ou adapter depuis le notebook starter) la fonction `compute_rouge` avec tokenizer whitespace (safe pour les écritures non-latines) pour calculer **ROUGE-1 F1** et **ROUGE-L F1** sur `Val.csv`.
- [ ] Décomposer les scores **par langue** (`subset`) — c'est le diagnostic le plus important : il révèle si le modèle sous-performe sur les langues à faibles ressources.
- [ ] Approximer localement la métrique **TargetLLM** (LLM-as-a-judge) en utilisant un modèle plus puissant (ex. un LLM disponible via API) pour noter un échantillon de réponses générées vs. réponses de référence, avant la soumission officielle sur Zindi.
- [ ] Comparer les résultats du fine-tuning à un **baseline zero-shot** (Gemma non fine-tuné avec le même prompt) pour quantifier le gain réel du fine-tuning — indispensable pour le rapport.

## Phase 5 — Analyse d'erreurs et itération

- [ ] Inspecter manuellement un échantillon de générations par langue (surtout amharique/luganda/akan) : hallucinations médicales, changement de langue inattendu (le modèle répond en anglais alors que la question est en amharique), réponses tronquées.
- [ ] Ajuster selon les observations : plus d'epochs, rang LoRA plus élevé, rééquilibrage des langues, longueur max de génération (`max_new_tokens`), paramètres de génération (`do_sample=False` + `num_beams` pour plus de fiabilité factuelle, ou légère température si les réponses sont trop répétitives).
- [ ] Documenter chaque itération (config, scores obtenus) pour pouvoir justifier les choix dans le rapport final.

## Phase 6 — Génération de la soumission finale

- [ ] Charger le meilleur checkpoint LoRA (fusionné ou chargé en adaptateur) et générer les réponses pour toutes les questions de `Test.csv`, avec le **même prompt** qu'à l'entraînement (langue résolue depuis `subset`).
- [ ] Nettoyer les sorties (retirer tokens spéciaux résiduels, espaces superflus).
- [ ] Construire le fichier de soumission au format exact de `SampleSubmission.csv` : colonnes `ID`, `TargetRLF1`, `TargetR1F1`, `TargetLLM`, les trois contenant la même réponse générée.
- [ ] Vérifier avant soumission : mêmes IDs que `Test.csv`, pas de valeurs vides, pas de doublons, encodage UTF-8 correct pour les scripts amharique/ge'ez.

## Phase 7 — Rédaction et soutenance (livrable de classe)

- [ ] Rapport structuré : contexte (cf. `PROJET.md`), données, méthode (pourquoi le fine-tuning de Gemma E2B plutôt que les baselines), résultats quantitatifs par langue, analyse d'erreurs, limites (biais de langue, taille du modèle, risques médicaux d'un LLM), pistes d'amélioration.
- [ ] Conserver traçabilité : scripts/notebooks versionnés, config des runs, courbes de loss, tableau comparatif baseline vs fine-tuned.
- [ ] (Optionnel) Soumettre effectivement le fichier sur la plateforme Zindi pour obtenir le score officiel sur le leaderboard.

---

## Risques à surveiller en priorité

1. **Couverture linguistique de Gemma** — les LLM généralistes sont surtout entraînés sur l'anglais et quelques langues à hautes ressources ; l'akan, le luganda et l'amharique sont probablement peu représentés. Le fine-tuning doit compenser cela, mais les gains seront sans doute plus limités sur ces langues.
2. **Déséquilibre du jeu d'entraînement** entre langues — sans rééquilibrage, risque fort de biais vers l'anglais/l'akan.
3. **Fiabilité médicale** — un LLM fine-tuné sur peu de données peut halluciner des informations de santé incorrectes ; à mentionner explicitement comme limite dans le rapport (le projet est éducatif, pas un dispositif médical réel).
4. **Instabilité du GPU gratuit Colab** — sessions limitées dans le temps, déconnexions, quota GPU non garanti. À gérer par des checkpoints fréquents sur Drive et une logique de reprise (cf. Phase 0 et Phase 3) ; prévoir aussi des runs plus courts/itératifs plutôt qu'un unique entraînement long.
