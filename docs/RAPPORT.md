# Rapport de projet — Fine-tuning d'un LLM pour le question-réponse santé multilingue en langues africaines à faibles ressources

**Master 2 Business Intelligence — UCAD**
**Projet de classe** (basé sur le challenge Zindi *Multilingual Health Question Answering in Low-Resource African Languages Challenge*, organisé en collaboration avec HASH et Makerere University — ce projet n'a pas été soumis officiellement au challenge, il s'agit d'un exercice académique utilisant le même jeu de données)

---

## 1. Introduction et contexte

### 1.1 Le problème adressé

L'accès à une information de santé fiable reste très inégal en Afrique subsaharienne, en particulier pour les questions de santé maternelle, sexuelle et reproductive (MSRH). Le consortium **HASH** (*Hub for Artificial Intelligence in Maternal, Sexual and Reproductive Health*), qui réunit l'Infectious Diseases Institute, Sunbird AI et Makerere University College of Computing and Information Sciences (Ouganda), a documenté plusieurs constats préoccupants qui motivent ce travail :

- Les taux de maternité précoce restent parmi les plus élevés au monde en Afrique subsaharienne (93 naissances pour 1 000 filles de 15-19 ans en 2023, source UNICEF).
- Plus d'un million d'infections sexuellement transmissibles (IST) sont contractées chaque jour dans le monde, majoritairement en Afrique, avec des niveaux croissants de résistance aux traitements.
- La majorité des ressources d'information de santé numériques existantes ne sont disponibles qu'en anglais ou en français, alors que les populations concernées s'expriment dans des langues locales (akan, luganda, swahili, amharique, etc.).

Dans ce contexte, HASH a lancé, en collaboration avec Zindi et l'ITU, un défi de data science : **Multilingual Health Question Answering in Low-Resource African Languages Challenge**, visant à développer des modèles capables de répondre à des questions de santé dans la langue même de l'utilisateur.

### 1.2 Objectif du projet

Ce projet de classe reprend le jeu de données et la tâche de ce challenge à des fins pédagogiques (sans participation officielle au leaderboard Zindi). L'objectif est de construire, documenter et évaluer un pipeline complet de **fine-tuning d'un grand modèle de langage (LLM)** — `google/gemma-4-E2B-it` — pour la génération de réponses de santé dans 5 langues : **anglais, akan, luganda, swahili et amharique**.

La démarche méthodologique complète (analyse exploratoire, préparation des données, fine-tuning, évaluation) est documentée pas à pas dans le dépôt du projet : [github.com/andilMc/gemmafro-e2b](https://github.com/andilMc/gemmafro-e2b).

---

## 2. Données

### 2.1 Structure et volumétrie

Le jeu de données fourni par les organisateurs est structuré en trois fichiers :

| Fichier | Rôle | Lignes | Colonnes |
|---|---|---|---|
| `Train.csv` | Entraînement | 29 815 | `ID`, `input` (question), `output` (réponse de référence), `subset` |
| `Val.csv` | Validation | 6 686 | idem |
| `Test.csv` | Test (réponses à générer) | 2 618 | `ID`, `input`, `subset` |

La colonne `subset` encode la langue et le pays d'origine sous la forme `<LangCode>_<CountryCode>` (ex. `Amh_Eth`). Cinq langues sont couvertes, réparties sur huit combinaisons langue-pays :

| Code langue | Langue | Pays associé(s) |
|---|---|---|
| Eng | Anglais | Ghana, Ouganda, Kenya, Éthiopie |
| Aka | Akan | Ghana |
| Lug | Luganda | Ouganda |
| Swa | Swahili | Kenya |
| Amh | Amharique | Éthiopie |

### 2.2 Analyse exploratoire des données (EDA)

Une analyse exploratoire complète a été menée avant toute modélisation (notebook `01_eda.ipynb`). Elle a mis en évidence quatre constats structurants pour la suite du projet :

**a) Aucun problème de qualité basique.** Aucune valeur manquante, aucun ID dupliqué dans `Train.csv`, `Val.csv` ou `Test.csv`.

**b) Un déséquilibre marqué entre langues.** La distribution de `Train.csv` est très inégale : `Eng_Uga` représente à elle seule 25,6 % des lignes (7 624), contre seulement 6,2 % pour `Amh_Eth` (1 845 lignes) — un rapport de près de 4,1 entre la langue majoritaire et la langue minoritaire :

| Subset | Train | Val | Test |
|---|---|---|---|
| Eng_Uga | 7 624 | 1 688 | 744 |
| Aka_Gha | 4 455 | 1 114 | 492 |
| Eng_Gha | 4 443 | 1 104 | 491 |
| Eng_Eth | 3 915 | 564 | 60 |
| Lug_Uga | 3 383 | 846 | 374 |
| Eng_Ken | 2 080 | 390 | 167 |
| Swa_Ken | 2 070 | 518 | 229 |
| Amh_Eth | 1 845 | 462 | 61 |

**c) Une hétérogénéité forte de la longueur et de la tokenisation selon la langue.** En tokenisant les questions/réponses avec le tokenizer de Gemma, deux phénomènes distincts sont apparus :

- Les réponses en **akan** et **luganda** sont nettement plus longues que les autres (moyenne de 233 et 245 tokens respectivement, avec des maximums proches de 1 000-1 060 tokens), contre 32 à 121 tokens en moyenne pour les variantes anglaises.
- Le **ratio tokens/mot** (nombre de tokens produits par mot du texte) est beaucoup plus élevé pour le luganda (3,06), l'amharique (3,01), l'akan (2,24) et le swahili (2,09) que pour l'anglais (1,23 à 1,34). Cela indique que le tokenizer de Gemma **fragmente fortement** ces langues à faibles ressources — signe d'une couverture insuffisante lors du pré-entraînement du modèle — ce qui a deux conséquences directes : un coût de calcul plus élevé à contenu égal, et une tâche de génération intrinsèquement plus difficile pour le modèle dans ces langues.

**d) Un fort taux de réponses dupliquées (39,4 % de `Train.csv`).** Ce chiffre élevé s'explique par la nature du dataset (de type FAQ santé), où plusieurs questions proches partagent une réponse canonique identique. Il ne s'agit pas de lignes corrompues (les ID restent tous uniques), mais ce constat a été noté comme facteur à surveiller (risque de biais vers des réponses "génériques").

Ces observations ont directement orienté les choix méthodologiques de la Phase 2 (rééquilibrage des langues, détermination empirique de la longueur maximale de séquence) détaillés ci-dessous.

---

## 3. Méthodologie

### 3.1 Choix de l'approche : fine-tuning plutôt que les baselines fournies

Le notebook de démarrage fourni par les organisateurs proposait deux baselines : une recherche par similarité TF-IDF (retrieval de la réponse la plus proche dans le jeu d'entraînement) et un modèle seq2seq multilingue (mT5/NLLB). Ce projet s'en écarte délibérément pour explorer une troisième voie : le **fine-tuning d'un LLM instruction-tuned récent, décodeur causal** (`google/gemma-4-E2B-it`), via la technique **LoRA/QLoRA**.

Ce choix se justifie par :
- La capacité d'un LLM instruction-tuned à produire des réponses fluides et contextuellement appropriées, potentiellement supérieures à un simple retrieval TF-IDF qui ne peut que recopier une réponse existante.
- La faisabilité technique sur un GPU accessible (Google Colab) grâce à la quantification 4-bit (QLoRA), qui permet d'entraîner un modèle de plusieurs milliards de paramètres sans matériel professionnel.
- L'intérêt pédagogique : ce choix expose à l'ensemble de l'écosystème moderne du fine-tuning de LLM (Transformers, PEFT, bitsandbytes), plus représentatif des pratiques actuelles en NLP appliqué que les baselines classiques.

### 3.2 LoRA / QLoRA : principe et configuration

**LoRA** (*Low-Rank Adaptation*) consiste à geler les poids du modèle pré-entraîné et à n'entraîner que de petites matrices de rang réduit insérées dans certaines couches (les projections d'attention et du MLP), ce qui réduit drastiquement le nombre de paramètres entraînables. **QLoRA** ajoute à cela le chargement du modèle de base en 4-bit (quantification NF4), ce qui divise par ~4 l'empreinte mémoire du modèle gelé.

Configuration retenue :
- Quantification : 4-bit NF4, double quantification activée (`bitsandbytes`)
- LoRA : rang `r=16`, `lora_alpha=32` (soit `alpha = 2×r`, un ratio usuel), `dropout=0.05`
- Modules ciblés : les projections d'attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`) et du MLP (`gate_proj`, `up_proj`, `down_proj`)
- Type de tâche : `CAUSAL_LM`

### 3.3 Préparation du dataset d'instruction-tuning (Phase 2)

Contrairement à un modèle seq2seq (mT5), Gemma est un modèle de **chat** attendant une mise en forme par tours de rôle (`user` / `model`). Le pipeline de préparation (notebook `02_build_dataset.ipynb`) a suivi les étapes suivantes :

1. **Template de prompt** incluant explicitement le nom complet de la langue cible (résolu depuis le code `subset`, ex. `Amh_Eth` → *Amharic*), pour conditionner la génération sur la langue attendue.
2. **Mise en forme via `tokenizer.apply_chat_template()`**, produisant deux textes par exemple : `prompt_text` (tour `user` seul, avec `add_generation_prompt=True`) et `full_text` (`user` + `model`).
3. **Masquage de la perte par longueur mesurée.** Plutôt que de repérer la fin du prompt par un pattern-matching sur le texte des tokens spéciaux (fragile, car dépendant du template exact), la perte est masquée sur les `len(tokenizer(prompt_text))` premiers tokens de `full_text`. Ce choix méthodologique s'est révélé décisif : l'exécution du pipeline a révélé que `gemma-4-E2B-it` utilise un template de tours `<|turn>user ... <turn|> <|turn>model ...`, différent du `<start_of_turn>` documenté pour d'autres versions de Gemma. Une approche par pattern-matching aurait échoué silencieusement sur ce nouveau modèle.
4. **Split train/dev interne** stratifié par langue (95 % / 5 %), en conservant `Val.csv` intact comme jeu de validation final. Résultat : 28 324 lignes d'entraînement interne, 1 491 lignes de développement.
5. **Rééquilibrage plafonné des langues sous-représentées.** Un rééquilibrage total sur la langue majoritaire (`Eng_Uga`) aurait dupliqué l'amharique plus de 4 fois — risque de mémorisation plutôt que de généralisation, d'autant plus que les séquences akan/luganda sont déjà longues (cf. §2.2c). Un sur-échantillonnage avec remise, plafonné à un facteur ×3 de la taille d'origine (et jamais au-delà de la taille de la langue majoritaire), a été appliqué. Résultat : un jeu d'entraînement final de **53 300 lignes**, avec un déséquilibre résiduel ramené d'un facteur ~4,1 à ~1,4 (langue majoritaire à 7 243, la plus rééquilibrée — l'amharique — passant de 1 753 à 5 259 lignes).
6. **Détermination empirique de la longueur maximale de séquence.** Plutôt que de fixer une valeur arbitraire, la distribution réelle des longueurs tokenisées de `full_text` a été mesurée sur le jeu rééquilibré : moyenne 195 tokens, médiane 150, 95ᵉ percentile à 477, maximum 1 137. `MAX_SEQ_LENGTH = 512` a été retenu (marge au-dessus du p95).

Le résultat de cette phase (fichiers `train_balanced.jsonl`, `dev_split.jsonl`, `val_formatted.jsonl`) est sauvegardé sur Google Drive pour être directement réutilisé par la phase d'entraînement, sans avoir à refaire la mise en forme à chaque session.

### 3.4 Environnement d'entraînement

Le projet a été mené entièrement sur **Google Colab**, en deux étapes :

1. **Phase initiale sur le GPU gratuit (Tesla T4, 16 Go de VRAM).** Contraintes fortes : sessions limitées dans le temps (~90 min d'inactivité, ~12h en continu), disponibilité du GPU non garantie, T4 ne supportant pas efficacement le calcul en `bf16` (architecture Turing) — d'où l'usage initial de `fp16`.
2. **Passage à Colab Pro (GPU L4, ~24 Go de VRAM, architecture Ada Lovelace)** en cours de projet, pour accélérer l'entraînement. Le L4 supporte nativement le `bf16`, plus stable numériquement que le `fp16`, sans nécessiter de changement d'écosystème logiciel (toujours CUDA + `bitsandbytes` + PEFT).

Une stratégie de **reprise automatique sur déconnexion** a été mise en place dès le départ (`transformers.trainer_utils.get_last_checkpoint`), avec sauvegarde fréquente des checkpoints directement sur Google Drive — indispensable étant donné les contraintes de sessions limitées, y compris sur Colab Pro. L'entraînement final s'est déroulé sur **plusieurs sessions Colab successives**, la reprise automatique permettant de poursuivre sans perte de progression.

### 3.5 Difficultés techniques rencontrées et solutions apportées

Le fine-tuning de `google/gemma-4-E2B-it` — un modèle très récemment publié — a nécessité de résoudre une série de problèmes de compatibilité, chacun diagnostiqué à partir des messages d'erreur obtenus lors des exécutions réelles. Cette section documente ces problèmes à des fins de traçabilité méthodologique.

| # | Problème | Cause | Solution retenue |
|---|---|---|---|
| 1 | `OutOfMemoryError` lors de la préparation du modèle pour l'entraînement quantifié | `peft.prepare_model_for_kbit_training()` convertit **tous** les paramètres fp16 non quantifiés (embeddings, `lm_head`, layer norms) en fp32 ; avec le grand vocabulaire de Gemma, cette conversion tente d'allouer ~8,75 Go supplémentaires | Remplacement par une préparation allégée (gel des poids + `gradient_checkpointing_enable()` + `enable_input_require_grads()`), suffisante pour LoRA puisque ces poids gelés n'ont pas besoin de la stabilité fp32 |
| 2 | `ValueError` : modules dispatchés sur CPU/disque | `device_map="auto"` a estimé (à tort, ou à cause de mémoire déjà occupée) que le modèle ne tenait pas entièrement sur le GPU, et `bitsandbytes` refuse ce mélange 4-bit + CPU sans configuration explicite | Forçage de `device_map={"": 0}` : le modèle tient dans ~6,7 Go en 4-bit, largement sous la VRAM disponible |
| 3 | `ValueError` : module cible non supporté par PEFT (`Gemma4ClippableLinear`) | Les projections d'attention de `gemma-4-E2B-it` sont enveloppées dans une classe custom `Gemma4ClippableLinear` (clipping d'activations, probablement pour la stabilité numérique), non reconnue par le mécanisme d'attachement de LoRA de PEFT | Fonction `unwrap_clippable_linears()` déballant chaque wrapper vers son `Linear4bit` interne (reconnu par PEFT) avant `get_peft_model()` — au prix de la désactivation de ce clipping pendant l'entraînement |
| 4 | `CastError` au chargement du dataset (`datasets.load_dataset`) | `train_balanced.jsonl` (4 colonnes) et `dev_split.jsonl` (6 colonnes) n'avaient pas le même schéma ; `datasets` tente d'unifier le schéma entre tous les fichiers passés dans un même appel `data_files` et échoue si les colonnes diffèrent | Chargement des deux fichiers via des appels `load_dataset` séparés, et alignement des colonnes sauvegardées en Phase 2 |
| 5 | `AttributeError` à la reprise depuis un checkpoint | En passant de `fp16` (T4) à `bf16` (L4), le dossier de checkpoints pointait toujours sur l'ancien run `fp16`, dont l'état contient un `scaler.pt` (GradScaler) inexistant en `bf16` (`accelerator.scaler` vaut `None`) | Utilisation d'un nouveau dossier de checkpoints par configuration de précision |
| 6 | `ValueError` : aucune colonne ne correspond à la signature du modèle | `Trainer` retire par défaut les colonnes ne correspondant pas aux arguments de `forward()` ; le collator maison lit `prompt_text`/`full_text` (texte brut), qui ne matchent aucun argument attendu | Ajout de `remove_unused_columns=False` dans `TrainingArguments` |
| 7 | `OutOfMemoryError` pendant le calcul de la loss (premier pas d'entraînement) | Le calcul de `cross_entropy` sur des logits de forme *batch × séquence × vocabulaire* (vocabulaire large chez Gemma) faisait déborder la VRAM du L4 à `per_device_train_batch_size=8` | Réduction du batch à 4 (accumulation de gradient ajustée pour conserver un batch effectif de 16) ; ajout de `model.config.use_cache = False`, nécessaire pour un gradient checkpointing correct |

Cette suite d'itérations illustre une réalité fréquente du travail avec des modèles récemment publiés : la documentation et l'écosystème logiciel (PEFT, `transformers`, `bitsandbytes`) n'ont pas toujours pleinement rattrapé les spécificités d'une nouvelle architecture, et une part significative du travail d'ingénierie consiste à diagnostiquer puis contourner ces incompatibilités.

### 3.6 Entraînement final

Configuration retenue pour l'entraînement final (GPU L4) :

| Paramètre | Valeur |
|---|---|
| Modèle de base | `google/gemma-4-E2B-it`, chargé en 4-bit (NF4) |
| Époques | 3 |
| Batch effectif | 16 (`per_device_train_batch_size=4` × `gradient_accumulation_steps=4`) |
| Taux d'apprentissage | 2×10⁻⁴ |
| Précision de calcul | `bf16` |
| Longueur max de séquence | 512 tokens |
| Rang LoRA | 16 (`alpha=32`, `dropout=0.05`) |

Résultat de l'entraînement : **9 996 pas** sur 3 époques, **loss d'entraînement finale de 0,668**, pour un temps de calcul cumulé d'environ 16,8 heures (réparti sur plusieurs sessions Colab, grâce à la reprise automatique sur checkpoint).

---

## 4. Résultats

### 4.1 Évaluation quantitative

L'évaluation (notebook `04_evaluate.ipynb`) compare, sur un échantillon de 500 exemples de `Val.csv`, le modèle **fine-tuné** au modèle **zero-shot** (même modèle de base, adaptateur LoRA désactivé via `model.disable_adapter()` — ce qui permet une comparaison rigoureuse sans recharger un second modèle). Les scores ROUGE sont calculés avec un tokenizer par espaces (indépendant de l'écriture, adapté aux scripts non-latins comme l'amharique).

**Résultats globaux :**

| Métrique | Zero-shot | Fine-tuné | Gain |
|---|---|---|---|
| ROUGE-1 F1 | 0,1294 | 0,3667 | ×2,8 |
| ROUGE-L F1 | 0,0794 | 0,3109 | ×3,9 |

**Résultats par langue :**

| Langue | ROUGE-1 (zero-shot) | ROUGE-1 (fine-tuné) | ROUGE-L (zero-shot) | ROUGE-L (fine-tuné) |
|---|---|---|---|---|
| Eng_Uga | 0,1850 | **0,5733** | 0,1004 | **0,5371** |
| Eng_Eth | 0,0702 | 0,4561 | 0,0530 | 0,4300 |
| Eng_Ken | 0,2043 | 0,4954 | 0,1127 | 0,4234 |
| Swa_Ken | 0,1965 | 0,3726 | 0,1223 | 0,3006 |
| Eng_Gha | 0,1852 | 0,3621 | 0,1115 | 0,2673 |
| Aka_Gha | 0,0952 | 0,2661 | 0,0740 | 0,1919 |
| Lug_Uga | 0,0069 | 0,1859 | 0,0061 | 0,1538 |
| Amh_Eth | 0,0512 | **0,1559** | 0,0402 | **0,1316** |

Le fine-tuning améliore les scores **sur toutes les langues sans exception**, confirmant l'efficacité de l'approche. Deux observations méritent d'être soulignées :

- Le gain relatif le plus spectaculaire concerne le **luganda** : le zero-shot obtenait un score quasi nul (0,0069 ROUGE-1), le modèle produisant des réponses dégénérées (répétitions incohérentes, cf. §4.2) faute de capacité native dans cette langue. Le fine-tuning multiplie ce score par 27, bien que le score absolu reste parmi les plus faibles.
- Les scores finaux confirment exactement la hiérarchie anticipée par l'EDA (§2.2c) : les langues à faible ressource et forte fragmentation de tokenisation (**amharique**, **luganda**, **akan**) affichent les scores fine-tunés les plus bas (0,156 à 0,266 en ROUGE-1), contre 0,362 à 0,573 pour les variantes anglaises. Ce résultat illustre concrètement la difficulté additionnelle que représente un déséquilibre de couverture linguistique du modèle pré-entraîné, malgré le rééquilibrage appliqué en Phase 2.

### 4.2 Évaluation qualitative

L'inspection manuelle d'exemples générés confirme et enrichit les résultats quantitatifs :

- **Le modèle zero-shot répond parfois dans une langue totalement différente de celle attendue** : une question en akan a obtenu une réponse en français ("*Voici une réponse en Akan, formulée...*"), une question en anglais une réponse démarrant en vietnamien. Le fine-tuning corrige systématiquement ce comportement : toutes les réponses inspectées après fine-tuning sont produites dans la langue attendue.
- **Le zero-shot peut produire des sorties dégénérées** sur les langues les moins représentées dans le pré-entraînement de Gemma : sur une question en luganda, la génération zero-shot a bouclé sur la répétition du mot "*kye*" des dizaines de fois consécutives — signe caractéristique d'un modèle hors de sa distribution d'entraînement. Après fine-tuning, la réponse devient un texte luganda cohérent et sur le sujet.
- **Une limite à signaler** : une réponse fine-tunée générée pour une question sur la PrEP (prophylaxie pré-exposition) contient la formule "*As of my last knowledge update in September 2021*" — un artefact typique des LLM généralistes, révélateur d'une réponse partiellement puisée dans les connaissances générales pré-entraînées du modèle plutôt qu'entièrement ancrée sur les données d'entraînement spécifiques au domaine.

### 4.3 Prédictions finales sur le jeu de test

Le pipeline complet a été exécuté jusqu'au bout (notebook `05_generate_submission.ipynb`) : le modèle fine-tuné a généré une réponse pour chacune des **2 618 questions de `Test.csv`**, sans erreur, avec les vérifications automatiques de format validées (IDs correspondants, pas de doublons, pas de valeurs vides). Un exemple par langue illustre la cohérence linguistique et thématique obtenue :

| Langue | Question (extrait) | Réponse générée (extrait) |
|---|---|---|
| Akan | *Fa nneɛma a wɔde bɛyɛ nkyerɛkyerɛ nneɛma...* | *Amanneɛbɔ ne Nsɛm a Wɔka Kyerɛ: Amanneɛbɔ ne nneɛma a wɔde bɛyɛ nkyerɛkyerɛ nneɛma...* |
| Amharique | *ክላሚዲያ ሳይታከም ቢቆይ በወንዶች ላይ የሚያስከትለው የረጅም ጊዜ ጉዳት ምንድን ነው?* | *የመካን መፍሰስ ሲያደንቅ፣ የሽንት ቱቦዎች እብጠት (urethritis) እና በከብት መራቢያ አካላት ላይ ጉዳት ሊያደርስ ይችላል።* |
| Anglais (Ouganda) | *Treatment for Gonorrhea?, please answer this using simple medical terms.* | *Gonorrhoa is caused by a bacteria called Neisseria Gonorrhea. Bacterial infections are treated by use of antibiotics...* |
| Luganda | *Bitundu ki eby'omubiri ebirina okuyonjebwa n'obwegendereza?* | *Obusannyalaze bw'omubiri bukyuka mu kiseera ky'okuvubuka kubanga omubiri gwo gwetegekera okukula...* |
| Swahili | *Je, mtu mwenye afya nzuri anaweza kuambukizwa virusi vya ukimwi/UKIMWI?* | *Mtu mwenye afya nzuri anaweza kuambukizwa virusi vya ukimwi/UKIMWI. Kuwa na afya nzuri kunakusaidia...* |

Ces exemples confirment, sur des données jamais vues pendant l'entraînement, ce qu'indiquaient déjà les scores ROUGE et l'inspection qualitative sur `Val.csv` : des réponses fluides, dans la langue attendue, et thématiquement pertinentes par rapport à la question posée. Le fichier complet des 2 618 prédictions est sauvegardé sur Google Drive (`submissions/submission_gemma4_e2b_finetuned.csv`) ; il n'a pas été soumis à la plateforme Zindi, ce projet étant un exercice académique.

---

## 5. Limites et pistes d'amélioration

- **Couverture linguistique résiduelle inégale.** Malgré le rééquilibrage, l'amharique et le luganda restent les langues les moins bien servies. Une piste d'amélioration serait un fine-tuning spécifique par langue, ou un rééquilibrage moins plafonné (au risque, à évaluer empiriquement, d'un sur-apprentissage sur des réponses dupliquées — cf. §2.2d).
- **Absence de validation médicale experte.** Les réponses générées n'ont pas été vérifiées par un professionnel de santé ; dans un contexte de déploiement réel, une telle validation serait indispensable avant toute mise à disposition du public (risque de désinformation médicale).
- **Métrique `TargetLLM` (LLM-as-a-judge) non mise en œuvre.** Cette métrique, utilisée par l'évaluation officielle du challenge Zindi, nécessite un accès API à un LLM juge externe, non configuré dans le cadre de ce projet académique. Un squelette de code est prévu dans `04_evaluate.ipynb` pour une mise en œuvre future.
- **Évaluation limitée à un échantillon de 500 lignes** de `Val.csv` (sur 6 686 disponibles), pour des raisons de temps de calcul ; une évaluation sur l'ensemble du jeu de validation renforcerait la robustesse statistique des résultats par langue, en particulier pour les langues les moins représentées (46 à 108 exemples dans l'échantillon actuel).
- **Artefacts de connaissances générales du modèle pré-entraîné** (cf. §4.2), qui pourraient être atténués par un prompt système plus contraignant ou un filtrage post-génération.

---

## 6. Conclusion

Ce projet a permis de construire, documenter et évaluer un pipeline complet de fine-tuning LoRA/QLoRA d'un LLM récent (`google/gemma-4-E2B-it`) pour la génération de réponses de santé dans cinq langues africaines à faibles ressources. La démarche a suivi une méthodologie structurée en sept phases (cadrage, analyse exploratoire, préparation du dataset, entraînement, évaluation, analyse d'erreurs, génération des prédictions finales), chacune documentée et reproductible via des notebooks dédiés.

Les résultats obtenus démontrent un **gain net et systématique du fine-tuning par rapport au modèle zero-shot** (ROUGE-1 multiplié par 2,8, ROUGE-L par 3,9 en moyenne), avec des améliorations particulièrement marquées sur les langues où le modèle de base était initialement quasi inopérant (luganda). L'écart de performance résiduel entre l'anglais et les langues locales à faibles ressources, anticipé dès l'analyse exploratoire par l'étude de la fragmentation de tokenisation, constitue le principal axe de progression identifié pour des travaux futurs.

Au-delà des résultats, ce projet a été l'occasion de traiter un ensemble représentatif de défis d'ingénierie propres au fine-tuning de LLM récents sur infrastructure limitée (gestion mémoire GPU, quantification, compatibilité des bibliothèques avec une architecture de modèle inédite), documentés en détail en section 3.5.

---

## Annexe — Structure du dépôt

```
gemmafro-e2b/
├── README.md
├── data/                              # Train.csv, Val.csv, Test.csv, SampleSubmission.csv
├── docs/
│   ├── PROJET.md                      # Objet et contexte du challenge
│   ├── FICHIERS.md                    # Détail des fichiers fournis
│   ├── WORKFLOW.md                    # Workflow détaillé, phase par phase
│   └── RAPPORT.md                     # Ce document
└── notebooks/
    ├── 00_setup_environment.ipynb     # Phase 0 — configuration Colab
    ├── 01_eda.ipynb                   # Phase 1 — analyse exploratoire
    ├── 02_build_dataset.ipynb         # Phase 2 — préparation du dataset
    ├── 03_finetune.ipynb              # Phase 3 — fine-tuning LoRA/QLoRA
    ├── 04_evaluate.ipynb              # Phase 4 — évaluation
    └── 05_generate_submission.ipynb   # Phase 6 — génération des prédictions finales
```
