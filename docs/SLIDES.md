# Plan de slides — Présentation en classe

Plan slide par slide (titre + puces) à recopier dans PowerPoint / Google Slides. Contenu détaillé et justifications dans [RAPPORT.md](RAPPORT.md).

---

### Slide 1 — Titre
- **Fine-tuning d'un LLM pour le question-réponse santé multilingue en langues africaines à faibles ressources**
- Sous-titre : basé sur le challenge Zindi/HASH *Multilingual Health Question Answering in Low-Resource African Languages Challenge*
- Master 2 Business Intelligence — UCAD
- [Votre nom], [date]

---

### Slide 2 — Contexte
- HASH : consortium (IDI, Sunbird AI, Makerere University) pour l'IA en santé maternelle, sexuelle et reproductive (MSRH) en Afrique
- Chiffres clés : 93 naissances/1000 chez les 15-19 ans en Afrique subsaharienne (2023) ; >1M d'IST contractées chaque jour dans le monde, majoritairement en Afrique
- Problème : l'information de santé numérique existe surtout en anglais/français, pas dans les langues locales

---

### Slide 3 — Objectif du projet
- Développer et évaluer un modèle capable de répondre à des questions de santé **dans la langue de la question** : anglais, akan, luganda, swahili, amharique
- Projet académique (jeu de données du challenge Zindi, sans soumission officielle au leaderboard)
- Approche retenue : fine-tuning d'un LLM (`google/gemma-4-E2B-it`) via LoRA/QLoRA

---

### Slide 4 — Les données
- Train : 29 815 lignes / Val : 6 686 / Test : 2 618
- Colonnes : `ID`, `input` (question), `output` (réponse), `subset` (langue_pays)
- 5 langues, 8 combinaisons langue-pays (ex. Eng_Uga, Aka_Gha, Amh_Eth...)
- Aucune valeur manquante, aucun ID dupliqué

---

### Slide 5 — EDA : déséquilibre des langues
- Eng_Uga = 25,6 % du Train (7 624 lignes) vs Amh_Eth = 6,2 % (1 845 lignes)
- Rapport ~4,1 entre langue majoritaire et minoritaire
- *(insérer le graphique de distribution par langue, Train vs Val)*

---

### Slide 6 — EDA : hétérogénéité de tokenisation
- Réponses akan/luganda beaucoup plus longues (moy. 233-245 tokens, max ~1000) vs anglais (32-121 tokens)
- Ratio tokens/mot : luganda 3,06 / amharique 3,01 / akan 2,24 / swahili 2,09 **vs** anglais 1,23-1,34
- → le tokenizer de Gemma fragmente fortement les langues à faibles ressources = tâche plus difficile

---

### Slide 7 — EDA : réponses dupliquées
- 39,4 % des réponses de Train sont dupliquées
- Explication : dataset de type FAQ, plusieurs questions proches → même réponse canonique
- Pas des lignes corrompues (IDs tous uniques), mais à surveiller (biais vers réponses génériques)

---

### Slide 8 — Pourquoi le fine-tuning plutôt que les baselines fournies
- Baselines du notebook de départ : retrieval TF-IDF, seq2seq mT5/NLLB
- Choix : fine-tuning LoRA/QLoRA d'un LLM instruction-tuned récent (Gemma E2B)
- Justification : meilleure fluidité potentielle qu'un simple retrieval ; faisable sur GPU accessible grâce à la quantification 4-bit ; représentatif des pratiques NLP actuelles

---

### Slide 9 — LoRA / QLoRA en bref
- LoRA : geler le modèle, entraîner seulement de petites matrices de rang réduit insérées dans certaines couches
- QLoRA : + charger le modèle de base en 4-bit (NF4) → mémoire divisée par ~4
- Config retenue : r=16, alpha=32, dropout=0.05, cible q/k/v/o_proj + gate/up/down_proj

---

### Slide 10 — Préparation du dataset (Phase 2)
- Format chat (rôles `user`/`model`), prompt incluant le nom de la langue cible
- Masquage de la perte par **longueur mesurée** (pas de pattern-matching sur les tokens spéciaux) → robuste au template `<|turn>...` propre à Gemma-4, découvert en cours de projet
- Split train/dev stratifié par langue (95/5)
- Rééquilibrage plafonné (×3 max) des langues minoritaires → 53 300 lignes, déséquilibre réduit de ~4,1× à ~1,4×
- `MAX_SEQ_LENGTH` déterminé empiriquement (p95 = 477 tokens) → 512

---

### Slide 11 — Environnement d'entraînement
- Google Colab : démarrage sur GPU gratuit (T4, 16 Go, fp16)
- Passage à Colab Pro (GPU L4, 24 Go, bf16 natif) pour accélérer
- Reprise automatique sur checkpoint (déconnexions de session) → entraînement réparti sur plusieurs sessions

---

### Slide 12 — Défis techniques rencontrés
- Modèle très récent → plusieurs incompatibilités logicielles à résoudre en cours de route :
  - OOM sur la préparation du modèle quantifié (embeddings trop grands en fp32)
  - Couche custom `Gemma4ClippableLinear` non reconnue par PEFT
  - Incohérences de schéma de données, gestion des checkpoints entre fp16/bf16
  - OOM pendant le calcul de la loss (grand vocabulaire) à batch trop élevé
- *(détail complet en annexe du rapport écrit)*

---

### Slide 13 — Entraînement final
- 3 époques, batch effectif 16, learning rate 2×10⁻⁴, bf16, séquences max 512 tokens
- 9 996 pas, loss finale = 0,668
- ~16,8h de calcul cumulées sur plusieurs sessions
- *(insérer la courbe de perte train/eval)*

---

### Slide 14 — Résultats globaux (ROUGE)
- ROUGE-1 : 0,129 (zero-shot) → 0,367 (fine-tuné) — ×2,8
- ROUGE-L : 0,079 (zero-shot) → 0,311 (fine-tuné) — ×3,9
- Amélioration sur **toutes** les langues, sans exception

---

### Slide 15 — Résultats par langue
- *(insérer le tableau ou le graphique comparatif zero-shot vs fine-tuné par langue)*
- Meilleurs scores fine-tunés : Eng_Uga (0,573), Eng_Ken (0,495)
- Plus faibles : Amh_Eth (0,156), Lug_Uga (0,186), Aka_Gha (0,266)
- Confirme l'hypothèse posée par l'EDA (langues les plus fragmentées = les plus difficiles)

---

### Slide 16 — Exemples qualitatifs
- Zero-shot : répond parfois dans la mauvaise langue (français/vietnamien sur une question akan/anglaise !)
- Zero-shot en luganda : sortie dégénérée ("kye kye kye..." en boucle)
- Fine-tuné : réponses cohérentes, dans la bonne langue, sur le sujet
- Limite observée : un artefact type "as of my last knowledge update..." (trace de connaissances générales du LLM)

---

### Slide 17 — Limites
- Écart persistant sur amharique/luganda malgré le rééquilibrage
- Aucune validation médicale experte des réponses générées
- Métrique `TargetLLM` (LLM-as-judge) non mise en œuvre (nécessite une clé API)
- Évaluation sur un échantillon de 500 lignes (sur 6 686 disponibles)

---

### Slide 18 — Pistes d'amélioration
- Fine-tuning séparé par langue pour les langues les plus faibles
- Évaluation complète sur tout `Val.csv`
- Mise en œuvre de la métrique LLM-as-judge
- Validation par des professionnels de santé avant tout usage réel

---

### Slide 19 — Conclusion
- Pipeline complet et documenté : EDA → préparation → fine-tuning → évaluation
- Gain net et mesuré du fine-tuning (×2,8 à ×3,9 selon la métrique)
- Principal enseignement : la couverture linguistique inégale du modèle pré-entraîné reste le facteur limitant majeur, anticipé dès l'EDA
- Projet représentatif des défis réels du fine-tuning de LLM récents sur infrastructure limitée

---

### Slide 20 — Merci / Questions
- Dépôt du projet : github.com/andilMc/gemmafro-e2b
- Questions ?
