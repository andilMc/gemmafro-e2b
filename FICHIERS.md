# Contenu du dossier

Voir [PROJET.md](PROJET.md) pour le contexte général du challenge.

## Fichiers nécessaires au projet

### `Train.csv` — **nécessaire**
Jeu d'entraînement : 29 815 paires question/réponse (colonnes `ID`, `input`, `output`, `subset`) réparties sur les 5 langues. C'est la seule source de vérité terrain à utiliser pour entraîner ou fine-tuner un modèle.

### `Val.csv` — **nécessaire**
Jeu de validation : 6 686 paires question/réponse, même format que `Train.csv`. Sert à évaluer le modèle (ROUGE-1, ROUGE-L, éventuellement un juge LLM) pendant le développement, avant de générer les prédictions finales sur `Test.csv`.

### `Test.csv` — **nécessaire**
Jeu de test : 2 618 questions seules (`ID`, `input`, `subset`, pas de `output`). C'est sur ce fichier qu'il faut générer les réponses à soumettre.

### `SampleSubmission.csv` — **nécessaire**
Modèle du format de soumission attendu par Zindi : colonnes `ID`, `TargetRLF1`, `TargetR1F1`, `TargetLLM`. Utile pour vérifier que le fichier produit a la bonne structure avant de le générer/soumettre. Les valeurs de démonstration ("Wuna dey craze, eweeh") sont des placeholders sans rapport avec le contenu réel à produire.

### `multilingual_health_qa_starter_notebook.ipynb` — **utile mais optionnel**
Notebook de démarrage fourni par les organisateurs. Il n'est pas strictement indispensable (vous pouvez écrire votre propre code), mais il fait gagner du temps car il contient déjà :
- le chargement et nettoyage des données,
- une fonction d'évaluation ROUGE-1/ROUGE-L,
- une baseline TF-IDF (retrieval),
- une baseline LLM multilingue (mT5 / NLLB) en zero-shot,
- un pipeline de fine-tuning du LLM sur `Train.csv`,
- la génération d'un fichier de soumission au bon format.

À conserver comme point de départ ou source d'inspiration pour votre propre pipeline (BI/M2 → vous pouvez notamment l'enrichir avec vos propres analyses exploratoires, comparaisons de modèles, etc.).

## Fichiers non nécessaires au traitement du projet (contexte informatif uniquement)

### `Zindi_challenge_Info_webinar_presentation.pdf` — **non nécessaire pour coder**
Support de présentation du webinar de lancement (Claire Babirye, Makerere University). Contient uniquement le contexte, l'objectif et la description du dataset — informations déjà résumées dans `PROJET.md`. Utile pour la partie "contexte/motivation" d'un rapport, mais sans aucune donnée ou instruction technique supplémentaire.

### `HASH_Presentation_to_Zindi.pdf` — **non nécessaire pour coder**
Présentation institutionnelle du consortium HASH (Dr Elizabeth Oseku) : mission, partenaires, chiffres sur la santé maternelle/sexuelle en Afrique, projets déjà financés par HASH, appel à rejoindre le réseau HASH. Purement contextuel/institutionnel, à citer éventuellement dans l'introduction d'un rapport mais sans impact sur l'implémentation.

### `manifest-39b8ca2d42ae9885e3ffdbbf7b4fdfb620260521-4309-vfdx2v.json` — **non nécessaire, supprimable**
Fichier généré automatiquement par la plateforme (Zindi) listant les fichiers censés se trouver dans l'archive téléchargée, avec leur taille et description. Le fichier lui-même l'indique : *"It can be safely deleted"*. Aucune utilité pour l'analyse ou la modélisation.
