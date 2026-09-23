# TF-IDF / NLLB baseline — notebook de démarrage Zindi

Copie du notebook de démarrage fourni par les organisateurs du challenge Zindi.
Ce n'est pas notre code : nous l'avons exécuté tel quel pour disposer d'un point
de comparaison indépendant (voir section 2 du rapport, `rapport-web/`).

Placé ici (plutôt que dans `TF-IDF/` à la racine du projet, qui n'est pas un
dépôt Git) uniquement pour pouvoir l'ouvrir directement dans Colab via le badge
en haut du notebook, et le pousser vers ce dépôt.

La dernière cellule (ajoutée par nous, pas par Zindi) sauvegarde le checkpoint
NLLB fine-tuné + ses prédictions par ligne sur Drive — jusque-là jamais
persisté. Elle doit être exécutée juste après la cellule de fine-tuning
(`Seq2SeqTrainer`), dans la même session Colab.
