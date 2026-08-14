# Objet du projet

## Titre
**Multilingual Health Question Answering in Low-Resource African Languages Challenge**
(Défi Zindi, organisé en collaboration avec HASH — *Hub for Artificial Intelligence in Maternal, Sexual and Reproductive Health* — et Makerere University, sous l'égide de l'ITU)

## Contexte
HASH est un consortium ougandais (Infectious Diseases Institute, Sunbird AI, Makerere University COCIS) qui promeut une IA responsable au service de la santé maternelle, sexuelle et reproductive (MSRH) en Afrique subsaharienne. Le constat de départ :
- Les taux de maternité précoce et de mortalité maternelle restent très élevés en Afrique subsaharienne.
- Plus d'1 million d'IST sont contractées chaque jour dans le monde, majoritairement en Afrique.
- L'accès à une information de santé fiable est souvent limité par la **barrière de la langue**, la majorité des ressources existantes n'étant disponibles qu'en anglais ou en français.

Ce défi Zindi a été monté pour combler ce manque en exploitant la connectivité mobile croissante du continent (des centaines de millions d'utilisateurs internet/mobile attendus d'ici 2030).

## Objectif du challenge
Développer un modèle **multilingue de question-réponse** capable de répondre à des questions de santé (maternelle, sexuelle et reproductive) dans **la même langue** que la question posée, parmi cinq langues :

| Code | Langue    | Pays associé(s) dans les données |
|------|-----------|-----------------------------------|
| Eng  | Anglais   | Ghana, Ouganda, Kenya, Éthiopie   |
| Aka  | Akan      | Ghana                              |
| Lug  | Luganda   | Ouganda                            |
| Swa  | Swahili   | Kenya                              |
| Amh  | Amharique | Éthiopie                           |

Il s'agit donc d'une tâche de **génération de texte séquence-à-séquence (seq2seq)** : en entrée une question de santé, en sortie une réponse fluide et médicalement correcte, dans la langue de la question.

## Utilisateurs finaux visés
- **Adolescents** cherchant une information confidentielle sur la santé sexuelle et reproductive dans leur langue.
- **Femmes enceintes et jeunes mères** ayant besoin d'informations fiables sur le suivi prénatal, l'accouchement et le post-partum.
- **Agents de santé communautaires et ONG** ayant besoin d'outils multilingues pour leurs interactions avec les patients.

## Données fournies
| Fichier | Rôle | Nb de lignes |
|---|---|---|
| `Train.csv` | Paires question/réponse pour l'entraînement | 29 815 |
| `Val.csv` | Paires question/réponse pour la validation | 6 686 |
| `Test.csv` | Questions seules (réponses à prédire) | 2 618 |

Colonnes :
- `Train.csv` / `Val.csv` : `ID`, `input` (question), `output` (réponse de référence), `subset` (code langue_pays, ex. `Amh_Eth`)
- `Test.csv` : `ID`, `input`, `subset`

## Métriques d'évaluation
La soumission est évaluée sur trois métriques (moyennées), calculées par Zindi à partir du fichier de soumission :
- **TargetR1F1** — score ROUGE-1 F1
- **TargetRLF1** — score ROUGE-L F1
- **TargetLLM** — score "LLM-as-a-Judge" (évaluation qualitative par un LLM juge)

Le fichier `SampleSubmission.csv` montre le format attendu : une colonne `ID` et une colonne par métrique, **contenant la même réponse générée** répétée dans les trois colonnes cibles.

## Pistes de modélisation (suggérées dans le notebook de départ)
1. **Baseline 1 — Retrieval TF-IDF** : recherche du plus proche voisin (n-grammes de caractères, robuste aux différentes écritures) parmi les questions d'entraînement, et renvoi de la réponse associée. Rapide mais qualité limitée.
2. **Baseline 2 — LLM multilingue pré-entraîné** (`google/mt5-small/base`, `facebook/nllb-200-distilled-600M`) utilisé en zero-shot puis **fine-tuné** sur `Train.csv` pour améliorer la qualité des réponses générées.

## Livrable attendu
Un fichier de soumission au format `SampleSubmission.csv` (`ID`, `TargetRLF1`, `TargetR1F1`, `TargetLLM`) généré à partir des questions de `Test.csv`, dans le cadre du projet de classe (Master 2 BI).
