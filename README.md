# Multilingual Health QA — Fine-tuning Gemma E2B

Projet de classe (Master 2 BI) pour le défi Zindi **Multilingual Health Question Answering in Low-Resource African Languages Challenge**, organisé en collaboration avec HASH et Makerere University.

## Structure du dépôt

```
gemmafro-e2b/
├── data/         # Jeux de données du challenge (CSV)
├── docs/         # Documentation du projet
└── notebooks/    # Notebooks (starter + travail de fine-tuning)
```

## Documentation du projet

- [docs/PROJET.md](docs/PROJET.md) — objet du challenge, contexte, données, métriques d'évaluation.
- [docs/FICHIERS.md](docs/FICHIERS.md) — détail de chaque fichier du dépôt et de sa nécessité.
- [docs/WORKFLOW.md](docs/WORKFLOW.md) — workflow suivi pour le fine-tuning de Gemma E2B sur Google Colab (GPU T4 gratuit), phase par phase.

## Données (`data/`)

| Fichier | Rôle |
|---|---|
| `Train.csv` | 29 815 paires question/réponse (entraînement) |
| `Val.csv` | 6 686 paires question/réponse (validation) |
| `Test.csv` | 2 618 questions (test, réponses à prédire) |
| `SampleSubmission.csv` | Format de soumission attendu par Zindi |

5 langues couvertes : anglais, akan, luganda, swahili, amharique (voir [docs/PROJET.md](docs/PROJET.md) pour le détail par pays).

## Méthode

Fine-tuning **LoRA/QLoRA** d'un modèle **Gemma E2B** (checkpoint instruction-tuned) sur Google Colab, détaillé pas à pas dans [docs/WORKFLOW.md](docs/WORKFLOW.md).

`notebooks/multilingual_health_qa_starter_notebook.ipynb` est le notebook de départ fourni par les organisateurs (baselines TF-IDF et mT5) ; il sert de point de départ mais n'est pas la méthode retenue pour ce projet.
