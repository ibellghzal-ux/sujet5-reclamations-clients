# Sujet 5 : classification de réclamations clients

Séance 3 : projet construit sur le squelette « projet à compléter ». Il classe une réclamation (texte, en anglais)
selon le produit financier concerné, en 9 catégories.

**Données** : vraies réclamations de la base Kaggle « Consumer Complaint Database » (selener), échantillon de
20 000 nettoyé en 19 331 lignes (`data/reclamations.csv`). Reproduction : `scripts/echantillon_kaggle.py` (à lancer
dans un notebook Kaggle) puis `python scripts/preparer_donnees.py echantillon.csv`. Détails : `docs/etude-de-conception.md`.

**Résultat** (jeu de test de 3 867 réclamations, F1 macro) : baseline 0,054 -> régression logistique « balanced » 0,742.
Détails et limites : `docs/choix-modele.md`.

**Lancer** : voir les consignes ci-dessous. `python -m monprojet.train` prend environ 1 minute et `python -m pytest`
environ 2 minutes (les tests relancent l'entraînement). Pour revérifier le code sur MaintAI :
`PROJET_DATASET=maintai python -m pytest` (PowerShell : `$env:PROJET_DATASET="maintai"`).

---

# Consignes du projet à compléter

Vous écrivez vous-mêmes le chargement des données, le découpage entraînement / test, les modèles à comparer et leur évaluation. La structure du projet et les tests sont fournis : **les tests disent ce qui est attendu**, et vous avez fini quand ils passent tous.

## 1. Installer (une fois)

```
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows ; macOS/Linux : source .venv/bin/activate
pip install -r requirements.txt
python -m pytest                    # au départ : 7 failed, 1 passed (c'est normal)
```

## 2. Régler config.py

Votre fichier CSV dans `data/`, puis dans `monprojet/config.py` : `DATA_PATH`, `TARGET`, `TASK`, `MAIN_METRIC` (celle de votre section 5), `SPLIT` et `TIME_COLUMN` pour une série temporelle, `TEXT_COLUMN` pour du texte libre. Pas encore de données ? Gardez l'exemple `data/exemple_pannes.csv`.

## 3. Écrire le code, fonction par fonction

| Ordre | Fichier | Fonction | Tests qui passent ensuite |
|---|---|---|---|
| 1 | `monprojet/data.py` | `load_data()`, `split()` | 3 passed |
| 2 | `monprojet/models.py` | `candidates()` : la baseline d'abord, puis au moins trois modèles | 5 passed |
| 3 | `monprojet/train.py` | `evaluate()`, puis `main()` | 8 passed |

Chaque fonction contient un commentaire `TODO` qui dit ce qu'elle doit faire et quelles fonctions de pandas ou scikit-learn utiliser. `feature_columns()` et `build_preprocessing()` sont fournies.

Après chaque fonction : `python -m pytest`. Une fois tout écrit : `python -m monprojet.train` affiche le tableau de comparaison.

## 4. Choisir

Remplissez `docs/choix-modele.md` : les résultats sur le jeu de test, le modèle retenu, l'écart avec la baseline, le critère d'acceptation atteint ou non.

## Ce que produit l'entraînement

| Fichier | Contenu |
|---|---|
| `models/model.joblib` | Le meilleur modèle selon la métrique principale |
| `models/metrics.json` | `main_metric`, `best`, et les scores de tous les modèles |
