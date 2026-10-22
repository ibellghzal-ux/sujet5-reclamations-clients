# Choix du modèle

Date : [jj/mm/aaaa] · Groupe : [noms]

## Contexte
Classer automatiquement une réclamation client (texte libre) selon le produit financier concerné (9 catégories),
pour l'aiguiller vers le bon service. Métrique principale : **F1 macro** (section 5), car toutes les catégories
comptent autant, y compris les rares.

## Données et découpage
`data/reclamations.csv` : 19 331 vraies réclamations (Kaggle, Consumer Complaint Database), **en anglais**, nettoyées
(catégories regroupées, doublons retirés, voir section 4). Seule la colonne `texte` est utilisée (TF-IDF, 5 000 mots
au maximum). Découpage aléatoire stratifié, graine 42 : 15 464 réclamations d'entraînement, **3 867 de test**, le
même jeu pour tous les modèles.

## Résultats sur le jeu de test

| Modèle | F1 macro (principale) | Autres métriques (accuracy / rappel macro / précision macro) |
|---|---|---|
| Baseline (classe majoritaire) | 0,054 | 0,324 / 0,111 / 0,036 |
| Régression logistique (`class_weight="balanced"`) | **0,742** | 0,810 / 0,786 / 0,712 |
| SVM linéaire (`class_weight="balanced"`) | 0,717 | 0,810 / 0,719 / 0,720 |
| Régression logistique | 0,684 | 0,814 / 0,647 / 0,804 |
| Forêt aléatoire (200 arbres) | 0,531 | 0,777 / 0,520 / 0,667 |

## Décision
**Régression logistique avec `class_weight="balanced"`** retenue (sauvegardée dans `models/model.joblib`) : meilleur
F1 macro (0,742) et meilleur rappel macro (0,786). Elle bat la baseline de **+0,69** de F1 macro (0,742 contre 0,054).
Le critère d'acceptation de la section 5 est **atteint** : F1 macro 0,742 ≥ 0,70, rappel macro 0,786 ≥ 0,70, écart
avec la baseline 0,69 ≥ 0,50. La pondération « balanced » fait gagner 6 points de F1 macro sur la même régression
sans pondération (0,684), surtout grâce au rappel des petites classes (0,647 -> 0,786), au prix d'une précision plus
faible (0,804 -> 0,712). La forêt aléatoire est nettement moins bonne (0,531) ; nous n'avons pas cherché à expliquer pourquoi (hypothèse : un texte représenté par 5 000 colonnes creuses lui convient mal).

## Limites
- **Catégories faibles** : « Payday or personal loan » (F1 0,43 ; rappel 0,49 ; 63 exemples de test) et « Vehicle
  loan or lease » (F1 0,59 ; précision 0,46 ; 55 exemples de test). Avec si peu d'exemples, ces scores sont instables.
- **Confusions principales** : 109 réclamations « Credit reporting » classées « Debt collection » (et 87 dans
  l'autre sens). Ces deux catégories se recoupent souvent dans la réalité ; une partie des « erreurs » vient
  de l'étiquette, pas du modèle.
- **Le critère d'acceptation a été fixé après un premier lancement** sur les vraies données (voir section 5) : il ne
  prouve pas une performance indépendante.
- **Textes en anglais, domaine financier américain** : aucune garantie sur du français ou d'autres types de réclamations.
- Un seul découpage (pas de validation croisée) et aucun réglage d'hyperparamètres ; l'écart entre la régression
  logistique « balanced » et le SVM (0,025 de F1) est modeste et pourrait changer avec une autre graine.
- Catégories regroupées et retirées par nos soins (section 4) : les scores ne sont pas comparables à ceux
  d'autres travaux sur cette base.
- Les textes contiennent des « XXXX » (données masquées par la source) qui n'ont pas été traités.
