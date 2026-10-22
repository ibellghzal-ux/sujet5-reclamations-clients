# Étude de conception — Sujet 5 : classification de réclamations clients

Groupe : [noms] · Version : [brouillon / finale] · Date : [jj/mm/aaaa]

## 1. Contexte et utilisateurs
- Utilisateur principal (un rôle précis) :

## 2. Problème et décision
- Décision que l'utilisateur prend avec le système :
- Fréquence :
- Coût d'une erreur, et pour qui :

## 3. Approche par sous-problème
| Sous-problème | Approche (règles / ML / DL / LLM / RAG) | Pourquoi pas une approche plus simple |
|---|---|---|

## 4. Données
| Jeu de données | Source et licence | Volume | Données personnelles ? |
|---|---|---|---|
| `data/reclamations.csv` : colonnes `texte` (texte libre de la réclamation, **en anglais**) et `categorie` (type de produit financier concerné, 9 catégories) | Kaggle, « Consumer Complaint Database » (utilisateur `selener`) : plaintes de consommateurs financiers américains (Bureau de protection financière des consommateurs). https://www.kaggle.com/datasets/selener/consumer-complaint-database . **Licence : à recopier depuis l'onglet License de la page Kaggle (non vérifiée).** | 19 331 réclamations après nettoyage (extraites d'un échantillon aléatoire de 20 000, graine 42) | Pas de nom direct (les noms, dates et numéros sont masqués dans le texte par « XXXX »), mais le texte libre peut contenir des détails personnels résiduels : ne pas republier les textes tels quels |

**Répartition des 9 catégories :** Credit reporting 6 261 · Debt collection 4 570 · Mortgage 2 800 · Credit or
prepaid card 2 142 · Bank account 1 498 · Student loan 1 118 · Money transfer 351 · Payday or personal loan 315 ·
Vehicle loan or lease 276.

**Préparation (reproductible, `scripts/echantillon_kaggle.py` puis `scripts/preparer_donnees.py`) :**
- Échantillon de 20 000 lignes ayant un texte de réclamation (le fichier d'origine est trop volumineux pour GitHub).
- Plusieurs catégories existent sous un ancien et un nouveau nom dans la base : elles sont regroupées (par exemple
  « Credit reporting » et « Credit reporting, credit repair services, or other personal consumer reports »).
- Retirées : « Consumer Loan » (ancienne catégorie ambiguë, qui mélange prêts auto et prêts personnels) et
  « Other financial service » (14 lignes) ; puis 142 textes en doublon et 16 textes de moins de 5 mots.

**Limites des données :**
- Textes en anglais, domaine financier américain : le modèle n'est pas testé sur du français ni sur d'autres
  types de réclamations (livraison, produit défectueux...).
- La catégorie est celle choisie par le client ou le service au dépôt : elle peut être discutable (par exemple
  « Credit reporting » et « Debt collection » se recoupent souvent).
- Classes très déséquilibrées : de 276 à 6 261 exemples.

## 5. Métriques, baseline et critères d'acceptation
- Baseline et son score : toujours prédire la classe majoritaire (`Credit reporting`). Sur le jeu de test :
  accuracy 0,324, rappel macro 0,111, **F1 macro 0,054**. L'accuracy de la baseline (0,324) est faible parce que
  les classes sont nombreuses, mais elle ne dit rien d'une classe rare : le rappel de la baseline est nul sur 8 des 9 catégories.
- Métrique principale, et pourquoi : le **F1 macro** (moyenne des F1 de chaque catégorie, toutes comptent pareil).
  Une réclamation mal classée est envoyée au mauvais service ; les petites catégories (276 à 351 exemples)
  doivent être aussi bien traitées que les grosses, ce que l'accuracy ne vérifie pas.
- Critères d'acceptation (métrique, nombre, jeu de données, condition) : **F1 macro ≥ 0,70 et rappel macro ≥ 0,70
  sur le jeu de test** (3 867 réclamations, 20 %, tenu à l'écart, découpage aléatoire stratifié, graine 42), et
  F1 macro au moins **0,50 au-dessus de la baseline**.
- Remarque d'honnêteté : une première version de ces seuils (F1 ≥ 0,85) avait été écrite pour un jeu fictif. Elle
  a été revue en passant aux vraies données, **après un premier lancement** : les seuils actuels ne sont donc pas
  totalement indépendants des résultats observés.

Réglages correspondants (`monprojet/config.py`) : `DATA_PATH` = `data/reclamations.csv` ; `TARGET` = `categorie` ;
`TASK` = `classification` ; `MAIN_METRIC` = `f1` ; `SPLIT` = `random` (pas de `TIME_COLUMN`) ; `TEXT_COLUMN` = `texte`.

| Vérification par le groupe voisin | Oui | Non |
|---|---|---|
| Une métrique est nommée | ☐ | ☐ |
| Un nombre à atteindre | ☐ | ☐ |
| Le jeu de données sur lequel on mesure | ☐ | ☐ |
| La condition (seuil, période, comparaison à la baseline) | ☐ | ☐ |

## 6. Architecture prévue
## 7. Choix technologiques
## 8. Risques, éthique, confidentialité
## 9. Limites et hors périmètre
