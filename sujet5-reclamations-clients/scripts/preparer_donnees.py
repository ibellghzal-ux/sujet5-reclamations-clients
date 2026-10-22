"""Nettoie l'échantillon Kaggle et écrit data/reclamations.csv.

Chaîne complète (reproductible) :
1. scripts/echantillon_kaggle.py, à lancer dans un notebook Kaggle avec le jeu
   « Consumer Complaint Database » (selener) attaché : échantillon de 20 000 réclamations
   (colonnes texte, categorie), graine 42.
2. ce script : python scripts/preparer_donnees.py chemin/vers/echantillon.csv
   -> regroupe les catégories qui existent sous deux noms (ancien et nouveau nom de la base),
      retire les catégories ambiguës ou trop rares, les doublons et les textes trop courts.
"""
import sys
from pathlib import Path

import pandas as pd

SORTIE = Path(__file__).resolve().parent.parent / "data" / "reclamations.csv"

# Anciens et nouveaux noms de la base d'origine -> une seule catégorie
REGROUPEMENT = {
    "Credit reporting": "Credit reporting",
    "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting",
    "Credit card": "Credit or prepaid card",
    "Prepaid card": "Credit or prepaid card",
    "Credit card or prepaid card": "Credit or prepaid card",
    "Bank account or service": "Bank account",
    "Checking or savings account": "Bank account",
    "Money transfers": "Money transfer",
    "Money transfer, virtual currency, or money service": "Money transfer",
    "Payday loan": "Payday or personal loan",
    "Payday loan, title loan, or personal loan": "Payday or personal loan",
    "Debt collection": "Debt collection",
    "Mortgage": "Mortgage",
    "Student loan": "Student loan",
    "Vehicle loan or lease": "Vehicle loan or lease",
    # Volontairement absentes : « Consumer Loan » (ancienne catégorie qui couvre à la fois
    # les prêts auto et les prêts personnels : ambiguë) et « Other financial service » (14 lignes).
}
MIN_MOTS = 5


def main(chemin):
    df = pd.read_csv(chemin)
    n0 = len(df)
    df = df[df["categorie"].isin(REGROUPEMENT)].copy()
    n1 = len(df)
    df["categorie"] = df["categorie"].map(REGROUPEMENT)
    df = df.drop_duplicates(subset="texte")
    n2 = len(df)
    df = df[df["texte"].str.split().str.len() >= MIN_MOTS]
    n3 = len(df)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    SORTIE.parent.mkdir(exist_ok=True)
    df.to_csv(SORTIE, index=False)
    print(f"lignes : {n0} -> {n1} (catégories ambiguës/rares retirées) -> {n2} (doublons) -> {n3} (textes courts)")
    print(df["categorie"].value_counts().to_string())
    print("Écrit :", SORTIE)


if __name__ == "__main__":
    main(sys.argv[1])
