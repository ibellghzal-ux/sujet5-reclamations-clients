"""À exécuter DANS UN NOTEBOOK KAGGLE (jeu « Consumer Complaint Database » de selener attaché).

Produit /kaggle/working/reclamations.csv : 20 000 réclamations (colonnes texte, categorie), graine 42.
Le fichier brut est trop volumineux pour GitHub : seul le fichier nettoyé par
scripts/preparer_donnees.py est versionné dans data/.
"""
import glob

import pandas as pd

chemin = glob.glob("/kaggle/input/**/*.csv", recursive=True)[0]
print("Fichier :", chemin)
df = pd.read_csv(chemin, low_memory=False)
print("Colonnes :", list(df.columns))


def trouve(mot):
    return next(c for c in df.columns if mot in c.lower())


texte, cible = trouve("narrative"), trouve("product")
df = df[[texte, cible]].dropna()
df = df.rename(columns={texte: "texte", cible: "categorie"})
df = df[df["categorie"].map(df["categorie"].value_counts()) >= 200]
df = df.sample(min(20000, len(df)), random_state=42)
df.to_csv("/kaggle/working/reclamations.csv", index=False)
print(df.shape)
print(df["categorie"].value_counts())
