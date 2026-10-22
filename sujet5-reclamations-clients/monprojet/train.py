"""Entraîne chaque modèle candidat, l'évalue sur le jeu de test et les compare.

Usage, depuis la racine du projet : python -m monprojet.train
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from monprojet import config
from monprojet.data import load_data, split
from monprojet.models import candidates

HIGHER_IS_BETTER = {"accuracy": True, "recall": True, "precision": True, "f1": True,
                    "mae": False, "rmse": False, "r2": True}


def build_preprocessing(X: pd.DataFrame) -> ColumnTransformer:
    """Prépare les colonnes : nombres mis à l'échelle, catégories encodées, texte vectorisé. (Fournie.)"""
    text = [config.TEXT_COLUMN] if config.TEXT_COLUMN else []
    numeric = [c for c in X.columns if c not in text and pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in X.columns if c not in text and c not in numeric]
    steps = []
    if numeric:
        steps.append(("nombres", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric))
    if categorical:
        steps.append(("categories", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                                  OneHotEncoder(handle_unknown="ignore")), categorical))
    if text:
        # Vocabulaire limité (5 000 mots) : sinon la matrice de texte devient énorme sur 19 000 réclamations
        steps.append(("texte", TfidfVectorizer(max_features=5000, min_df=3, sublinear_tf=True),
                      config.TEXT_COLUMN))
    # sparse_threshold par défaut : le texte reste en matrice creuse (sinon des Go de mémoire)
    return ColumnTransformer(steps)


def evaluate(model, X_test, y_test) -> dict:
    """Prédit sur le jeu de test et renvoie un dictionnaire {nom de la métrique: valeur}.

    Attendu :
    - classification : "accuracy", "recall", "precision", "f1"
      (average="binary" s'il y a deux classes, "macro" sinon ; zero_division=0) ;
    - régression : "mae", "rmse", "r2" ;
    - des float, arrondis à 3 décimales.

    Pistes : model.predict(X_test) ; les fonctions importées en haut du fichier ; np.sqrt.
    """
    y_pred = model.predict(X_test)
    if config.TASK == "classification":
        average = "binary" if y_test.nunique() == 2 else "macro"
        scores = {
            "accuracy": accuracy_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred, average=average, zero_division=0),
            "precision": precision_score(y_test, y_pred, average=average, zero_division=0),
            "f1": f1_score(y_test, y_pred, average=average, zero_division=0),
        }
    else:
        scores = {
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred),
        }
    return {nom: round(float(valeur), 3) for nom, valeur in scores.items()}


def main() -> None:
    """Entraîne chaque candidat, l'évalue, compare, et sauvegarde.

    Plan :
    1. charger les données (load_data) et les découper (split) ;
    2. pour chaque (nom, estimateur) de candidates() :
       model = Pipeline([("preparation", build_preprocessing(X_train)), ("modele", estimateur)]),
       l'entraîner sur X_train, y_train, puis l'évaluer avec evaluate() ;
    3. afficher un tableau de comparaison (pd.DataFrame(resultats).T), trié selon config.MAIN_METRIC ;
    4. trouver le meilleur modèle selon config.MAIN_METRIC (HIGHER_IS_BETTER dit dans quel sens) ;
    5. créer config.MODELS_DIR, y sauvegarder le meilleur modèle (joblib.dump) dans model.joblib,
       et écrire metrics.json : {"main_metric": ..., "best": nom du meilleur, "results": {nom: scores}}.
    """
    # 1. Données : le jeu de test ne sert qu'à mesurer
    df = load_data()
    X_train, X_test, y_train, y_test = split(df)

    # 2. Entraînement et évaluation (même jeu de test, même graine pour tous)
    resultats, modeles = {}, {}
    for nom, estimateur in candidates().items():
        model = Pipeline([
            ("preparation", build_preprocessing(X_train)),
            ("modele", estimateur),
        ])
        model.fit(X_train, y_train)
        resultats[nom] = evaluate(model, X_test, y_test)
        modeles[nom] = model

    # 3. Tableau de comparaison, trié selon la métrique principale
    metrique = config.MAIN_METRIC
    sens = HIGHER_IS_BETTER[metrique]
    tableau = pd.DataFrame(resultats).T.sort_values(metrique, ascending=not sens)
    print(f"\nJeu : {config.DATA_PATH.name} | entraînement : {len(X_train)} | test : {len(X_test)}")
    print(f"Métrique principale : {metrique}\n")
    print(tableau.to_string())

    # 4. Meilleur modèle (la baseline n'est pas candidate au titre)
    sans_baseline = tableau.drop(index=next(iter(resultats)))
    meilleur = sans_baseline[metrique].idxmax() if sens else sans_baseline[metrique].idxmin()
    print(f"\nMeilleur modèle : {meilleur}")

    # 5. Sauvegarde
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(modeles[meilleur], config.MODELS_DIR / "model.joblib")
    sortie = {"main_metric": metrique, "best": meilleur, "results": resultats}
    (config.MODELS_DIR / "metrics.json").write_text(
        json.dumps(sortie, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
