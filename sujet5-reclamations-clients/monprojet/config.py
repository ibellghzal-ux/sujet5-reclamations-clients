"""Configuration du projet : c'est le seul fichier à adapter à votre sujet."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Sujet 5 (réclamations clients) par défaut. Pour revérifier le code sur MaintAI (Lab 2.1, partie 1) :
#   Windows PowerShell : $env:PROJET_DATASET="maintai"   |   macOS/Linux : export PROJET_DATASET=maintai
MAINTAI = os.environ.get("PROJET_DATASET", "reclamations") == "maintai"

# --- Vos données -------------------------------------------------------------
DATA_PATH = ROOT / "data" / ("exemple_pannes.csv" if MAINTAI else "reclamations.csv")
TARGET = "failure" if MAINTAI else "categorie"      # la colonne à prédire
FEATURES = None              # None = toutes les colonnes sauf la cible ; sinon une liste de noms
TEXT_COLUMN = None if MAINTAI else "texte"   # le nom d'une colonne de texte libre (réclamations…), sinon None

# --- Type de problème --------------------------------------------------------
TASK = "classification"      # "classification" ou "regression"
MAIN_METRIC = "recall" if MAINTAI else "f1"   # F1 macro pour le sujet 5 ; classification : "accuracy", "recall", "precision", "f1"
                             # regression : "mae", "rmse", "r2"

# --- Découpage entraînement / test -------------------------------------------
SPLIT = "random"             # "random" ; ou "chronologique" pour une série temporelle
TIME_COLUMN = None           # colonne de date, obligatoire si SPLIT = "chronologique"
TEST_SIZE = 0.2              # part des données gardée pour le test
SEED = 42                    # graine : mêmes résultats à chaque lancement

MODELS_DIR = ROOT / "models"
