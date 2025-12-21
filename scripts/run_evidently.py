import pandas as pd
import os
import sys
from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset
from evidently.tests import TestShareOfDriftedColumns, TestColumnDrift

# --- Configuration ---
# Chemins des fichiers de données
REF_DATA_PATH = 'data/version1.xlsx'
CURRENT_DATA_PATH = 'data/version2.xlsx'
# Chemin du rapport HTML généré par Evidently
OUTPUT_REPORT_PATH = 'evidently_reports/data_drift_report.html'
# Fichier indicateur pour le statut de la dérive (Non utilisé, on utilise $GITHUB_OUTPUT)
# STATUS_FILE_PATH = 'data_drift_status.txt'

# Colonne cible (label) - Assurez-vous que ce nom est correct
TARGET_COLUMN = 'label' 

# Seuil de colonnes en dérive (par exemple, si plus de 50% des colonnes sont en dérive, on considère qu'il y a un problème)
MAX_DRIFTED_COLUMNS_SHARE = 0.5

# --- Préparation ---
os.makedirs(os.path.dirname(OUTPUT_REPORT_PATH), exist_ok=True)

# --- 1. Chargement des données ---
try:
    # Utilisation de l'index 0 pour lire la première feuille par défaut
    ref_data = pd.read_excel(REF_DATA_PATH)
    current_data = pd.read_excel(CURRENT_DATA_PATH)
except FileNotFoundError as e:
    print(f"Erreur: Fichier de données non trouvé: {e}")
    # Sortie d'erreur pour GitHub Actions
    if 'GITHUB_OUTPUT' in os.environ:
        with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
            f.write("drift_status=ERROR\n")
    sys.exit(1)
except Exception as e:
    print(f"Erreur lors du chargement des données: {e}")
    # Sortie d'erreur pour GitHub Actions
    if 'GITHUB_OUTPUT' in os.environ:
        with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
            f.write("drift_status=ERROR\n")
    sys.exit(1)

# --- 2. Exécution de la suite de tests Evidently ---
data_drift_suite = TestSuite(tests=[
    # Test général de dérive des données
    DataDriftTestPreset(),
    # Test spécifique sur la colonne cible (label)
    TestColumnDrift(column_name=TARGET_COLUMN, lt=0.05), # Teste si la dérive est inférieure à 5% (ajuster si besoin)
    # Test sur la proportion de colonnes en dérive
    TestShareOfDriftedColumns(lt=MAX_DRIFTED_COLUMNS_SHARE)
])

print("🚀 Exécution de la suite de tests Evidently AI pour la détection de dérive...")
data_drift_suite.run(reference_data=ref_data, current_data=current_data, column_mapping=None)

# --- 3. Analyse des résultats et écriture du statut ---
is_passed = data_drift_suite.as_dict()['summary']['all_passed']

if is_passed:
    status = "NO_DRIFT"
    print("✅ SUCCÈS: Aucune dérive de données significative détectée.")
else:
    status = "DRIFT_DETECTED"
    print("⚠️  ALERTE: Dérive de données détectée. Un ré-entraînement avec correction est nécessaire.")

# Écriture du statut dans la variable de sortie GitHub Actions
if 'GITHUB_OUTPUT' in os.environ:
    with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
        f.write(f"drift_status={status}\n")

# Sauvegarde du rapport HTML pour l'analyse
data_drift_suite.save_html(OUTPUT_REPORT_PATH)
print(f"Rapport HTML de dérive sauvegardé: {OUTPUT_REPORT_PATH}")

# Sortie du script
sys.exit(0)
