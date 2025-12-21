import pandas as pd
import os
import sys

from evidently.test_suite import TestSuite
from evidently.tests import TestColumnDrift
from evidently import ColumnMapping

# ======================================================
# CONFIGURATION
# ======================================================

REF_DATA_PATH = "data/version1.xlsx"
CURRENT_DATA_PATH = "data/version2.xlsx"
OUTPUT_REPORT_PATH = "evidently_reports/target_drift_report.html"

TARGET_COLUMN = "target"

os.makedirs(os.path.dirname(OUTPUT_REPORT_PATH), exist_ok=True)

# ======================================================
# 1. CHARGEMENT DES DONNÉES
# ======================================================

try:
    ref_data = pd.read_excel(REF_DATA_PATH)
    current_data = pd.read_excel(CURRENT_DATA_PATH)
except Exception as e:
    print(f"❌ Erreur chargement données : {e}")
    sys.exit(1)

# ======================================================
# 2. COLUMN MAPPING (TARGET SEULEMENT)
# ======================================================

column_mapping = ColumnMapping(
    target=TARGET_COLUMN,
)

# ======================================================
# 3. TEST DE DRIFT SUR LA DISTRIBUTION DU LABEL
# ======================================================

test_suite = TestSuite(
    tests=[
        TestColumnDrift(column_name=TARGET_COLUMN)
    ]
)

print("🚀 Détection du drift sur la distribution du target...")

test_suite.run(
    reference_data=ref_data[[TARGET_COLUMN]],
    current_data=current_data[[TARGET_COLUMN]],
    column_mapping=column_mapping,
)

# ======================================================
# 4. RÉSULTAT
# ======================================================

is_passed = test_suite.as_dict()["summary"]["all_passed"]

if is_passed:
    status = "NO_TARGET_DRIFT"
    print("✅ La distribution du target est stable.")
else:
    status = "TARGET_DRIFT_DETECTED"
    print("⚠️ Drift détecté dans la distribution du target.")

# GitHub Actions output
if "GITHUB_OUTPUT" in os.environ:
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write(f"drift_status={status}\n")

# ======================================================
# 5. RAPPORT HTML
# ======================================================

test_suite.save_html(OUTPUT_REPORT_PATH)
print(f"📄 Rapport sauvegardé : {OUTPUT_REPORT_PATH}")

sys.exit(0)
