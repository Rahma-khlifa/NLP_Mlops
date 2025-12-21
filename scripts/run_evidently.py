import pandas as pd
import os
import sys

from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset
from evidently.tests import TestColumnDrift, TestShareOfDriftedColumns
from evidently import ColumnMapping

# ======================================================
# CONFIGURATION
# ======================================================

REF_DATA_PATH = "data/version1.xlsx"
CURRENT_DATA_PATH = "data/version2.xlsx"
OUTPUT_REPORT_PATH = "evidently_reports/data_drift_report.html"

TEXT_COLUMN = "comments"
TARGET_COLUMN = "target"

MAX_DRIFTED_COLUMNS_SHARE = 0.5

os.makedirs(os.path.dirname(OUTPUT_REPORT_PATH), exist_ok=True)

# ======================================================
# 1. CHARGEMENT DES DONNÉES
# ======================================================

try:
    ref_data = pd.read_excel(REF_DATA_PATH)
    current_data = pd.read_excel(CURRENT_DATA_PATH)
except Exception as e:
    print(f"❌ Erreur chargement données : {e}")
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write("drift_status=ERROR\n")
    sys.exit(1)

# ======================================================
# 2. EXTRACTION DES FEATURES NLP (SANS TEXTE BRUT)
# ======================================================

def extract_nlp_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)

    df["text_length"] = df[TEXT_COLUMN].str.len()
    df["word_count"] = df[TEXT_COLUMN].str.split().str.len()
    df["avg_word_length"] = (
        df[TEXT_COLUMN].str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.split()
        .apply(lambda x: sum(len(w) for w in x) / len(x) if len(x) > 0 else 0)
    )

    df["uppercase_ratio"] = df[TEXT_COLUMN].apply(
        lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
    )

    return df[
        [
            "text_length",
            "word_count",
            "avg_word_length",
            "uppercase_ratio",
            TARGET_COLUMN,
        ]
    ]


ref_features = extract_nlp_features(ref_data)
current_features = extract_nlp_features(current_data)

# ======================================================
# 3. COLUMN MAPPING (OBLIGATOIRE POUR NLP)
# ======================================================

column_mapping = ColumnMapping(
    target=TARGET_COLUMN,
    numerical_features=[
        "text_length",
        "word_count",
        "avg_word_length",
        "uppercase_ratio",
    ],
)

# ======================================================
# 4. SUITE DE TESTS EVIDENTLY
# ======================================================

data_drift_suite = TestSuite(
    tests=[
        DataDriftTestPreset(),
        TestColumnDrift(column_name=TARGET_COLUMN),
        TestShareOfDriftedColumns(lt=MAX_DRIFTED_COLUMNS_SHARE),
    ]
)

print("🚀 Exécution de la détection de data drift NLP...")

data_drift_suite.run(
    reference_data=ref_features,
    current_data=current_features,
    column_mapping=column_mapping,
)

# ======================================================
# 5. ANALYSE DU RÉSULTAT
# ======================================================

is_passed = data_drift_suite.as_dict()["summary"]["all_passed"]

if is_passed:
    status = "NO_DRIFT"
    print("✅ Aucune dérive significative détectée.")
else:
    status = "DRIFT_DETECTED"
    print("⚠️ Dérive détectée — ré-entraînement recommandé.")

if "GITHUB_OUTPUT" in os.environ:
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write(f"drift_status={status}\n")

# ======================================================
# 6. RAPPORT HTML
# ======================================================

data_drift_suite.save_html(OUTPUT_REPORT_PATH)
print(f"📄 Rapport sauvegardé : {OUTPUT_REPORT_PATH}")

sys.exit(0)
