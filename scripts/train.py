"""
Script d'entraînement des modèles avec MLflow - MLOps Election (Version de débogage)
Entraîne et évalue les modèles ML classiques et TunBERT
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import time
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# ML classiques
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from scipy.sparse import issparse

# MLflow & DagsHub
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.exceptions import MlflowException
import dagshub

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')

# Model registry for deployment artifacts
MODEL_REGISTRY_DIR = BASE_DIR / 'model_registry' / 'Best_Election_Model'
MODEL_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

# XGBoost (optionnel)
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️  XGBoost non installé - modèle ignoré")

print("="*80)
print("🚀 ENTRAÎNEMENT DES MODÈLES - MLOps Election")
print("="*80)
print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📁 Répertoire: {BASE_DIR}")
print()

# ... (Les fonctions setup_mlflow, load_preprocessed_data, load_vectorizer, get_ml_models, calculate_metrics restent identiques) ...
def setup_mlflow():
    """Configure MLflow tracking avec DagsHub"""
    DAGSHUB_USERNAME = os.getenv('DAGSHUB_USERNAME', 'rahmmaakhlefa')
    DAGSHUB_REPO = os.getenv('DAGSHUB_REPO_NAME', 'tunsent-mlops')
    DAGSHUB_TOKEN = os.getenv('DAGSHUB_TOKEN', '')
    
    MLFLOW_TRACKING_URI = os.getenv(
        'MLFLOW_TRACKING_URI',
        f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow"
    )
    # Allow overriding the experiment name from environment (useful if experiment was deleted)
    
    print("\n🔧 Configuration MLflow & DagsHub")
    print("-" * 80)
    
    # Configure DagsHub authentication if token is available
    if DAGSHUB_TOKEN:
        os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USERNAME
        os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN
        print(f"✅ DagsHub authentication configured with token")
    else:
        # Only try dagshub.init() if no token (local development)
        try:
            dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO, mlflow=True)
            print(f"✅ DagsHub initialisé: {DAGSHUB_USERNAME}/{DAGSHUB_REPO}")
        except Exception as e:
            print(f"⚠️  DagsHub warning: {e}")
    
    # Configurer MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("tunsentt")
    print(f"✅ MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    print()


# ============================================================================
# Preprocessing will be done in-memory - no separate loading needed
# ============================================================================


def get_ml_models():
    """Retourne la configuration des modèles ML"""
    models = {
        'Neural_Network': MLPClassifier(hidden_layer_sizes=(10, 10), activation='logistic', solver='adam', max_iter=500, random_state=42),
        'Logistic_Regression': LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=1000, random_state=42),
        'Random_Forest': RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1, class_weight='balanced', random_state=42),
        'SVM_Linear': SVC(kernel='linear', random_state=42),
        'SVM_RBF': SVC(kernel='rbf', random_state=42),
    }
    if HAS_XGB:
        models['XGBoost'] = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, eval_metric='logloss', random_state=42, verbosity=0)
    return models

def calculate_metrics(y_true, y_pred):
    """Calcule toutes les métriques"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0)
    }


def train_ml_models(X_train, X_test, y_train, y_test, vectorizer=None):
    """Entraîne tous les modèles ML avec MLflow tracking"""
    print("🤖 ENTRAÎNEMENT DES MODÈLES ML CLASSIQUES")
    print("="*80)
    
    models = get_ml_models()
    results = []
    trained_models = {}
    failed_models = []
    best_run_id = None
    best_f1 = -1
    
    for model_name, model in models.items():
        print(f"\n🔄 Entraînement: {model_name}")
        
        try:
            Xtr = X_train.toarray() if issparse(X_train) else X_train
            Xte = X_test.toarray() if issparse(X_test) else X_test
        except Exception:
            Xtr, Xte = X_train, X_test
        
        with mlflow.start_run(run_name=f"ML_{model_name}"):
            # Le bloc try/except principal est déplacé pour entourer tout le run
            try:
                start_time = time.time()
                model.fit(Xtr, y_train)
                training_time = time.time() - start_time
                
                y_pred = model.predict(Xte)
                metrics = calculate_metrics(y_test, y_pred)
                
                mlflow.log_param("model_type", model_name)
                mlflow.log_param("algorithm", type(model).__name__)
                
                if hasattr(model, 'get_params'):
                    params = model.get_params()
                    for key, value in params.items():
                        if value is not None and not callable(value):
                            try: mlflow.log_param(f"hp_{key}", value)
                            except: pass
                
                mlflow.log_metric("accuracy", metrics['accuracy'])
                mlflow.log_metric("precision", metrics['precision'])
                mlflow.log_metric("recall", metrics['recall'])
                mlflow.log_metric("f1_score", metrics['f1_score'])
                mlflow.log_metric("training_time_seconds", training_time)
                
                # Log modèle avec signature et input_example
                try:
                    # Prepare a small input example for signature inference
                    try:
                        X_sample = Xtr[:1]
                        if hasattr(X_sample, 'toarray'):
                            X_sample_df = pd.DataFrame(X_sample.toarray())
                        else:
                            X_sample_df = pd.DataFrame(X_sample)
                        y_sample_pred = model.predict(X_sample_df)
                        signature = infer_signature(X_sample_df, y_sample_pred)
                        input_example = X_sample_df.iloc[[0]]
                    except Exception:
                        signature = None
                        input_example = None

                    unique_artifact_path = f"model_{model_name}"
                    print(f"--- DÉBUT: Enregistrement du modèle '{model_name}' ---")
                    print(f"   Nom de l'artefact: {unique_artifact_path}")
                    print(f"   Nom dans le registre: Election_{model_name}")

                    mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path=unique_artifact_path,
                        registered_model_name=f"Election_{model_name}",
                        signature=signature,
                        input_example=input_example
                    )
                    print(f"--- SUCCÈS: Modèle '{model_name}' enregistré sans erreur. ---")
                except Exception as e:
                    print("\n" + "="*80)
                    print(f"❌ ERREUR FATALE LORS DE L'ENREGISTREMENT DU MODÈLE '{model_name}'")
                    print("="*80)
                    import traceback
                    print("Traceback complet:")
                    traceback.print_exc()
                    print("="*80)
                    print("Le script va s'arrêter. Le dossier du modèle n'a PAS été envoyé à MLflow.")
                    sys.exit(1)

                # Log vectorizer as artifact in THIS run so it's available for deployment
                if vectorizer is not None:
                    import tempfile
                    # Create a temp dir and write the vectorizer with a stable filename
                    with tempfile.TemporaryDirectory() as td:
                        vec_tmp = os.path.join(td, 'tfidf_vectorizer.pkl')
                        with open(vec_tmp, 'wb') as vf:
                            pickle.dump(vectorizer, vf)
                        try:
                            mlflow.log_artifact(vec_tmp, artifact_path='vectorizer')
                            print(f"   ✅ Vectorizer sauvegardé comme artifact")
                        except Exception as _e:
                            print(f"   ⚠️  Warning: failed to log vectorizer artifact: {_e}")

                cm = confusion_matrix(y_test, y_pred)
                cm_df = pd.DataFrame(cm, index=['Actual_0', 'Actual_1'], columns=['Pred_0', 'Pred_1'])
                
                # Log confusion matrix as artifact (in-memory, no file)
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
                    tmp_path = tmp.name
                    cm_df.to_csv(tmp_path)
                
                try:
                    mlflow.log_artifact(tmp_path, artifact_path='confusion_matrices')
                finally:
                    # Ensure file is closed before deletion
                    try:
                        os.unlink(tmp_path)
                    except (PermissionError, OSError):
                        pass  # File will be cleaned up by OS eventually
                
                trained_models[model_name] = model
                results.append({'Model': model_name, **metrics, 'Training_Time': training_time})
                
                # Track best run for later
                if metrics['f1_score'] > best_f1:
                    best_f1 = metrics['f1_score']
                    best_run_id = mlflow.active_run().info.run_id
                
                print(f"   ✅ Terminé - F1: {metrics['f1_score']:.4f} ({training_time:.2f}s)")
                
            except Exception as e:
                print(f"   ❌ Erreur inattendue dans le run: {e}")
                failed_models.append(model_name)
                mlflow.log_param("status", "failed")
                mlflow.log_param("error", str(e))
    
    if failed_models:
        print(f"\n⚠️  Modèles échoués: {failed_models}")
    
    return results, trained_models, best_run_id

# ... (Les fonctions save_results_summary et main restent identiques) ...
def save_results_summary(results):
    """Sauvegarde le résumé des résultats"""
    print("\n📊 RÉSUMÉ DES RÉSULTATS")
    print("="*80)
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('f1_score', ascending=False)
    
    print(df_results.to_string(index=False))
    
    # Sauvegarder CSV dans model registry
    results_path = MODEL_REGISTRY_DIR / 'ml_models_results.csv'
    df_results.to_csv(results_path, index=False)
    print(f"\n✅ Résultats sauvegardés: {results_path}")
    
    best = df_results.iloc[0]
    print(f"\n🏆 MEILLEUR MODÈLE: {best['Model']}")
    print(f"   F1-Score: {best['f1_score']:.4f}")
    print(f"   Accuracy: {best['accuracy']:.4f}")
    
    return df_results

def main():
    """Pipeline principal d'entraînement"""
    try:
        # Configurer MLflow
        setup_mlflow()

        # CLI: optional custom data path
        parser = argparse.ArgumentParser()
        parser.add_argument('--data-path', '-d', help='Optional path to raw data Excel file (overrides default)')
        args = parser.parse_args()

        # Always run preprocessing in-memory using functions from scripts/preprocess.py
        # Ensure scripts folder is importable
        SCRIPTS_DIR = Path(__file__).resolve().parent
        if str(SCRIPTS_DIR) not in sys.path:
            sys.path.insert(0, str(SCRIPTS_DIR))

        try:
            import preprocess
        except Exception as e:
            raise RuntimeError(f"Could not import preprocess module: {e}")

        # Determine data path
        data_path = Path(args.data_path) if args.data_path else (preprocess.DATA_DIR / 'version1.xlsx')
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # Run preprocessing functions in-memory
        print("\n" + "="*80)
        print("🧪 PREPROCESSING - Integrated in Training Pipeline")
        print("="*80)
        df = preprocess.load_and_explore_data(data_path)
        df = preprocess.preprocess_text(df, text_col='comments')
        X, vectorizer = preprocess.vectorize_text(df, text_col='cleaned', max_features=5000)
        y = df['target'].values
        X_train, X_val, X_test, y_train, y_val, y_test = preprocess.split_data(X, y, test_size=0.15, val_size=0.15, random_state=42)
        
        # Entraîner les modèles ML (pass vectorizer to log as artifact in each run)
        results, trained_models, best_run_id = train_ml_models(X_train, X_test, y_train, y_test, vectorizer=vectorizer)
        
        # Sauvegarder résumé
        df_results = save_results_summary(results)
        
        # Sauvegarder le vectorizer dans model_registry (co-located with production model)
        vectorizer_path = MODEL_REGISTRY_DIR / 'tfidf_vectorizer.pkl'
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(vectorizer, f)
        print(f"✅ Vectorizer sauvegardé: {vectorizer_path}")
        
        if best_run_id:
            print(f"✅ Best run ID for registration: {best_run_id}")
        
        print("\n" + "="*80)
        print("✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
        print("="*80)
        print(f"📁 Model Registry: {MODEL_REGISTRY_DIR}")
        print(f"🔗 MLflow UI: Consultez DagsHub pour voir les runs")
        print(f"\n📦 Next step: python scripts/register_best_model.py")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())