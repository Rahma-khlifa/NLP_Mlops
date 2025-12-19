"""
Script de nettoyage MLflow RAPIDE - Suppression automatique sans confirmation
ATTENTION: Utiliser avec précaution!
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')

DAGSHUB_USERNAME = os.getenv('DAGSHUB_USERNAME', 'rahmmaakhlefa')
DAGSHUB_REPO = os.getenv('DAGSHUB_REPO_NAME', 'tunsent-mlops')
DAGSHUB_TOKEN = os.getenv('DAGSHUB_TOKEN', '')
MLFLOW_TRACKING_URI = os.getenv(
    'MLFLOW_TRACKING_URI',
    f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow"
)

print("="*80)
print("⚡ NETTOYAGE MLFLOW RAPIDE - Suppression automatique")
print("="*80)


def main():
    # Configuration
    if DAGSHUB_TOKEN:
        os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USERNAME
        os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    
    # Supprimer tous les experiments (sauf Default)
    print("\n🗑️  Suppression des experiments...")
    try:
        experiments = client.search_experiments()
        deleted_exp = 0
        
        for exp in experiments:
            if exp.name.lower() != "default" and exp.experiment_id != "0":
                try:
                    # Supprimer les runs d'abord
                    runs = client.search_runs(experiment_ids=[exp.experiment_id])
                    for run in runs:
                        try:
                            client.delete_run(run.info.run_id)
                        except:
                            pass
                    
                    # Supprimer l'experiment
                    client.delete_experiment(exp.experiment_id)
                    print(f"   ✅ Supprimé: {exp.name}")
                    deleted_exp += 1
                except Exception as e:
                    print(f"   ⚠️  {exp.name}: {e}")
        
        print(f"\n   Total: {deleted_exp} experiment(s) supprimé(s)")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
    
    # Supprimer tous les modèles enregistrés
    print("\n📦 Suppression des modèles enregistrés...")
    try:
        models = client.search_registered_models()
        deleted_models = 0
        
        for model in models:
            try:
                # Supprimer toutes les versions
                versions = client.search_model_versions(f"name='{model.name}'")
                for version in versions:
                    try:
                        client.delete_model_version(model.name, version.version)
                    except:
                        pass
                
                # Supprimer le modèle
                client.delete_registered_model(model.name)
                print(f"   ✅ Supprimé: {model.name}")
                deleted_models += 1
            except Exception as e:
                print(f"   ⚠️  {model.name}: {e}")
        
        print(f"\n   Total: {deleted_models} modèle(s) supprimé(s)")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
    
    print("\n" + "="*80)
    print("✅ NETTOYAGE TERMINÉ")
    print("="*80)
    
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
