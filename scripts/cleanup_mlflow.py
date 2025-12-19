"""
Script de nettoyage MLflow - Supprime les anciens experiments et modèles
Utilise l'API REST de MLflow pour forcer la suppression permanente
"""
import os
import sys
import requests
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
print("🧹 NETTOYAGE MLFLOW - Suppression des anciens experiments et modèles")
print("="*80)
print(f"📍 URI: {MLFLOW_TRACKING_URI}")
print(f"👤 User: {DAGSHUB_USERNAME}")
print()


def get_auth_headers():
    """Retourne les headers d'authentification pour l'API REST"""
    if DAGSHUB_TOKEN:
        return {
            'Authorization': f'Bearer {DAGSHUB_TOKEN}',
            'Content-Type': 'application/json'
        }
    return {'Content-Type': 'application/json'}


def delete_experiment_permanently(client, exp_id, exp_name):
    """Supprime définitivement un experiment (soft delete puis hard delete)"""
    print(f"\n🗑️  Suppression de l'experiment: {exp_name} (ID: {exp_id})")
    
    try:
        # Étape 1: Soft delete via MLflow client
        try:
            client.delete_experiment(exp_id)
            print(f"   ✅ Soft delete réussi")
        except Exception as e:
            print(f"   ⚠️  Soft delete: {e}")
        
        # Étape 2: Hard delete via API REST
        # MLflow API endpoint pour suppression permanente
        base_url = MLFLOW_TRACKING_URI.replace('.mlflow', '')
        api_url = f"{base_url}/api/2.0/mlflow/experiments/delete"
        
        payload = {"experiment_id": exp_id}
        headers = get_auth_headers()
        
        response = requests.post(api_url, json=payload, headers=headers)
        
        if response.status_code in [200, 404]:
            print(f"   ✅ Suppression permanente réussie")
            return True
        else:
            print(f"   ⚠️  Réponse API: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False


def delete_registered_model(client, model_name):
    """Supprime un modèle enregistré et toutes ses versions"""
    print(f"\n🗑️  Suppression du modèle enregistré: {model_name}")
    
    try:
        # Récupérer toutes les versions
        versions = client.search_model_versions(f"name='{model_name}'")
        
        if not versions:
            print(f"   ℹ️  Aucune version trouvée")
            return True
        
        print(f"   📦 {len(versions)} version(s) trouvée(s)")
        
        # Supprimer chaque version
        for version in versions:
            try:
                client.delete_model_version(model_name, version.version)
                print(f"      ✅ Version {version.version} supprimée")
            except Exception as e:
                print(f"      ⚠️  Version {version.version}: {e}")
        
        # Supprimer le modèle lui-même
        try:
            client.delete_registered_model(model_name)
            print(f"   ✅ Modèle '{model_name}' supprimé")
            return True
        except Exception as e:
            print(f"   ⚠️  Suppression du modèle: {e}")
            return False
            
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return False


def delete_runs_in_experiment(client, exp_id):
    """Supprime tous les runs d'un experiment"""
    try:
        runs = client.search_runs(experiment_ids=[exp_id])
        
        if not runs:
            return 0
        
        count = 0
        for run in runs:
            try:
                client.delete_run(run.info.run_id)
                count += 1
            except Exception as e:
                print(f"      ⚠️  Run {run.info.run_id}: {e}")
        
        return count
    except Exception:
        return 0


def main():
    """Fonction principale de nettoyage"""
    
    # Configuration MLflow
    if DAGSHUB_TOKEN:
        os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USERNAME
        os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    
    # ========================================================================
    # PARTIE 1: Lister tous les experiments
    # ========================================================================
    print("\n📋 ÉTAPE 1: Liste des experiments")
    print("-" * 80)
    
    try:
        all_experiments = client.search_experiments()
        
        if not all_experiments:
            print("   ℹ️  Aucun experiment trouvé")
        else:
            print(f"   Trouvé {len(all_experiments)} experiment(s):\n")
            for exp in all_experiments:
                lifecycle = exp.lifecycle_stage
                status_icon = "🗑️" if lifecycle == "deleted" else "✅"
                print(f"   {status_icon} [{exp.experiment_id}] {exp.name} ({lifecycle})")
    except Exception as e:
        print(f"   ❌ Erreur lors de la liste: {e}")
        all_experiments = []
    
    # ========================================================================
    # PARTIE 2: Supprimer les experiments (sauf "Default")
    # ========================================================================
    print("\n\n🗑️  ÉTAPE 2: Suppression des experiments")
    print("-" * 80)
    
    # Demander confirmation
    print("\n⚠️  ATTENTION: Cette action va supprimer TOUS les experiments (sauf 'Default')")
    print("   Cela inclut tous les runs, métriques et artifacts associés.")
    
    response = input("\n   Continuer? (oui/non): ").strip().lower()
    
    if response not in ['oui', 'yes', 'y', 'o']:
        print("\n❌ Annulé par l'utilisateur")
        return 0
    
    deleted_count = 0
    for exp in all_experiments:
        # Ne pas supprimer l'experiment "Default"
        if exp.name.lower() == "default" or exp.experiment_id == "0":
            print(f"\n⏭️  Ignoré: {exp.name} (experiment système)")
            continue
        
        # Supprimer les runs d'abord
        runs_deleted = delete_runs_in_experiment(client, exp.experiment_id)
        if runs_deleted > 0:
            print(f"   🗑️  {runs_deleted} run(s) supprimé(s)")
        
        # Supprimer l'experiment
        if delete_experiment_permanently(client, exp.experiment_id, exp.name):
            deleted_count += 1
    
    print(f"\n   ✅ {deleted_count} experiment(s) supprimé(s)")
    
    # ========================================================================
    # PARTIE 3: Lister et supprimer les modèles enregistrés
    # ========================================================================
    print("\n\n📦 ÉTAPE 3: Suppression des modèles enregistrés")
    print("-" * 80)
    
    try:
        registered_models = client.search_registered_models()
        
        if not registered_models:
            print("   ℹ️  Aucun modèle enregistré trouvé")
        else:
            print(f"   Trouvé {len(registered_models)} modèle(s) enregistré(s):\n")
            for model in registered_models:
                print(f"   📦 {model.name}")
            
            print("\n⚠️  Supprimer tous ces modèles?")
            response = input("   Continuer? (oui/non): ").strip().lower()
            
            if response in ['oui', 'yes', 'y', 'o']:
                model_deleted_count = 0
                for model in registered_models:
                    if delete_registered_model(client, model.name):
                        model_deleted_count += 1
                
                print(f"\n   ✅ {model_deleted_count} modèle(s) supprimé(s)")
            else:
                print("\n   ⏭️  Suppression des modèles annulée")
    
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
    
    # ========================================================================
    # RÉSUMÉ FINAL
    # ========================================================================
    print("\n" + "="*80)
    print("✅ NETTOYAGE TERMINÉ")
    print("="*80)
    print(f"   🗑️  Experiments supprimés: {deleted_count}")
    print(f"   📦 Modèles supprimés: {model_deleted_count if 'model_deleted_count' in locals() else 0}")
    print()
    print("💡 Vous pouvez maintenant relancer votre entraînement avec:")
    print("   python scripts/train.py")
    print()
    
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n❌ Interrompu par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
