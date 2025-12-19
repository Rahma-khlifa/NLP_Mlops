"""
Download best model and vectorizer from MLflow to local model_registry.
ROBUST VERSION: Handles various artifact structures from MLflow.
"""
import os
import sys
import shutil
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')

        
# Config
DAGSHUB_USERNAME = os.getenv('DAGSHUB_USERNAME', '')
DAGSHUB_REPO = os.getenv('DAGSHUB_REPO_NAME', 'NLP_Mlops')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow")
EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT', 'tunsentt')

# Destination
DEST_DIR = BASE_DIR / 'model_registry' / 'Best_Election_Model'
DEST_PATH = DEST_DIR / 'production.pkl'


def list_artifacts_recursive(client, run_id, path=""):
    """Recursively list all artifacts."""
    artifacts = []
    try:
        for item in client.list_artifacts(run_id, path):
            if item.is_dir:
                artifacts.extend(list_artifacts_recursive(client, run_id, item.path))
            else:
                artifacts.append(item.path)
    except Exception as e:
        print(f"⚠️  Error listing artifacts: {e}")
    return artifacts


def download_from_run(client, run_id):
    """Download model and vectorizer from a specific run."""
    print(f"\n🎯 Downloading artifacts from run: {run_id}")
    
    # Get run info
    run = client.get_run(run_id)
    model_name = run.data.params.get('model_type', 'Unknown')
    f1_score = run.data.metrics.get('f1_score', 0)
    
    print(f"   Model: {model_name}")
    print(f"   F1-Score: {f1_score:.4f}")
    
    # List all artifacts
    print("\n📋 Listing artifacts...")
    all_artifacts = list_artifacts_recursive(client, run_id)
    
    if not all_artifacts:
        print(f"❌ No artifacts found in run {run_id}")
        return False
    
    print(f"   Found {len(all_artifacts)} artifact(s):")
    for art in all_artifacts[:20]:
        print(f"      - {art}")
    
    # ============================================================
    # STRATEGY 1: Look for model.pkl in standard locations
    # ============================================================
    # When mlflow.sklearn.log_model(artifact_path="model") is used,
    # the model is saved as "model/model.pkl"
    model_pkl_candidates = [
        a for a in all_artifacts 
        if a.endswith('/model.pkl') or a == 'model.pkl'
    ]
    
    # ============================================================
    # STRATEGY 2: Also check for Election_*/model.pkl pattern
    # ============================================================
    if not model_pkl_candidates:
        model_pkl_candidates = [
            a for a in all_artifacts 
            if 'election' in a.lower() and a.endswith('.pkl') and 'vectorizer' not in a.lower()
        ]
    
    # ============================================================
    # STRATEGY 3: Any .pkl in model/ directory
    # ============================================================
    if not model_pkl_candidates:
        model_pkl_candidates = [
            a for a in all_artifacts 
            if a.startswith('model/') and a.endswith('.pkl')
        ]
    
    # Find vectorizer
    vec_artifacts = [
        a for a in all_artifacts 
        if 'vectorizer' in a.lower() and a.endswith('.pkl')
    ]
    
    # Validate we found both
    if not model_pkl_candidates:
        print(f"\n❌ No model.pkl found")
        print(f"   Available artifacts: {all_artifacts}")
        return False
    
    if not vec_artifacts:
        print(f"\n❌ No vectorizer.pkl found")
        print(f"   Available artifacts: {all_artifacts}")
        return False
    
    # Download model
    model_artifact_path = model_pkl_candidates[0]
    print(f"\n⬇️  Downloading model: {model_artifact_path}")
    try:
        local_model_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, 
            artifact_path=model_artifact_path
        )
        DEST_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_model_path, DEST_PATH)
        print(f"✅ Model saved: {DEST_PATH}")
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Download vectorizer
    vec_artifact_path = vec_artifacts[0]
    print(f"⬇️  Downloading vectorizer: {vec_artifact_path}")
    try:
        vec_local = mlflow.artifacts.download_artifacts(
            run_id=run_id, 
            artifact_path=vec_artifact_path
        )
        vec_dest = DEST_DIR / 'tfidf_vectorizer.pkl'
        shutil.copy2(vec_local, vec_dest)
        print(f"✅ Vectorizer saved: {vec_dest}")
    except Exception as e:
        print(f"❌ Failed to download vectorizer: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Optional: Register in MLflow Model Registry
    try:
        reg_model_name = f"Election_{model_name}"
        # Use parent directory of model.pkl
        if '/' in model_artifact_path:
            model_dir = model_artifact_path.rsplit('/', 1)[0]
        else:
            model_dir = 'model'
        
        model_uri = f"runs:/{run_id}/{model_dir}"
        
        print(f"\n🔐 Registering model as '{reg_model_name}'...")
        result = mlflow.register_model(model_uri, reg_model_name)
        print(f"✅ Registered as version {result.version}")
    except Exception as e:
        print(f"⚠️  Registration warning (non-critical): {e}")
    
    return True


def main():
    print("="*80)
    print("📦 DOWNLOADING BEST MODEL FROM MLFLOW")
    print("="*80)
    print(f"🔧 MLflow URI: {MLFLOW_TRACKING_URI}\n")
    
    # Configure authentication
    DAGSHUB_TOKEN = os.getenv('DAGSHUB_TOKEN', '')
    if DAGSHUB_TOKEN:
        os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_USERNAME
        os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    
    # Get experiment
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if not exp:
        print(f"❌ Experiment '{EXPERIMENT_NAME}' not found")
        return 1
    
    print(f"🔎 Finding runs from experiment '{EXPERIMENT_NAME}'...")
    
    # Strategy: Get RECENT runs with good F1 scores
    # (Latest training should be the best one to use)
    try:
        df = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="metrics.f1_score > 0",
            order_by=["start_time DESC"],  # Most recent first
            max_results=10  # Check last 10 runs to be safe
        )
    except Exception as e:
        print(f"❌ Error searching runs: {e}")
        return 1
    
    if df.empty:
        print("❌ No runs found with f1_score > 0")
        return 1
    
    print(f"   Found {len(df)} recent runs\n")
    
    # Try each recent run until we find one with both artifacts
    for idx, row in df.iterrows():
        run_id = row['run_id']
        
        success = download_from_run(client, run_id)
        
        if success:
            print("\n" + "="*80)
            print("🎉 SUCCESS! Model ready for deployment")
            print("="*80)
            print(f"📁 Location: {DEST_DIR}")
            print(f"   ✅ production.pkl")
            print(f"   ✅ tfidf_vectorizer.pkl")
            print(f"\n📦 Next step: Deploy your API!")
            return 0
        else:
            print(f"\n⚠️  Run {run_id} incomplete, trying next...\n")
            print("-" * 80)
    
    print("\n❌ Could not find a complete run with both model and vectorizer")
    print("💡 Make sure train.py completed successfully")
    print("💡 Check MLflow UI to verify artifacts were logged")
    return 1


if __name__ == '__main__':
    sys.exit(main())