import os
import mlflow
import dagshub
from dotenv import load_dotenv
from pathlib import Path

# Load .env
BASE_DIR = Path(__file__).resolve().parent.parent
dotenv_path = BASE_DIR / '.env'
print(f"DEBUG: Looking for .env at {dotenv_path}")
load_dotenv(dotenv_path)

# Print all keys from .env to see what's available
with open(dotenv_path, 'r') as f:
    keys = [line.split('=')[0].strip() for line in f if '=' in line and not line.startswith('#')]
print(f"DEBUG: Keys found in .env: {keys}")

user = os.getenv('DAGSHUB_USERNAME', 'NOT_SET')
repo = os.getenv('DAGSHUB_REPO_NAME', 'NOT_SET')
token = os.getenv('DAGSHUB_TOKEN', 'NOT_SET')

print(f"DEBUG: DAGSHUB_USERNAME={user}")
print(f"DEBUG: DAGSHUB_REPO_NAME={repo}")
print(f"DEBUG: DAGSHUB_TOKEN={'PRESENT' if token != 'NOT_SET' else 'MISSING'}")

uri = f"https://dagshub.com/{user}/{repo}.mlflow"
print(f"DEBUG: MLFLOW_TRACKING_URI={uri}")

print("\nTesting Dagshub Init...")
try:
    dagshub.init(repo_owner=user, repo_name=repo, mlflow=True)
    print("✅ dagshub.init success")
except Exception as e:
    print(f"❌ dagshub.init failed: {e}")

print("\nTesting MLflow Connection...")
try:
    mlflow.set_tracking_uri(uri)
    if token != 'NOT_SET':
        os.environ['MLFLOW_TRACKING_USERNAME'] = user
        os.environ['MLFLOW_TRACKING_PASSWORD'] = token
    
    # Try to list experiments
    exps = mlflow.search_experiments()
    print(f"✅ mlflow search_experiments success: {len(exps)} experiments found")
    for e in exps:
        print(f"  - {e.name}")
except Exception as e:
    print(f"❌ mlflow connection failed: {e}")
    import traceback
    traceback.print_exc()
