import os
import dagshub
from dagshub.upload import Repo
from dotenv import load_dotenv

load_dotenv()

repo_owner = os.getenv("DAGSHUB_USERNAME", "rahmmaakhlefa")
repo_name = os.getenv("DAGSHUB_REPO_NAME", "NLP_Mlops")
token = os.getenv("DAGSHUB_TOKEN")

print(f"Uploading data/version1.xlsx to {repo_owner}/{repo_name}...")

try:
    # Use the more modern dagshub.upload method if available, or Repo.upload_files
    try:
        from dagshub.upload import upload_files
        upload_files(f"{repo_owner}/{repo_name}", ["data/version1.xlsx"], token=token)
        print("✅ Upload successful (via upload_files)!")
    except ImportError:
        repo = Repo(repo_owner, repo_name, token=token)
        repo.upload_files(["data/version1.xlsx"])
        print("✅ Upload successful (via Repo.upload_files)!")
except Exception as e:
    print(f"❌ Upload failed: {e}")
    import traceback
    traceback.print_exc()
