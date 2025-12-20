import os
from dagshub.upload import Repo
from dotenv import load_dotenv

load_dotenv()

repo_owner = os.getenv("DAGSHUB_USERNAME", "rahmmaakhlefa")
repo_name = os.getenv("DAGSHUB_REPO_NAME", "NLP_Mlops")
token = os.getenv("DAGSHUB_TOKEN")

print(f"Uploading data/version1.xlsx to {repo_owner}/{repo_name}...")

try:
    repo = Repo(repo_owner, repo_name, token=token)
    repo.upload_file("data/version1.xlsx", "data/version1.xlsx")
    print("✅ Upload successful!")
except Exception as e:
    print(f"❌ Upload failed: {e}")
