import os
import shutil
import subprocess
from pathlib import Path
from dotenv import load_dotenv

def run(cmd, env=None):
    print(f"Executing: {cmd}")
    result = subprocess.run(cmd, env=env, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print(f"Success: {result.stdout}")
    return result.returncode

# 1. Load env
load_dotenv()
env = os.environ.copy()

# 2. Restore from cache
target_md5 = "ea73f9022182842285cfc439ded81f10"
# In DVC cache, path is md5[0:2] / md5[2:]
cache_file = Path(".dvc/cache/files/md5/ea/73f9022182842285cfc439ded81f10")
dest = Path("data/version1.xlsx")

if cache_file.exists():
    print(f"Found cache file at {cache_file}")
    shutil.copy(cache_file, dest)
    print(f"Restored to {dest}")
else:
    print(f"Cache file NOT found at {cache_file}")
    # Search for it
    candidates = list(Path(".dvc/cache").rglob("*73f9022182842285cfc439ded81f10*"))
    if candidates:
        shutil.copy(candidates[0], dest)
        print(f"Restored from {candidates[0]} to {dest}")
    else:
        print("CRITICAL: File not found in cache!")

# 3. Verify restore
if dest.exists():
    print(f"Verified: {dest} exists, size: {dest.stat().st_size}")
else:
    print(f"FAILED to restore {dest}")

# 4. DVC Add
run("python -m dvc add data/version1.xlsx")

# 5. DVC Push
# Ensure AWS creds are in the environment (load_dotenv handled it)
run("python -m dvc push -r origin -v", env=env)
