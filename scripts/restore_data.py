import os
import shutil
from pathlib import Path

target_md5 = "ea73f9022182842285cfc439ded81f10"
cache_dir = Path(".dvc/cache/files/md5")
dest = Path("data/version1.xlsx")

found = False
for p in cache_dir.rglob("*"):
    if p.is_file() and target_md5 in p.name:
        print(f"Found cache file: {p}")
        shutil.copy(p, dest)
        print(f"Copied to {dest}")
        found = True
        break

if not found:
    print("Could not find file in cache.")
    # Fallback: list cache
    print("Cache contents:")
    for p in cache_dir.rglob("*"):
        if p.is_file():
            print(p)
