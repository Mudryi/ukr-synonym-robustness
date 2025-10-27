#!/usr/bin/env python3
from huggingface_hub import hf_hub_download
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dest", default="resources/fasttext_uk", help="Destination directory")
parser.add_argument("--repo", default="lang-uk/fasttext_uk_cbow", help="HuggingFace repo id")
parser.add_argument("--filename", default="cbow.uk.300.bin", help="File name in the repo")
args = parser.parse_args()

dest = Path(args.dest)
dest.mkdir(parents=True, exist_ok=True)

print(f"Downloading {args.filename} from {args.repo} -> {dest}")
path = hf_hub_download(repo_id=args.repo, filename=args.filename, local_dir=str(dest), repo_type="model")
print("Saved to:", path)
