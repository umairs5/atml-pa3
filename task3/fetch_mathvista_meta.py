"""Question metadata for MathVista testmini (types, choices, precision, unit), without the images.
Only these columns of the parquet file are read, so the images are not downloaded.

python -m task3.fetch_mathvista_meta --out ATML-PA3-Data/mathvista_meta.json
"""
import argparse

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

from common.utils import save_json

REPO = "AI4Math/MathVista"
FILE = "data/testmini-00000-of-00001-725687bf7a18d64b.parquet"
FIELDS = ["pid", "question_type", "answer_type", "precision", "choices", "unit", "answer"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ATML-PA3-Data/mathvista_meta.json")
    args = ap.parse_args()
    revision = HfApi().dataset_info(REPO).sha
    with HfFileSystem().open(f"datasets/{REPO}@{revision}/{FILE}") as f:
        rows = pq.read_table(f, columns=FIELDS).to_pylist()
    assert len(rows) == 1000
    save_json({"revision": revision, "questions": {r["pid"]: r for r in rows}}, args.out)
    print(len(rows), "questions, revision", revision)


if __name__ == "__main__":
    main()
