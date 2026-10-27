"""Download only the COCO 2014 images referenced by splits/task2/images.json.

python -m task2.download_images --out /content/coco
"""
import argparse
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from common.utils import load_json, save_json

URL = "http://images.cocodataset.org/{split}/COCO_{split}_{id:012d}.jpg"


def image_path(root, split, image_id):
    return os.path.join(root, split, f"COCO_{split}_{image_id:012d}.jpg")


def ok(path):
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def fetch(job):
    split, i, path = job
    if os.path.exists(path) and ok(path):
        return "cached"
    for attempt in range(4):
        try:
            tmp = path + ".part"
            urllib.request.urlretrieve(URL.format(split=split, id=i), tmp)
            os.replace(tmp, path)
            if ok(path):
                return "downloaded"
        except Exception:
            time.sleep(2 * (attempt + 1))
    return "failed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="splits/task2/images.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    ids = load_json(args.images)
    jobs = []
    for split, lst in ids.items():
        os.makedirs(os.path.join(args.out, split), exist_ok=True)
        jobs += [(split, i, image_path(args.out, split, i)) for i in lst]
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        status = list(ex.map(fetch, jobs))
    summary = {s: status.count(s) for s in set(status)}
    summary["seconds"] = round(time.time() - t0, 1)
    failed = [f"{s}/{i}" for (s, i, _), st in zip(jobs, status) if st == "failed"]
    if failed:
        summary["failed_ids"] = failed
    save_json(summary, os.path.join(args.out, "download_info.json"))
    print(summary)


if __name__ == "__main__":
    main()
