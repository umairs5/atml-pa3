"""Run the frozen SigLIP encoder once over every selected image and cache the patch features.

python -m task2.cache_features --images /content/coco --out /content/features

Output: features.npy (N, 196, 768) float16 in 14x14 row-major patch order, index.json
(COCO image id -> row) and feature_info.json (timing, memory, checks).
"""
import argparse
import os
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import SiglipImageProcessor, SiglipVisionModel

from common.utils import load_json, save_json
from task2.download_images import image_path

ENCODER = "google/siglip-base-patch16-224"


class Images(Dataset):
    def __init__(self, root, items, proc):
        self.root, self.items, self.proc = root, items, proc

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        split, i = self.items[k]
        with Image.open(image_path(self.root, split, i)) as im:
            px = self.proc(images=im.convert("RGB"), return_tensors="pt")["pixel_values"][0]
        return px


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder written by task2.download_images")
    ap.add_argument("--ids", default="splits/task2/images.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ids = load_json(args.ids)
    items = [(s, i) for s in ["train2014", "val2014"] for i in ids[s]]
    assert len({i for _, i in items}) == len(items), "COCO 2014 image ids should be unique across splits"

    proc = SiglipImageProcessor.from_pretrained(ENCODER)  # fixed 224x224 resize + normalization
    enc = SiglipVisionModel.from_pretrained(ENCODER, torch_dtype=torch.float32).to(device).eval()
    loader = DataLoader(Images(args.images, items, proc), batch_size=args.batch_size, num_workers=4)

    arr = np.lib.format.open_memmap(os.path.join(args.out, "features.npy"), mode="w+",
                                    dtype=np.float16, shape=(len(items), 196, 768))
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0, row, checked = time.time(), 0, False
    with torch.no_grad():
        for px in loader:
            px = px.to(device)
            out = enc(pixel_values=px, output_hidden_states=True)
            feats = out.last_hidden_state  # encoder output with the final post_layernorm applied once
            if not checked:
                manual = enc.vision_model.post_layernorm(out.hidden_states[-1])
                ln_once = torch.allclose(manual, feats, atol=1e-4)
                assert ln_once and feats.shape[1:] == (196, 768), feats.shape
                checked = True
            arr[row:row + len(px)] = feats.half().cpu().numpy()
            row += len(px)
    arr.flush()
    secs = time.time() - t0

    save_json({str(i): k for k, (_, i) in enumerate(items)}, os.path.join(args.out, "index.json"))
    info = {"encoder": ENCODER, "n_images": len(items), "shape": [len(items), 196, 768], "dtype": "float16",
            "patch_order": "14x14 row-major", "post_layernorm_applied_once": True,
            "preprocessing": proc.to_dict(), "seconds": round(secs, 1),
            "images_per_second": round(len(items) / secs, 1),
            "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None,
            "gpu": torch.cuda.get_device_name() if device == "cuda" else "cpu",
            "bytes": os.path.getsize(os.path.join(args.out, "features.npy"))}
    save_json(info, os.path.join(args.out, "feature_info.json"))
    print({k: v for k, v in info.items() if k != "preprocessing"})


if __name__ == "__main__":
    main()
