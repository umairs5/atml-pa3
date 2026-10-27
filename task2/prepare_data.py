"""Select the task 2 data (COCO captions, annotation-derived pairs, VQAv2 complementary pairs).

python -m task2.prepare_data --raw RAW --out splits/task2

RAW holds the unzipped COCO 2014 annotation folder (annotations/) and the VQAv2 question,
annotation and complementary-pair json files. Everything is selected with seed 6304 and written
to small json files, so training only needs these files plus the images they reference.
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
from transformers import AutoTokenizer

from common.utils import save_json
from task2.vqa_norm import consensus

SEED = 6304
CAPTION_PROMPT = "Describe the image in one sentence."
QA_SUFFIX = "Answer the question using a single word or phrase."
MAX_QUESTION, MAX_CAPTION, MAX_ANSWER = 80, 64, 16
VQA_TRAIN = {"yes/no": 640, "number": 256, "other": 640}
VQA_DEV = {"yes/no": 12, "number": 8, "other": 12}
VQA_TEST = {"yes/no": 36, "number": 24, "other": 36}

# presence positives need one clearly visible instance, counting needs every instance to be
# at least COCO "medium" size, so tiny or crowd-annotated objects do not make counts ambiguous
PRESENCE_MIN_FRAC = 0.01
COUNT_MIN_AREA = 32 * 32
PRESENCE_CAP, COUNT_CAP = 100, 60  # training pairs per category
NO_COUNTING = {"broccoli", "skis", "scissors"}  # mass nouns / pairs, counts are not well defined

PLURAL = {"person": "people", "mouse": "mice", "knife": "knives", "sheep": "sheep",
          "bus": "buses", "bench": "benches", "couch": "couches", "wine glass": "wine glasses",
          "sandwich": "sandwiches", "toothbrush": "toothbrushes", "tv": "tvs"}
NO_ARTICLE = {"skis", "scissors", "broccoli"}
# extra caption words used to reject presence negatives whose captions mention the object
SYNONYMS = {
    "person": "man men woman women people boy boys girl girls child children kid kids player players "
              "guy guys lady ladies skier skiers surfer surfers snowboarder skateboarder rider riders",
    "bicycle": "bike bikes", "motorcycle": "motorbike motorbikes bike bikes",
    "airplane": "plane planes jet jets aircraft airliner", "boat": "ship ships boats",
    "traffic light": "stoplight signal", "fire hydrant": "hydrant", "parking meter": "meter meters",
    "dog": "puppy puppies", "sheep": "lamb lambs", "cow": "cows cattle bull bulls calf",
    "handbag": "purse bag bags", "suitcase": "luggage suitcases", "sports ball": "ball balls",
    "baseball bat": "bat bats", "baseball glove": "glove mitt", "tennis racket": "racket racquet",
    "wine glass": "glass glasses wine", "cup": "mug mugs cups", "hot dog": "hotdog hotdogs",
    "donut": "doughnut doughnuts donuts", "couch": "sofa sofas couches", "potted plant": "plant plants",
    "dining table": "table tables", "tv": "television tvs", "laptop": "computer laptops",
    "remote": "controller remotes", "cell phone": "phone phones cellphone", "refrigerator": "fridge",
    "teddy bear": "teddy bears", "hair drier": "dryer drier hairdryer", "skis": "ski",
}


def plural(name):
    if name in PLURAL:
        return PLURAL[name]
    return name + ("es" if re.search(r"(s|x|ch|sh)$", name) else "s")


def with_article(name):
    if name in NO_ARTICLE:
        return name
    return ("an " if name[0] in "aeiou" else "a ") + name


def keywords(name):
    words = set(name.split()) | {plural(name).split()[-1]} | set(SYNONYMS.get(name, "").split())
    return words


def load(path):
    with open(path) as f:
        return json.load(f)


class Coco:
    """Captions and instance annotations of one COCO 2014 split, reduced to what we need."""

    def __init__(self, raw, split):
        caps = load(os.path.join(raw, "annotations", f"captions_{split}.json"))
        self.images = {im["id"]: im for im in caps["images"]}
        self.captions = defaultdict(list)
        for a in sorted(caps["annotations"], key=lambda a: a["id"]):
            self.captions[a["image_id"]].append((a["id"], " ".join(a["caption"].split())))
        inst = load(os.path.join(raw, "annotations", f"instances_{split}.json"))
        self.cats = {c["id"]: c["name"] for c in inst["categories"]}
        self.objects = defaultdict(list)  # image_id -> [(category name, area, iscrowd)]
        for a in inst["annotations"]:
            self.objects[a["image_id"]].append((self.cats[a["category_id"]], a["area"], a["iscrowd"]))
        del caps, inst

    def caption_words(self, image_id):
        text = " ".join(c for _, c in self.captions[image_id]).lower()
        return set(re.findall(r"[a-z]+", text))

    def presence_positive(self, image_id, name):
        im = self.images[image_id]
        min_area = PRESENCE_MIN_FRAC * im["height"] * im["width"]
        return any(n == name and not crowd and area >= min_area for n, area, crowd in self.objects[image_id])

    def presence_negative(self, image_id, name):
        if any(n == name for n, _, _ in self.objects[image_id]):
            return False
        # COCO misses objects sometimes; drop negatives whose captions mention the category
        return not (keywords(name) & self.caption_words(image_id))

    def count(self, image_id, name):
        """Annotated count, or None if it is crowd-annotated, has small instances, or is outside 1-4."""
        objs = [(area, crowd) for n, area, crowd in self.objects[image_id] if n == name]
        if not objs or any(crowd for _, crowd in objs) or any(area < COUNT_MIN_AREA for area, _ in objs):
            return None
        return len(objs) if len(objs) <= 4 else None


def presence_question(name):
    return f"Does this image contain {with_article(name)}?"


def count_question(name):
    return f"How many {plural(name)} can you see?"


def presence_pair(coco, rng, name, pos_pool, neg_pool, used):
    pos = [i for i in pos_pool if i not in used and coco.presence_positive(i, name)]
    neg = [i for i in neg_pool if i not in used and coco.presence_negative(i, name)]
    if not pos or not neg:
        return None
    p, n = int(rng.choice(pos)), int(rng.choice(neg))
    items = [{"image_id": p, "answer": "yes"}, {"image_id": n, "answer": "no"}]
    if rng.random() < 0.5:
        items = items[::-1]
    return {"kind": "presence", "category": name, "question": presence_question(name), "items": items}


def count_pair(rng, name, by_count):
    """by_count: count -> list of image ids. Picks two different counts uniformly, then images."""
    counts = sorted(c for c, ims in by_count.items() if ims)
    if len(counts) < 2:
        return None
    c1, c2 = rng.choice(counts, size=2, replace=False)
    i1, i2 = int(rng.choice(by_count[c1])), int(rng.choice(by_count[c2]))
    if i1 == i2:
        return None
    return {"kind": "counting", "category": name, "question": count_question(name),
            "items": [{"image_id": i1, "answer": str(c1)}, {"image_id": i2, "answer": str(c2)}]}


def count_index(coco, image_ids, exclude=()):
    idx = defaultdict(lambda: defaultdict(list))
    for i in image_ids:
        if i in exclude:
            continue
        for name in {n for n, _, _ in coco.objects[i]}:
            if name in NO_COUNTING:
                continue
            c = coco.count(i, name)
            if c is not None:
                idx[name][c].append(i)
    return idx


def load_vqa(raw, split):
    qs = {q["question_id"]: q for q in load(os.path.join(raw, f"v2_OpenEnded_mscoco_{split}_questions.json"))["questions"]}
    anns = {a["question_id"]: a for a in load(os.path.join(raw, f"v2_mscoco_{split}_annotations.json"))["annotations"]}
    pairs = load(os.path.join(raw, f"v2_mscoco_{split}_complementary_pairs.json"))
    return qs, anns, pairs


def vqa_candidates(raw, split, ntok, stats):
    """Complementary pairs whose consensus answers differ, each with >= 8/10 agreement."""
    qs, anns, pairs = load_vqa(raw, split)
    out = defaultdict(list)
    st = Counter()
    for q1, q2 in pairs:
        st["pairs"] += 1
        if q1 not in anns or q2 not in anns:
            st["missing_annotation"] += 1
            continue
        a1, a2 = anns[q1], anns[q2]
        text = " ".join(qs[q1]["question"].split())
        if text != " ".join(qs[q2]["question"].split()) or a1["image_id"] == a2["image_id"]:
            st["question_or_image_mismatch"] += 1
            continue
        if a1["answer_type"] != a2["answer_type"]:
            st["type_mismatch"] += 1
            continue
        items = []
        for a in (a1, a2):
            raw_answers = [x["answer"] for x in a["answers"]]
            canon, votes, target = consensus(raw_answers)
            items.append({"question_id": a["question_id"], "image_id": a["image_id"], "target": target,
                          "canonical": canon, "votes": votes, "answers": raw_answers})
        if min(it["votes"] for it in items) < 8:
            st["low_agreement"] += 1
            continue
        if items[0]["canonical"] == items[1]["canonical"]:
            st["same_answer"] += 1
            continue
        if ntok(text) > MAX_QUESTION or max(ntok(it["target"]) for it in items) > MAX_ANSWER:
            st["too_long"] += 1
            continue
        st["kept"] += 1
        out[a1["answer_type"]].append({"pair_id": f"{q1}_{q2}", "answer_type": a1["answer_type"],
                                       "question": text, "items": items})
    stats[f"vqa_{split}"] = dict(st, **{f"kept_{k}": len(v) for k, v in out.items()})
    return out


def take_vqa(cands, quota, rng, used=None):
    chosen = []
    for t, n in quota.items():
        pool = list(cands[t])
        rng.shuffle(pool)
        picked = []
        for p in pool:
            ims = {it["image_id"] for it in p["items"]}
            if used is not None and ims & used:
                continue
            picked.append(p)
            if used is not None:
                used |= ims
            if len(picked) == n:
                break
        assert len(picked) == n, f"only {len(picked)} {t} pairs available"
        chosen += picked
    return chosen


def add_mismatch(pairs, rng):
    """Fixed mismatched-image mapping: each pair borrows the same-side image of another pair."""
    order = rng.permutation(len(pairs))
    for k, i in enumerate(order):
        other = pairs[order[(k + 1) % len(order)]]
        for s, it in enumerate(pairs[i]["items"]):
            it["mismatch_image_id"] = other["items"][s]["image_id"]
    return pairs


def eval_grounding(coco, rng, val_order, used, n_each, prefix):
    pairs = []
    names = sorted(set(coco.cats.values()))
    # presence: one pair per category, categories drawn without replacement
    for name in rng.permutation(names):
        if sum(p["kind"] == "presence" for p in pairs) == n_each:
            break
        pool = val_order[:6000]  # a shuffled slice keeps the scan cheap
        p = presence_pair(coco, rng, str(name), pool, pool, used)
        if p:
            used |= {it["image_id"] for it in p["items"]}
            pairs.append(p)
    # counting: distinct categories as well
    idx = count_index(coco, val_order, exclude=used)
    for name in rng.permutation(sorted(idx)):
        if sum(p["kind"] == "counting" for p in pairs) == n_each:
            break
        by_count = {c: [i for i in ims if i not in used] for c, ims in idx[str(name)].items()}
        p = count_pair(rng, str(name), by_count)
        if p:
            used |= {it["image_id"] for it in p["items"]}
            pairs.append(p)
    assert sum(p["kind"] == "presence" for p in pairs) == n_each
    assert sum(p["kind"] == "counting" for p in pairs) == n_each
    for k, p in enumerate(pairs):
        p["pair_id"] = f"{prefix}_{p['kind'][:4]}_{k:03d}"
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default="splits/task2")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    ntok = lambda s: len(tok(s, add_special_tokens=False).input_ids)
    stats = {"seed": SEED}

    # ---------------- training (train2014) ----------------
    tr = Coco(args.raw, "train2014")
    rng = np.random.default_rng([SEED, 1])
    caps, cap_images = [], []
    for i in rng.permutation(sorted(tr.images)):
        i = int(i)
        ok = [(aid, c) for aid, c in tr.captions[i] if ntok(c) <= MAX_CAPTION]
        if len(ok) < 3:
            continue
        pick = rng.choice(len(ok), size=3, replace=False)
        caps += [{"image_id": i, "caption_id": ok[j][0], "caption": ok[j][1]} for j in sorted(pick)]
        cap_images.append(i)
        if len(cap_images) == 6000:
            break
    stats["caption_train"] = {"images": len(cap_images), "examples": len(caps)}

    vqa_train = take_vqa(vqa_candidates(args.raw, "train2014", ntok, stats), VQA_TRAIN, np.random.default_rng([SEED, 2]))
    vqa_images = {it["image_id"] for p in vqa_train for it in p["items"]}

    # annotation-derived pools reuse the caption and VQA training images
    rng = np.random.default_rng([SEED, 3])
    pool = sorted(set(cap_images) | vqa_images)
    pres = []
    for name in sorted(set(tr.cats.values())):
        pos = [i for i in pool if tr.presence_positive(i, name)]
        neg = [i for i in pool if tr.presence_negative(i, name)]
        for p in rng.permutation(pos)[:PRESENCE_CAP]:
            items = [{"image_id": int(p), "answer": "yes"}, {"image_id": int(rng.choice(neg)), "answer": "no"}]
            if rng.random() < 0.5:
                items = items[::-1]
            pres.append({"kind": "presence", "category": name, "question": presence_question(name), "items": items})
    cnt = []
    idx = count_index(tr, pool)
    for name in sorted(idx):
        seen = set()
        for _ in range(COUNT_CAP * 5):
            p = count_pair(rng, name, idx[name])
            if p is None:
                break
            key = tuple(sorted(it["image_id"] for it in p["items"]))
            if key in seen:
                continue
            seen.add(key)
            cnt.append(p)
            if len(seen) == COUNT_CAP:
                break
    for k, p in enumerate(pres):
        p["pair_id"] = f"train_pres_{k:05d}"
    for k, p in enumerate(cnt):
        p["pair_id"] = f"train_coun_{k:05d}"
    stats["grounding_train"] = {
        "presence_pairs": len(pres), "presence_categories": len({p["category"] for p in pres}),
        "counting_pairs": len(cnt), "counting_categories": len({p["category"] for p in cnt}),
        "counting_answers": dict(Counter(it["answer"] for p in cnt for it in p["items"])),
        "image_pool": len(pool)}
    train_images = sorted(set(cap_images) | vqa_images
                          | {it["image_id"] for p in pres + cnt for it in p["items"]})
    del tr

    # ---------------- development / final test (val2014) ----------------
    va = Coco(args.raw, "val2014")
    rng = np.random.default_rng([SEED, 4])
    val_order = [int(i) for i in rng.permutation(sorted(va.images))]
    used = set()  # every evaluation set is image-disjoint from every other
    vqa_cands = vqa_candidates(args.raw, "val2014", ntok, stats)
    vqa_test = add_mismatch(take_vqa(vqa_cands, VQA_TEST, rng, used), rng)
    vqa_dev = add_mismatch(take_vqa(vqa_cands, VQA_DEV, rng, used), rng)
    gr_test = add_mismatch(eval_grounding(va, rng, val_order, used, 32, "test"), rng)
    gr_dev = add_mismatch(eval_grounding(va, rng, val_order, used, 12, "dev"), rng)
    cap_dev = []
    for i in val_order:
        if i in used:
            continue
        ok = [(aid, c) for aid, c in va.captions[i] if ntok(c) <= MAX_CAPTION]
        if not ok:
            continue
        aid, c = ok[int(rng.integers(len(ok)))]
        cap_dev.append({"image_id": i, "caption_id": aid, "caption": c})
        used.add(i)
        if len(cap_dev) == 96:
            break

    files = {"captions_train": caps, "captions_dev": cap_dev,
             "grounding_train_presence": pres, "grounding_train_counting": cnt,
             "grounding_dev": gr_dev, "grounding_test": gr_test,
             "vqa_train": vqa_train, "vqa_dev": vqa_dev, "vqa_test": vqa_test}
    for name, obj in files.items():
        save_json(obj, os.path.join(args.out, f"{name}.json"))

    def ims(pairs):
        return {it["image_id"] for p in pairs for it in p["items"]}
    eval_sets = {"captions_dev": {c["image_id"] for c in cap_dev}, "grounding_dev": ims(gr_dev),
                 "grounding_test": ims(gr_test), "vqa_dev": ims(vqa_dev), "vqa_test": ims(vqa_test)}
    names = list(eval_sets)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            assert not eval_sets[names[a]] & eval_sets[names[b]], (names[a], names[b])
    val_images = sorted(set().union(*eval_sets.values()))
    save_json({"train2014": train_images, "val2014": val_images}, os.path.join(args.out, "images.json"))

    stats["eval"] = {k: len(v) for k, v in eval_sets.items()}
    stats["eval_pairs"] = {"grounding_dev": len(gr_dev), "grounding_test": len(gr_test),
                           "vqa_dev": len(vqa_dev), "vqa_test": len(vqa_test)}
    stats["images"] = {"train2014": len(train_images), "val2014": len(val_images)}
    stats["rules"] = {
        "caption_prompt": CAPTION_PROMPT, "qa_suffix": QA_SUFFIX,
        "token_caps": {"question": MAX_QUESTION, "caption": MAX_CAPTION, "answer": MAX_ANSWER},
        "presence_positive": f"non-crowd instance with area >= {PRESENCE_MIN_FRAC:.0%} of the image",
        "presence_negative": "no annotation of the category and no caption word matching it",
        "counting": f"1-4 instances, none crowd, all with area >= {COUNT_MIN_AREA} px; skipped for {sorted(NO_COUNTING)}",
        "train_caps_per_category": {"presence": PRESENCE_CAP, "counting": COUNT_CAP},
        "vqa": "complementary pairs, same answer type, both consensus answers >= 8/10 votes and different"}
    save_json(stats, os.path.join(args.out, "selection_info.json"))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
