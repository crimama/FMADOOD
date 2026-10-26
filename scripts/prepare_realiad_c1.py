#!/usr/bin/env python3
"""Build the Real-IAD C1 (single-view) tree in VisA layout from the official zips.

Split: official realiad_jsons_sv (train = normal only; test = normal + NG).
Layout: <out>/<cat>/train/good, <out>/<cat>/test/{good,bad}, <out>/<cat>/ground_truth/bad.
Only official split membership is used; no label or mask is read except to copy it.
"""
import argparse, hashlib, json, sys, zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from PIL import Image


def build(args):
    cat, root, out = args
    root, out = Path(root), Path(out)
    meta = json.loads((root / "meta/realiad_jsons_sv" / f"{cat}.json").read_text())
    prefix = meta["meta"]["prefix"].rstrip("/")
    normal = meta["meta"]["normal_class"]
    zf = zipfile.ZipFile(root / "realiad_1024" / f"{cat}.zip")
    names = set(zf.namelist())
    counts = dict(train_good=0, test_good=0, test_bad=0, masks=0)
    bad_shape = []
    digest = hashlib.sha256()

    def copy(member, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = zf.read(member)
        dest.write_bytes(data)
        digest.update(member.encode() + hashlib.sha256(data).digest())

    for split in ("train", "test"):
        for item in meta[split]:
            member = f"{prefix}/{item['image_path']}"
            if member not in names:
                raise FileNotFoundError(member)
            name = Path(item["image_path"]).name
            if item["anomaly_class"] == normal:
                kind = "train/good" if split == "train" else "test/good"
                copy(member, out / cat / kind / name)
                counts["train_good" if split == "train" else "test_good"] += 1
            else:
                if split == "train":
                    raise ValueError(f"{cat}: unexpected NG in train")
                copy(member, out / cat / "test/bad" / name)
                counts["test_bad"] += 1
                mask_member = f"{prefix}/{item['mask_path']}"
                if mask_member not in names:
                    raise FileNotFoundError(mask_member)
                copy(mask_member, out / cat / "ground_truth/bad" / (Path(name).stem + ".png"))
                counts["masks"] += 1
                with Image.open(out / cat / "test/bad" / name) as im, \
                        Image.open(out / cat / "ground_truth/bad" / (Path(name).stem + ".png")) as mk:
                    if im.size != mk.size:
                        bad_shape.append((name, im.size, mk.size))
    return cat, counts, bad_shape, digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/media/hunim/realiad")
    parser.add_argument("--out", required=True)
    parser.add_argument("--categories", nargs="*")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    cats = args.categories or sorted(p.stem for p in (Path(args.root) / "meta/realiad_jsons_sv").glob("*.json"))
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(build, [(c, args.root, args.out) for c in cats]))
    summary = {c: dict(counts=n, shape_mismatch=len(b), shape_mismatch_examples=b[:3], content_sha256=d)
               for c, n, b, d in results}
    totals = {k: sum(v["counts"][k] for v in summary.values()) for k in ("train_good", "test_good", "test_bad", "masks")}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "_prepare_summary.json").write_text(json.dumps(dict(totals=totals, categories=summary), indent=2))
    print(json.dumps(totals), "shape_mismatch", sum(v["shape_mismatch"] for v in summary.values()))
    if any(v["shape_mismatch"] for v in summary.values()):
        sys.exit(2)


if __name__ == "__main__":
    main()
