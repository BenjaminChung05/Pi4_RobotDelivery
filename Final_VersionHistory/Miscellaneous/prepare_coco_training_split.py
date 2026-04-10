from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


def load_coco(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_coco(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def split_ids(image_ids: list[int], val_ratio: float, seed: int) -> tuple[set[int], set[int]]:
    rng = random.Random(seed)
    shuffled = image_ids[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_ratio))) if len(shuffled) > 1 else 0
    val_ids = set(shuffled[:val_count])
    train_ids = set(shuffled[val_count:])
    if not train_ids and val_ids:
        moved = next(iter(val_ids))
        val_ids.remove(moved)
        train_ids.add(moved)
    return train_ids, val_ids


def subset_coco(coco: dict, keep_ids: set[int]) -> dict:
    images = [img for img in coco.get("images", []) if int(img["id"]) in keep_ids]
    image_id_set = {int(img["id"]) for img in images}
    anns = [ann for ann in coco.get("annotations", []) if int(ann["image_id"]) in image_id_set]
    return {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": images,
        "annotations": anns,
        "categories": coco.get("categories", []),
    }


def copy_images(images: list[dict], templates_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for img in images:
        src = templates_dir / img["file_name"]
        dst = out_dir / img["file_name"]
        if not src.exists():
            continue
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/val COCO split from template COCO JSON.")
    parser.add_argument("--coco", default="templates/coco_templates.json", help="Input COCO JSON")
    parser.add_argument("--templates", default="templates", help="Templates image directory")
    parser.add_argument("--out", default="dataset", help="Output dataset directory")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    coco_path = Path(args.coco).resolve()
    templates_dir = Path(args.templates).resolve()
    out_root = Path(args.out).resolve()

    if not coco_path.exists():
        raise FileNotFoundError(f"COCO file not found: {coco_path}")
    if not templates_dir.exists():
        raise FileNotFoundError(f"Templates dir not found: {templates_dir}")

    coco = load_coco(coco_path)
    image_ids = [int(img["id"]) for img in coco.get("images", [])]
    if not image_ids:
        raise RuntimeError("No images in COCO JSON.")

    train_ids, val_ids = split_ids(image_ids, args.val_ratio, args.seed)
    coco_train = subset_coco(coco, train_ids)
    coco_val = subset_coco(coco, val_ids)

    ann_dir = out_root / "annotations"
    train_img_dir = out_root / "images" / "train"
    val_img_dir = out_root / "images" / "val"

    save_coco(ann_dir / "instances_train.json", coco_train)
    save_coco(ann_dir / "instances_val.json", coco_val)
    copy_images(coco_train["images"], templates_dir, train_img_dir)
    copy_images(coco_val["images"], templates_dir, val_img_dir)

    print(f"Wrote: {ann_dir / 'instances_train.json'}")
    print(f"Wrote: {ann_dir / 'instances_val.json'}")
    print(f"Train images: {len(coco_train['images'])}, Val images: {len(coco_val['images'])}")


if __name__ == "__main__":
    main()
