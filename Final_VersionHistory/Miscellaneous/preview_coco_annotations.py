from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_coco(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Render COCO bbox/segmentation previews to disk.")
    parser.add_argument("--coco", default="dataset/annotations/instances_train.json", help="COCO annotation path")
    parser.add_argument("--images", default="dataset/images/train", help="Image directory")
    parser.add_argument("--out", default="dataset/preview", help="Output preview directory")
    parser.add_argument("--limit", type=int, default=20, help="Max preview images")
    args = parser.parse_args()

    coco_path = Path(args.coco).resolve()
    images_dir = Path(args.images).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    coco = load_coco(coco_path)
    categories = {int(cat["id"]): cat["name"] for cat in coco.get("categories", [])}
    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco.get("annotations", []):
        anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    count = 0
    for image_info in coco.get("images", []):
        if count >= args.limit:
            break

        image_id = int(image_info["id"])
        file_name = image_info["file_name"]
        image_path = images_dir / file_name
        img = cv2.imread(str(image_path))
        if img is None:
            continue

        for ann in anns_by_image.get(image_id, []):
            x, y, w, h = ann["bbox"]
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cat_name = categories.get(int(ann["category_id"]), "unknown")
            cv2.putText(
                img,
                cat_name,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

            for poly in ann.get("segmentation", []):
                pts = []
                for i in range(0, len(poly), 2):
                    pts.append([int(poly[i]), int(poly[i + 1])])
                if len(pts) >= 3:
                    pts_np = np.array([[p] for p in pts], dtype=np.int32)
                    cv2.polylines(img, [pts_np], True, (255, 180, 0), 1)

        out_path = out_dir / file_name
        cv2.imwrite(str(out_path), img)
        count += 1

    print(f"Wrote {count} preview images to {out_dir}")


if __name__ == "__main__":
    main()
