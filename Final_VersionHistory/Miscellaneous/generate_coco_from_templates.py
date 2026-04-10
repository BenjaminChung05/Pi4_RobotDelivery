from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class CocoCategory:
    id: int
    name: str


def canonical_label(stem: str) -> str:
    return stem.strip().upper().replace("-", "_").replace(" ", "_")


def choose_template_files(templates_dir: Path) -> list[Path]:
    selected: dict[str, Path] = {}
    for path in templates_dir.iterdir():
        if not path.is_file():
            continue
        if path.name.startswith("._"):
            continue
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue

        key = canonical_label(path.stem)
        prev = selected.get(key)
        if prev is None:
            selected[key] = path
            continue

        # Prefer PNG if both PNG/JPG exist for same symbol.
        if prev.suffix.lower() in {".jpg", ".jpeg"} and path.suffix.lower() == ".png":
            selected[key] = path

    return sorted(selected.values(), key=lambda p: canonical_label(p.stem))


def contour_from_image(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _, mask_norm = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_norm = cv2.morphologyEx(mask_norm, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_norm = cv2.morphologyEx(mask_norm, cv2.MORPH_CLOSE, kernel, iterations=2)

    best_contour = None
    best_area = 0.0
    for mask in (mask_inv, mask_norm):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area > best_area:
            best_contour = contour
            best_area = area

    return best_contour, best_area


def contour_to_segmentation(contour):
    epsilon = 0.003 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    points = approx.reshape(-1, 2).tolist()
    flat = [float(v) for pt in points for v in pt]

    # COCO segmentation polygon needs at least 3 points (6 values).
    if len(flat) < 6:
        points = contour.reshape(-1, 2).tolist()
        flat = [float(v) for pt in points for v in pt]

    return [flat]


def build_coco(templates_dir: Path, output_json: Path) -> dict:
    files = choose_template_files(templates_dir)
    if not files:
        raise RuntimeError(f"No template images found in {templates_dir}")

    categories: list[CocoCategory] = []
    category_ids: dict[str, int] = {}
    images = []
    annotations = []

    image_id = 1
    annotation_id = 1

    for file_path in files:
        label = canonical_label(file_path.stem)
        if label not in category_ids:
            cat_id = len(categories) + 1
            categories.append(CocoCategory(id=cat_id, name=label))
            category_ids[label] = cat_id

        image = cv2.imread(str(file_path))
        if image is None:
            continue

        h, w = image.shape[:2]
        images.append(
            {
                "id": image_id,
                "width": int(w),
                "height": int(h),
                "file_name": file_path.name,
            }
        )

        contour, area = contour_from_image(image)
        if contour is None or area <= 0.0:
            # Fallback to whole-image bbox when contour extraction fails.
            bbox = [0.0, 0.0, float(w), float(h)]
            segmentation = [[0.0, 0.0, float(w), 0.0, float(w), float(h), 0.0, float(h)]]
            ann_area = float(w * h)
        else:
            x, y, bw, bh = cv2.boundingRect(contour)
            bbox = [float(x), float(y), float(bw), float(bh)]
            segmentation = contour_to_segmentation(contour)
            ann_area = float(area)

        annotations.append(
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_ids[label],
                "bbox": bbox,
                "area": ann_area,
                "iscrowd": 0,
                "segmentation": segmentation,
            }
        )

        image_id += 1
        annotation_id += 1

    coco = {
        "info": {
            "description": "Template symbols converted to COCO format",
            "version": "1.0",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": cat.id,
                "name": cat.name,
                "supercategory": "symbol",
            }
            for cat in categories
        ],
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(coco, indent=2), encoding="utf-8")
    return coco


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate COCO JSON from template images.")
    parser.add_argument("--templates", default="templates", help="Templates directory path")
    parser.add_argument("--out", default="templates/coco_templates.json", help="Output COCO JSON path")
    args = parser.parse_args()

    templates_dir = Path(args.templates).resolve()
    out_path = Path(args.out).resolve()
    coco = build_coco(templates_dir, out_path)

    print(f"Wrote {out_path}")
    print(f"images={len(coco['images'])} annotations={len(coco['annotations'])} categories={len(coco['categories'])}")


if __name__ == "__main__":
    main()
