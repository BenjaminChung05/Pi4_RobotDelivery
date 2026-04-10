from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


def canonical_label(stem: str) -> str:
    token = stem.strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "STOP_SIGN": "STOP",
        "OCTAGON": "STOP",
        "ARROW_LEFT": "LEFT",
        "LEFT_ARROW": "LEFT",
        "LEFTTURN": "LEFT",
        "ARROW_RIGHT": "RIGHT",
        "RIGHT_ARROW": "RIGHT",
        "RIGHTTURN": "RIGHT",
    }
    return aliases.get(token, token)


def pick_templates(templates_dir: Path) -> dict[str, Path]:
    selected: dict[str, Path] = {}
    for path in templates_dir.iterdir():
        if not path.is_file():
            continue
        if path.name.startswith("._"):
            continue
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue

        label = canonical_label(path.stem)
        prev = selected.get(label)
        if prev is None:
            selected[label] = path
            continue
        # Prefer PNG when duplicates exist.
        if prev.suffix.lower() in {".jpg", ".jpeg"} and path.suffix.lower() == ".png":
            selected[label] = path
    return selected


def foreground_mask(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, m1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _, m2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    mask = m1 if cv2.countNonZero(m1) < cv2.countNonZero(m2) else m2
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask


def random_background(size: int, rng: random.Random) -> np.ndarray:
    bg = np.zeros((size, size, 3), dtype=np.uint8)
    base = np.array(
        [rng.randint(30, 220), rng.randint(30, 220), rng.randint(30, 220)],
        dtype=np.uint8,
    )
    bg[:] = base

    # Add smooth gradient.
    if rng.random() < 0.8:
        axis = rng.choice([0, 1])
        grad = np.linspace(0.85, 1.15, size, dtype=np.float32)
        if axis == 0:
            grad_img = grad[:, None]
        else:
            grad_img = grad[None, :]
        for c in range(3):
            channel = bg[:, :, c].astype(np.float32) * grad_img
            bg[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)

    # Add low-amplitude texture noise.
    noise = np.random.normal(0.0, rng.uniform(4.0, 14.0), size=(size, size, 3)).astype(np.float32)
    bg = np.clip(bg.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return bg


def apply_motion_blur(image: np.ndarray, rng: random.Random) -> np.ndarray:
    k = rng.choice([3, 5, 7])
    kernel = np.zeros((k, k), dtype=np.float32)
    if rng.random() < 0.5:
        kernel[k // 2, :] = 1.0 / k
    else:
        kernel[:, k // 2] = 1.0 / k
    return cv2.filter2D(image, -1, kernel)


def warp_symbol(symbol: np.ndarray, mask: np.ndarray, out_size: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    h, w = symbol.shape[:2]
    scale = rng.uniform(0.35, 0.8)
    target = int(out_size * scale)
    ratio = target / max(h, w)
    nw, nh = max(4, int(w * ratio)), max(4, int(h * ratio))

    symbol_rs = cv2.resize(symbol, (nw, nh), interpolation=cv2.INTER_LINEAR)
    mask_rs = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

    angle = rng.uniform(-35.0, 35.0)
    m = cv2.getRotationMatrix2D((nw / 2.0, nh / 2.0), angle, 1.0)
    symbol_rt = cv2.warpAffine(symbol_rs, m, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask_rt = cv2.warpAffine(mask_rs, m, (nw, nh), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

    jitter = int(max(2, 0.08 * min(nw, nh)))
    src = np.float32([[0, 0], [nw - 1, 0], [nw - 1, nh - 1], [0, nh - 1]])
    dst = src + np.float32(
        [[rng.randint(-jitter, jitter), rng.randint(-jitter, jitter)] for _ in range(4)]
    )
    p = cv2.getPerspectiveTransform(src, dst)
    symbol_wp = cv2.warpPerspective(symbol_rt, p, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask_wp = cv2.warpPerspective(mask_rt, p, (nw, nh), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

    return symbol_wp, mask_wp


def compose(symbol: np.ndarray, mask: np.ndarray, size: int, rng: random.Random) -> np.ndarray:
    canvas = random_background(size, rng)
    sh, sw = symbol.shape[:2]
    if sh >= size or sw >= size:
        return canvas

    x = rng.randint(0, size - sw)
    y = rng.randint(0, size - sh)

    roi = canvas[y : y + sh, x : x + sw]
    m = (mask > 0).astype(np.uint8)
    inv = 1 - m

    for c in range(3):
        roi[:, :, c] = roi[:, :, c] * inv + symbol[:, :, c] * m

    out = canvas
    if rng.random() < 0.5:
        alpha = rng.uniform(0.8, 1.2)
        beta = rng.uniform(-20.0, 20.0)
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
    if rng.random() < 0.35:
        out = cv2.GaussianBlur(out, (3, 3), rng.uniform(0.2, 1.2))
    if rng.random() < 0.25:
        out = apply_motion_blur(out, rng)
    return out


def make_background_only(size: int, rng: random.Random) -> np.ndarray:
    out = random_background(size, rng)
    if rng.random() < 0.6:
        for _ in range(rng.randint(1, 5)):
            color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
            pt1 = (rng.randint(0, size - 1), rng.randint(0, size - 1))
            pt2 = (rng.randint(0, size - 1), rng.randint(0, size - 1))
            cv2.line(out, pt1, pt2, color, rng.randint(1, 3), cv2.LINE_AA)
    return out


def write_labels(path: Path, labels: list[str]) -> None:
    lines = [f"{idx} {label.lower()}" for idx, label in enumerate(labels)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate(
    templates_dir: Path,
    out_dir: Path,
    train_per_class: int,
    val_per_class: int,
    image_size: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    np.random.seed(seed)

    template_map = pick_templates(templates_dir)
    if not template_map:
        raise RuntimeError(f"No templates found in {templates_dir}")

    labels = sorted(template_map.keys())
    if "BACKGROUND" not in labels:
        labels.append("BACKGROUND")

    for split, count in (("train", train_per_class), ("val", val_per_class)):
        for label in labels:
            (out_dir / split / label).mkdir(parents=True, exist_ok=True)

            if label == "BACKGROUND":
                for idx in range(count):
                    img = make_background_only(image_size, rng)
                    out_path = out_dir / split / label / f"bg_{idx:05d}.jpg"
                    cv2.imwrite(str(out_path), img)
                continue

            src = cv2.imread(str(template_map[label]))
            if src is None:
                continue
            m = foreground_mask(src)

            for idx in range(count):
                symbol, symbol_mask = warp_symbol(src, m, image_size, rng)
                img = compose(symbol, symbol_mask, image_size, rng)
                out_path = out_dir / split / label / f"{label.lower()}_{idx:05d}.jpg"
                cv2.imwrite(str(out_path), img)

    write_labels(out_dir / "labels.txt", labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic classification dataset from templates.")
    parser.add_argument("--templates", default="templates", help="Template image directory")
    parser.add_argument("--out", default="dataset_synth_cls", help="Output dataset directory")
    parser.add_argument("--train-per-class", type=int, default=200, help="Train images per class")
    parser.add_argument("--val-per-class", type=int, default=50, help="Validation images per class")
    parser.add_argument("--size", type=int, default=224, help="Output image size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    templates_dir = Path(args.templates).resolve()
    out_dir = Path(args.out).resolve()
    generate(
        templates_dir=templates_dir,
        out_dir=out_dir,
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
        image_size=args.size,
        seed=args.seed,
    )

    print(f"Synthetic dataset ready at: {out_dir}")
    print(f"train-per-class={args.train_per_class}, val-per-class={args.val_per_class}")
    print(f"labels file: {out_dir / 'labels.txt'}")


if __name__ == "__main__":
    main()
