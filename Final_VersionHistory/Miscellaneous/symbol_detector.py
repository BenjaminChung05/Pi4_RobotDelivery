from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from utils import clamp


def _canonical_label(label: str) -> str:
    token = label.strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "STOP_SIGN": "STOP",
        "OCTAGON": "STOP",
        "BUTTON": "STOP",
        "HAZARD": "STOP",
        "ARROW_LEFT": "LEFT",
        "LEFT_ARROW": "LEFT",
        "LEFTTURN": "LEFT",
        "LEFT": "LEFT",
        "ARROW_RIGHT": "RIGHT",
        "RIGHT_ARROW": "RIGHT",
        "RIGHTTURN": "RIGHT",
        "RIGHT": "RIGHT",
        "FINGERPRINT": "FINGERPRINT",
        "RECYCLE": "RECYCLE",
        "QR": "QR",
    }
    return aliases.get(token, token)


ACTIVE_SYMBOL_LABELS = {"STOP", "LEFT", "RIGHT", "RECYCLE", "QR", "FINGERPRINT"}


def _template_key_from_path(label: str) -> str:
    return label.strip().upper().replace("-", "_").replace(" ", "_")


def preprocess_gray(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray


def preprocess_bin(bgr: np.ndarray) -> np.ndarray:
    gray = preprocess_gray(bgr)
    min_dim = min(gray.shape[:2])
    block = min(141, max(21, (min_dim // 2) * 2 - 1))
    c_val = 7 if float(np.std(gray)) < 25.0 else 15
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, c_val
    )
    k = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
    return th


def preprocess_edges(bgr: np.ndarray) -> np.ndarray:
    gray = preprocess_gray(bgr)
    edges = cv2.Canny(gray, 50, 140)
    k = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=2)
    return edges


def _normalize_bin(bin_img: np.ndarray) -> np.ndarray:
    if bin_img is None:
        return bin_img
    white = int(np.count_nonzero(bin_img))
    if white > (bin_img.size // 2):
        return cv2.bitwise_not(bin_img)
    return bin_img


def _contour_roi_bin(bin_img: np.ndarray, cnt: np.ndarray) -> Optional[np.ndarray]:
    if bin_img is None or cnt is None:
        return None
    x, y, w, h = cv2.boundingRect(cnt)
    if w <= 1 or h <= 1:
        return None
    roi = bin_img[y : y + h, x : x + w]
    roi = _normalize_bin(roi)
    roi = cv2.resize(roi, (50, 50), interpolation=cv2.INTER_NEAREST)
    return roi


def _bin_iou(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST)
    a_bin = (a > 127).astype(np.uint8)
    b_bin = (b > 127).astype(np.uint8)
    inter = int(np.count_nonzero(a_bin & b_bin))
    union = int(np.count_nonzero(a_bin | b_bin))
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def largest_contour(bin_img: np.ndarray) -> Optional[np.ndarray]:
    cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


def is_arrow(cnt: np.ndarray, defect_depth: int = 2200) -> bool:
    peri = cv2.arcLength(cnt, True)
    if peri <= 1e-6:
        return False
    approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)
    if approx is None:
        return False
    if len(approx) < 7 or len(approx) > 12:
        return False
    try:
        hull_idx = cv2.convexHull(approx, returnPoints=False)
        if hull_idx is None or len(hull_idx) < 3:
            return False
        defects = cv2.convexityDefects(approx, hull_idx)
        if defects is None:
            return False
        adaptive_depth = max(int(0.015 * peri * 256.0), int(defect_depth))
        meaningful = 0
        for i in range(defects.shape[0]):
            _, _, _, depth = defects[i, 0]
            if depth > adaptive_depth:
                meaningful += 1
        return meaningful == 2
    except cv2.error:
        return False


def convexity_defect_count(cnt: np.ndarray, defect_depth: int = 1200) -> int:
    try:
        hull_idx = cv2.convexHull(cnt, returnPoints=False)
        if hull_idx is None or len(hull_idx) < 3:
            return 0
        defects = cv2.convexityDefects(cnt, hull_idx)
        if defects is None:
            return 0
        count = 0
        for i in range(defects.shape[0]):
            _, _, _, depth = defects[i, 0]
            if depth > defect_depth:
                count += 1
        return count
    except cv2.error:
        return 0


def compute_hu_moments(cnt: np.ndarray) -> Optional[np.ndarray]:
    """Compute rotation-invariant Hu moments for shape matching."""
    try:
        moments = cv2.HuMoments(cnt)
        return np.array([float(m[0]) for m in moments])
    except cv2.error:
        return None


def hu_moment_distance(m1: np.ndarray, m2: np.ndarray) -> float:
    """Compute distance between Hu moment vectors (log-scale for numerical stability)."""
    if m1 is None or m2 is None:
        return float("inf")
    d = 0.0
    for i in range(min(len(m1), len(m2))):
        h1 = float(m1[i]) if m1[i] != 0 else 1e-10
        h2 = float(m2[i]) if m2[i] != 0 else 1e-10
        d += abs(np.log10(abs(h1)) - np.log10(abs(h2)))
    return float(d)


def arrow_direction(cnt: np.ndarray) -> Optional[str]:
    pts = cnt.reshape(-1, 2).astype(np.float32)
    if len(pts) < 3:
        return None

    tail_ref = None
    try:
        hull_idx = cv2.convexHull(cnt, returnPoints=False)
        if hull_idx is not None and len(hull_idx) >= 3:
            defects = cv2.convexityDefects(cnt, hull_idx)
            if defects is not None and defects.shape[0] > 0:
                deepest_i = int(np.argmax(defects[:, 0, 3]))
                far_idx = int(defects[deepest_i, 0, 2])
                tail_ref = pts[far_idx]
    except cv2.error:
        tail_ref = None

    if tail_ref is None:
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            return None
        tail_ref = np.array([float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"])], dtype=np.float32)

    d = np.sum((pts - tail_ref) ** 2, axis=1)
    tip = pts[int(np.argmax(d))]
    dx = float(tail_ref[0] - tip[0])
    dy = float(tail_ref[1] - tip[1])
    if abs(dx) > abs(dy):
        return "RIGHT" if dx > 0 else "LEFT"
    return "DOWN" if dy > 0 else "UP"


@dataclass
class FastFilterConfig:
    roi_top_ratio: float = 0.15
    roi_bottom_ratio: float = 0.98
    roi_left_ratio: float = 0.05
    roi_right_ratio: float = 0.95
    threshold_value: int = 90
    min_area: float = 90.0
    padding: int = 12
    morph_kernel_size: int = 3
    use_otsu: bool = True
    min_bbox_size: int = 14
    max_bbox_aspect_ratio: float = 3.8
    min_fill_ratio: float = 0.18
    max_center_y_ratio: float = 0.80


@dataclass
class SymbolDetectorConfig:
    model_path: str = "models/symbol_classifier.tflite"
    labels_path: str = "models/labels.txt"
    templates_dir: str = "templates"
    input_size: tuple[int, int] = (224, 224)
    confidence_threshold: float = 0.60
    shape_match_threshold: float = 0.48
    shape_match_gap_min: float = 0.010
    template_confidence_threshold: float = 0.20
    prefer_template_label: bool = True
    enable_qr_detector: bool = True
    fast_filter: FastFilterConfig = field(default_factory=FastFilterConfig)


@dataclass
class SymbolCandidate:
    found: bool
    bbox: tuple[int, int, int, int] | None = None
    area: float = 0.0
    roi_bbox: tuple[int, int, int, int] | None = None
    mask: np.ndarray | None = None


@dataclass
class SymbolResult:
    enabled: bool
    label: str | None = None
    confidence: float = 0.0
    detected_shape: str | None = None
    shape_confidence: float = 0.0
    action_label: str | None = None
    bbox: tuple[int, int, int, int] | None = None
    accepted: bool = False
    reason: str | None = None


@dataclass
class ShapeTemplate:
    label: str
    image_bgr: np.ndarray
    image_gray: np.ndarray
    image_edge: np.ndarray
    contour: np.ndarray
    verts: int
    extent: float
    solidity: float
    circularity: float
    bin_img: np.ndarray
    hu_moments: Optional[np.ndarray] = None
    defect_count: int = 0


@dataclass
class FeatureTemplate:
    label: str
    kp: list
    des: np.ndarray


@dataclass
class TemplateMatchInfo:
    label: str
    kp_count: int
    des: np.ndarray


class TemplateLibrary:
    def __init__(self, templates_dir: str, swap_octagon_hazard: bool = False):
        self.templates_dir = templates_dir
        self.swap_octagon_hazard = bool(swap_octagon_hazard)
        self.shape_templates: List[ShapeTemplate] = []
        self.feature_templates: List[FeatureTemplate] = []

    @staticmethod
    def _shape_descriptor(cnt: np.ndarray) -> Tuple[int, float, float, float]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True) if peri > 1e-6 else None
        verts = int(len(approx)) if approx is not None else 0

        area = float(cv2.contourArea(cnt))
        x, y, w, h = cv2.boundingRect(cnt)
        extent = area / float(max(1, w * h))

        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / max(1e-6, hull_area)

        circularity = (4.0 * np.pi * area) / max(1e-6, peri * peri)
        return verts, float(extent), float(solidity), float(circularity)

    @staticmethod
    def _resize_template_image(image_bgr: np.ndarray, size: tuple[int, int] = (160, 160)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        resized_bgr = cv2.resize(image_bgr, size, interpolation=cv2.INTER_AREA)
        resized_gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)
        resized_edge = cv2.Canny(resized_gray, 60, 160)
        return resized_bgr, resized_gray, resized_edge

    @staticmethod
    def _contour_features(contour: np.ndarray) -> tuple[int, float, float]:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True) if perimeter > 1e-6 else contour
        vertices = int(len(approx))
        _, _, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w / max(1, h))
        area = float(cv2.contourArea(contour))
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / max(1e-6, hull_area)
        return vertices, aspect_ratio, solidity

    def load(self) -> "TemplateLibrary":
        if not os.path.isdir(self.templates_dir):
            raise RuntimeError(f"Templates dir not found: {self.templates_dir}")

        files = [
            f for f in os.listdir(self.templates_dir)
            if f.lower().endswith((".png", ".jpg")) and not f.startswith("._")
        ]
        by_base = {}
        for f in files:
            base = os.path.splitext(f)[0].strip().upper()
            ext = os.path.splitext(f)[1].lower()
            prev = by_base.get(base)
            if prev is None:
                by_base[base] = f
            elif prev.lower().endswith(".jpg") and ext == ".png":
                by_base[base] = f
        files = list(by_base.values())
        if not files:
            raise RuntimeError(f"No template images found in {self.templates_dir}")

        akaze = cv2.AKAZE_create()
        self.shape_templates = []
        self.feature_templates = []

        for fn in sorted(files):
            label = os.path.splitext(fn)[0].strip().upper()
            if self.swap_octagon_hazard:
                if label == "HAZARD":
                    label = "OCTAGON"
                elif label == "OCTAGON":
                    label = "HAZARD"
            label = _canonical_label(label)
            if label not in ACTIVE_SYMBOL_LABELS:
                continue
            path = os.path.join(self.templates_dir, fn)

            image = cv2.imread(path)
            if image is None:
                continue

            if label in {"FINGERPRINT"}:
                gray = preprocess_gray(image)
                kp, des = akaze.detectAndCompute(gray, None)
                if des is not None and len(kp) >= 16:
                    self.feature_templates.append(FeatureTemplate(label, kp, des))
                if label == "FINGERPRINT":
                    continue

            th = preprocess_bin(image)
            cnt = largest_contour(th)
            if cnt is None or cv2.contourArea(cnt) < 200:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            roi = th[y : y + h, x : x + w]
            roi = _normalize_bin(roi)
            roi = cv2.resize(roi, (50, 50), interpolation=cv2.INTER_NEAREST)
            verts, extent, solidity, circularity = self._shape_descriptor(cnt)
            hu_moms = compute_hu_moments(cnt)
            defects = convexity_defect_count(cnt, defect_depth=1700)
            resized_bgr, resized_gray, resized_edge = self._resize_template_image(image)
            self.shape_templates.append(
                ShapeTemplate(
                    label=label,
                    image_bgr=resized_bgr,
                    image_gray=resized_gray,
                    image_edge=resized_edge,
                    contour=cnt,
                    verts=verts,
                    extent=extent,
                    solidity=solidity,
                    circularity=circularity,
                    bin_img=roi,
                    hu_moments=hu_moms,
                    defect_count=defects,
                )
            )

        if not self.shape_templates and not self.feature_templates:
            raise RuntimeError("No templates loaded successfully.")
        return self


class TemplateSymbolDetector:
    def __init__(
        self,
        lib: TemplateLibrary,
        min_area: int = 1800,
        match_thresh: float = 0.30,
        cross_shape_thresh: float = 0.72,
        fp_min_good: int = 12,
        fp_ratio: float = 0.75,
    ):
        self.shape_templates = lib.shape_templates
        self.feature_templates = lib.feature_templates
        self.arrow_templates = [t for t in self.shape_templates if t.label in {"UP", "DOWN", "LEFT", "RIGHT"}]
        self.non_arrow_templates = [t for t in self.shape_templates if t.label not in {"UP", "DOWN", "LEFT", "RIGHT"}]

        self.min_area = int(min_area)
        self.match_thresh = float(match_thresh)
        self.cross_shape_thresh = float(cross_shape_thresh)
        self.cross_fallback_thresh = max(self.cross_shape_thresh, 0.90)
        self.arrow_defect_depth = 2200

        self.fp_min_good = int(fp_min_good)
        self.fp_ratio = float(fp_ratio)

        self.akaze = cv2.AKAZE_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.qr = cv2.QRCodeDetector()
        self.arrow_bin_thresh = 0.26
        self.qr_bin_thresh = 0.34
        self.recycle_bin_thresh = 0.45
        self.stop_bin_thresh = 0.30
        self.qr_template = next((t for t in self.shape_templates if t.label == "QR"), None)
        self.recycle_template = next((t for t in self.shape_templates if t.label == "RECYCLE"), None)
        self.stop_template = next((t for t in self.shape_templates if t.label == "STOP"), None)

    @staticmethod
    def _shape_descriptor(cnt: np.ndarray) -> tuple[int, float, float, float]:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True) if peri > 1e-6 else None
        verts = int(len(approx)) if approx is not None else 0

        area = float(cv2.contourArea(cnt))
        x, y, w, h = cv2.boundingRect(cnt)
        extent = area / float(max(1, w * h))

        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / max(1e-6, hull_area)

        circularity = (4.0 * np.pi * area) / max(1e-6, peri * peri)
        return verts, float(extent), float(solidity), float(circularity)

    @staticmethod
    def _is_specular_reflection(crop_bgr: np.ndarray) -> bool:
        """Detect if region is likely a specular reflection (very bright, low contrast interior)."""
        if crop_bgr is None or crop_bgr.size == 0:
            return False
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))
        std_brightness = float(np.std(gray))
        
        # Reflections are: very bright (>220) and low internal contrast (<20)
        if mean_brightness > 220 and std_brightness < 20:
            return True
        
        # Also check for saturated pixels (values > 250)
        saturated = float(np.count_nonzero(gray > 250)) / float(max(1, gray.size))
        if saturated > 0.35 and mean_brightness > 200:
            return True
            
        return False

    def _geometry_based_template_match(self, crop_bgr: np.ndarray) -> tuple[str | None, float]:
        """Light-invariant classification using only shape descriptors (Hu moments, defects, solidity)."""
        if not self.shape_templates:
            return None, 0.0

        # Reject obvious reflections first
        if self._is_specular_reflection(crop_bgr):
            return None, 0.0

        th = preprocess_bin(crop_bgr)
        cnt = largest_contour(th)
        if cnt is None or cv2.contourArea(cnt) < 180.0:
            return None, 0.0

        cand_hu = compute_hu_moments(cnt)
        cand_defects = convexity_defect_count(cnt, defect_depth=1700)
        cand_verts, cand_extent, cand_solidity, cand_circularity = self._shape_descriptor(cnt)

        best_label: str | None = None
        best_score = float("inf")

        for template in self.shape_templates:
            # Compare Hu moments (rotation/scale invariant)
            hu_score = hu_moment_distance(cand_hu, template.hu_moments)
            
            # Compare geometric descriptors
            vertex_penalty = 0.15 * abs(float(cand_verts - template.verts))
            extent_penalty = 0.30 * abs(cand_extent - template.extent)
            solidity_penalty = 0.25 * abs(cand_solidity - template.solidity)
            circularity_penalty = 0.20 * abs(cand_circularity - template.circularity)
            defect_penalty = 0.10 * abs(cand_defects - template.defect_count)

            combined_score = (
                (0.40 * hu_score) +     # Hu moments: most important for shape
                (0.20 * vertex_penalty) +
                (0.15 * extent_penalty) +
                (0.15 * solidity_penalty) +
                (0.10 * circularity_penalty)
            )
            
            if template.label in {"LEFT", "RIGHT"}:
                # For arrows, defects are critical (2 sharp indentations)
                combined_score += (0.15 * defect_penalty)

            if combined_score < best_score:
                best_score = combined_score
                best_label = template.label

        if best_label is None or best_score > 3.0:
            return None, 0.0
        return best_label, float(best_score)

    @staticmethod
    def _descriptor_penalty(cand_desc: Tuple[int, float, float, float], templ: ShapeTemplate) -> float:
        c_verts, c_extent, c_solidity, c_circularity = cand_desc
        p_verts = 0.02 * abs(float(c_verts - templ.verts))
        p_extent = 0.35 * abs(c_extent - templ.extent)
        p_solidity = 0.25 * abs(c_solidity - templ.solidity)
        p_circularity = 0.20 * abs(c_circularity - templ.circularity)
        return float(p_verts + p_extent + p_solidity + p_circularity)

    @staticmethod
    def _resize_template_image(image_bgr: np.ndarray, size: tuple[int, int] = (160, 160)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        resized_bgr = cv2.resize(image_bgr, size, interpolation=cv2.INTER_AREA)
        resized_gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)
        resized_edge = cv2.Canny(resized_gray, 60, 160)
        return resized_bgr, resized_gray, resized_edge

    def _direct_template_match(self, crop_bgr: np.ndarray) -> tuple[str | None, float]:
        """Light-invariant classification: use geometry-based matching instead of pixel-level intensity."""
        return self._geometry_based_template_match(crop_bgr)

    def _rotated_template_match(
        self,
        crop_bgr: np.ndarray,
        labels: set[str],
        angles: tuple[int, ...] = (-30, -20, -10, 0, 10, 20, 30),
    ) -> tuple[str | None, float]:
        """Light-invariant matching across rotations using geometry descriptors (already rotation-invariant)."""
        if not self.shape_templates:
            return None, float("inf")

        # Hu moments are rotation-invariant; primary rotation handling is now done implicitly
        # For additional robustness, try candidate multiple times with small rotations
        best_label: str | None = None
        best_score = float("inf")
        
        for angle in [0, 15, -15]:  # Reduced angles since Hu moments handle rotation
            if angle != 0:
                center = (crop_bgr.shape[1] / 2.0, crop_bgr.shape[0] / 2.0)
                matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
                rotated = cv2.warpAffine(
                    crop_bgr,
                    matrix,
                    (crop_bgr.shape[1], crop_bgr.shape[0]),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(255, 255, 255),
                )
            else:
                rotated = crop_bgr

            label, score = self._geometry_based_template_match(rotated)
            if label in labels and score < best_score:
                best_score = score
                best_label = label

        if best_label is None or best_score > 3.0:
            return None, float("inf")
        return best_label, float(best_score)

    def _contour_template_match(self, crop_bgr: np.ndarray) -> tuple[str | None, float, float]:
        if not self.shape_templates:
            return None, float("inf"), 0.0

        th = preprocess_bin(crop_bgr)
        cnt = largest_contour(th)
        if cnt is None or cv2.contourArea(cnt) < 180.0:
            return None, float("inf"), 0.0

        cand_desc = self._shape_descriptor(cnt)
        best_per_label: dict[str, float] = {}
        for templ in self.shape_templates:
            s = cv2.matchShapes(cnt, templ.contour, cv2.CONTOURS_MATCH_I1, 0.0)
            s += self._descriptor_penalty(cand_desc, templ)
            prev = best_per_label.get(templ.label)
            if prev is None or s < prev:
                best_per_label[templ.label] = float(s)

        if not best_per_label:
            return None, float("inf"), 0.0

        ranked = sorted(best_per_label.items(), key=lambda kv: kv[1])
        best_label, best_score = ranked[0]
        second_best = ranked[1][1] if len(ranked) > 1 else (best_score + 1.0)
        margin = float(second_best - best_score)
        return best_label, float(best_score), margin

    @staticmethod
    def _count_triangle_contours(image_bgr: np.ndarray) -> int:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        _, mask_norm = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        kernel = np.ones((3, 3), dtype=np.uint8)
        triangle_count = 0
        seen_boxes: list[tuple[int, int, int, int]] = []

        for mask in (mask_inv, mask_norm):
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < 60.0:
                    continue
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True) if perimeter > 1e-6 else contour
                if len(approx) != 3:
                    continue
                x, y, w, h = cv2.boundingRect(approx)
                if w < 6 or h < 6:
                    continue
                if any(abs(x - bx) < 4 and abs(y - by) < 4 and abs(w - bw) < 4 and abs(h - bh) < 4 for bx, by, bw, bh in seen_boxes):
                    continue
                seen_boxes.append((x, y, w, h))
                triangle_count += 1

        return triangle_count

    def detect(self, frame_bgr: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None, debug: bool = False):
        x_off = y_off = 0
        work = frame_bgr

        if roi is not None:
            x1, y1, x2, y2 = roi
            work = frame_bgr[y1:y2, x1:x2]
            x_off, y_off = x1, y1

        bin_img = preprocess_bin(work)
        cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        edges = preprocess_edges(work)

        results = []
        debug_items = []

        try:
            qr_text, qr_points, _ = self.qr.detectAndDecode(work)
            if qr_points is not None and qr_text and qr_text.strip():
                pts = qr_points.astype(int)
                x, y, w, h = cv2.boundingRect(pts)
                if w * h >= self.min_area:
                    results.append({"label": "QR", "bbox": (x + x_off, y + y_off, w, h), "score": -2.0})
        except cv2.error:
            pass

        for c in cnts:
            if cv2.contourArea(c) < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(c)
            if w <= 0 or h <= 0:
                continue

            cand = work[y : y + h, x : x + w]

            th = preprocess_bin(cand)
            cnt = largest_contour(th)
            if cnt is None or cv2.contourArea(cnt) < 200:
                continue
            cand_desc = self._shape_descriptor(cnt)
            cand_bin = _contour_roi_bin(th, cnt)
            if cand_bin is None:
                continue

            fp_score = self._match_fingerprint(cand)
            if fp_score is not None:
                results.append({"label": "FINGERPRINT", "bbox": (x + x_off, y + y_off, w, h), "score": fp_score})
                continue

            if self._count_triangle_contours(cand) >= 3:
                results.append({"label": "RECYCLE", "bbox": (x + x_off, y + y_off, w, h), "score": 0.06})
                continue

            c_verts, c_extent, c_solidity, c_circularity = cand_desc

            best_arrow_label = None
            best_arrow_iou = 0.0
            for t in self.arrow_templates:
                iou = _bin_iou(cand_bin, t.bin_img)
                if iou > best_arrow_iou:
                    best_arrow_iou = iou
                    best_arrow_label = t.label

            qr_iou = _bin_iou(cand_bin, self.qr_template.bin_img) if self.qr_template is not None else 0.0
            recycle_iou = _bin_iou(cand_bin, self.recycle_template.bin_img) if self.recycle_template is not None else 0.0
            stop_iou = _bin_iou(cand_bin, self.stop_template.bin_img) if self.stop_template is not None else 0.0

            if recycle_iou >= self.recycle_bin_thresh:
                if recycle_iou >= max(qr_iou, best_arrow_iou, stop_iou) + 0.02:
                    results.append({"label": "RECYCLE", "bbox": (x + x_off, y + y_off, w, h), "score": float(1.0 - recycle_iou)})
                    continue

            arrow_like = is_arrow(cnt, defect_depth=self.arrow_defect_depth) and c_solidity < 0.92 and c_verts >= 6
            if (
                best_arrow_label is not None
                and best_arrow_iou >= self.arrow_bin_thresh
                and arrow_like
                and best_arrow_iou >= max(qr_iou, recycle_iou, stop_iou) + 0.03
            ):
                results.append({"label": best_arrow_label, "bbox": (x + x_off, y + y_off, w, h), "score": float(1.0 - best_arrow_iou)})
                continue

            if qr_iou >= self.qr_bin_thresh and qr_iou >= (recycle_iou + 0.04):
                results.append({"label": "QR", "bbox": (x + x_off, y + y_off, w, h), "score": float(1.0 - qr_iou)})
                continue

            if stop_iou >= self.stop_bin_thresh:
                results.append({"label": "STOP", "bbox": (x + x_off, y + y_off, w, h), "score": float(1.0 - stop_iou)})
                continue

        results.sort(key=lambda r: r["score"])
        if debug:
            return results, edges, debug_items
        return results, edges

    def _match_fingerprint(self, cand_bgr: np.ndarray) -> Optional[float]:
        if not self.feature_templates:
            return None

        edges = preprocess_edges(cand_bgr)
        edge_density = float(np.count_nonzero(edges)) / float(max(1, edges.size))
        if edge_density < 0.015:
            return None

        gray = preprocess_gray(cand_bgr)
        kp, des = self.akaze.detectAndCompute(gray, None)
        if des is None or len(kp) < 8:
            return None

        best_good = 0
        fingerprint_templates = [t for t in self.feature_templates if t.label == "FINGERPRINT"]
        templates = fingerprint_templates if fingerprint_templates else self.feature_templates
        for t in templates:
            knn = self.bf.knnMatch(des, t.des, k=2)
            good = 0
            for m, n in knn:
                if m.distance < self.fp_ratio * n.distance:
                    good += 1
            best_good = max(best_good, good)

        if best_good >= max(8, self.fp_min_good - 4):
            return float(-best_good)
        return None

    def _match_feature(self, cand_bgr: np.ndarray, label: str, min_good: int, ratio: float) -> Optional[float]:
        feats = [t for t in self.feature_templates if t.label == label]
        if not feats:
            return None

        gray = preprocess_gray(cand_bgr)
        kp, des = self.akaze.detectAndCompute(gray, None)
        if des is None or len(kp) < 8:
            return None

        best_good = 0
        for t in feats:
            knn = self.bf.knnMatch(des, t.des, k=2)
            good = 0
            for m, n in knn:
                if m.distance < ratio * n.distance:
                    good += 1
            best_good = max(best_good, good)

        if best_good >= min_good:
            return float(-best_good)
        return None


class TFLiteSymbolDetector:
    def __init__(self, config: SymbolDetectorConfig | None = None) -> None:
        self.config = config or SymbolDetectorConfig()
        self.enabled = True
        self.reason = "Week 2 template detector loaded."
        self.qr_detector = cv2.QRCodeDetector() if self.config.enable_qr_detector else None

        library = TemplateLibrary(self.config.templates_dir).load()
        self.detector = TemplateSymbolDetector(library)
        self.shape_templates = self.detector.shape_templates
        self.feature_templates = self.detector.feature_templates

    @staticmethod
    def _is_line_like_region(crop_bgr: np.ndarray) -> bool:
        if crop_bgr is None or crop_bgr.size == 0:
            return False
        bin_img = preprocess_bin(crop_bgr)
        cnt = largest_contour(bin_img)
        if cnt is None or cv2.contourArea(cnt) < 120.0:
            return False

        x, y, w, h = cv2.boundingRect(cnt)
        if w < 2 or h < 2:
            return False

        area = float(cv2.contourArea(cnt))
        fill_ratio = area / float(max(1, w * h))
        aspect = max(w, h) / float(max(1, min(w, h)))
        width_ratio = w / float(max(1, bin_img.shape[1]))
        height_ratio = h / float(max(1, bin_img.shape[0]))
        touches_bottom = (y + h) >= (bin_img.shape[0] - 2)
        touches_top = y <= 1
        touches_left = x <= 1
        touches_right = (x + w) >= (bin_img.shape[1] - 2)

        # Floor lines typically form long, sparse blobs that run through a crop.
        if touches_bottom and (aspect >= 3.0 or fill_ratio <= 0.34):
            return True
        if touches_top and touches_bottom and aspect >= 2.6:
            return True
        if touches_top and touches_bottom and width_ratio <= 0.32:
            return True
        if touches_left and touches_right and height_ratio <= 0.32:
            return True
        if aspect >= 5.0 and width_ratio <= 0.36:
            return True
        if touches_bottom and height_ratio >= 0.45 and width_ratio <= 0.20 and aspect >= 2.0:
            return True
        return False

    @staticmethod
    def _arrow_direction_ok(crop_bgr: np.ndarray, expected_label: str) -> bool:
        if expected_label not in {"LEFT", "RIGHT"}:
            return True
        if crop_bgr is None or crop_bgr.size == 0:
            return False

        mask = preprocess_bin(crop_bgr)
        cnt = largest_contour(mask)
        if cnt is None or cv2.contourArea(cnt) < 220.0:
            return False

        # Outdoor glare can corrupt template scores; enforce geometric arrow agreement.
        if not is_arrow(cnt, defect_depth=1700):
            return False
        direction = arrow_direction(cnt)
        return direction == expected_label

    @staticmethod
    def _looks_like_arrow(crop_bgr: np.ndarray) -> bool:
        if crop_bgr is None or crop_bgr.size == 0:
            return False
        mask = preprocess_bin(crop_bgr)
        cnt = largest_contour(mask)
        if cnt is None or cv2.contourArea(cnt) < 220.0:
            return False
        if not is_arrow(cnt, defect_depth=1700):
            return False
        direction = arrow_direction(cnt)
        return direction in {"LEFT", "RIGHT"}

    def fast_filter(self, frame_bgr: np.ndarray) -> SymbolCandidate:
        cfg = self.config.fast_filter
        frame_h, frame_w = frame_bgr.shape[:2]
        x1 = int(frame_w * cfg.roi_left_ratio)
        y1 = int(frame_h * cfg.roi_top_ratio)
        x2 = int(frame_w * cfg.roi_right_ratio)
        y2 = int(frame_h * cfg.roi_bottom_ratio)
        roi = frame_bgr[y1:y2, x1:x2]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if cfg.use_otsu:
            _, mask_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            _, mask_norm = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        else:
            _, mask_inv = cv2.threshold(gray, cfg.threshold_value, 255, cv2.THRESH_BINARY_INV)
            _, mask_norm = cv2.threshold(gray, cfg.threshold_value, 255, cv2.THRESH_BINARY)

        kernel_size = cfg.morph_kernel_size
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_norm = cv2.morphologyEx(mask_norm, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_norm = cv2.morphologyEx(mask_norm, cv2.MORPH_CLOSE, kernel, iterations=2)

        best_contour = None
        best_area = 0.0
        best_mask = mask_inv
        roi_area = float((y2 - y1) * (x2 - x1))
        for mask in (mask_inv, mask_norm):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            for contour in sorted(contours, key=cv2.contourArea, reverse=True):
                area = float(cv2.contourArea(contour))
                if area < cfg.min_area:
                    continue
                if area > roi_area * 0.70:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                if x <= 1 or y <= 1 or x + w >= mask.shape[1] - 1 or y + h >= mask.shape[0] - 1:
                    continue
                if area > best_area:
                    best_area = area
                    best_contour = contour
                    best_mask = mask
                break

        if best_contour is None:
            return SymbolCandidate(found=False, roi_bbox=(x1, y1, x2 - x1, y2 - y1), mask=best_mask)

        contour = best_contour
        area = best_area
        box_x, box_y, box_w, box_h = cv2.boundingRect(contour)
        if box_w < cfg.min_bbox_size or box_h < cfg.min_bbox_size:
            return SymbolCandidate(found=False, area=area, roi_bbox=(x1, y1, x2 - x1, y2 - y1), mask=best_mask)

        # Reject thin/stripe-like candidates that usually come from the floor line.
        fill_ratio = area / float(max(1, box_w * box_h))
        if fill_ratio < cfg.min_fill_ratio:
            return SymbolCandidate(found=False, area=area, roi_bbox=(x1, y1, x2 - x1, y2 - y1), mask=best_mask)

        center_y = y1 + box_y + (box_h / 2.0)
        if center_y > (frame_h * cfg.max_center_y_ratio):
            return SymbolCandidate(found=False, area=area, roi_bbox=(x1, y1, x2 - x1, y2 - y1), mask=best_mask)

        aspect = max(box_w, box_h) / max(1, min(box_w, box_h))
        if aspect > cfg.max_bbox_aspect_ratio:
            return SymbolCandidate(found=False, area=area, roi_bbox=(x1, y1, x2 - x1, y2 - y1), mask=best_mask)

        pad = cfg.padding
        box_x = max(0, x1 + box_x - pad)
        box_y = max(0, y1 + box_y - pad)
        box_w = min(frame_w - box_x, box_w + (2 * pad))
        box_h = min(frame_h - box_y, box_h + (2 * pad))
        return SymbolCandidate(found=True, bbox=(box_x, box_y, box_w, box_h), area=area, roi_bbox=(x1, y1, x2 - x1, y2 - y1), mask=best_mask)

    def classify(self, frame_bgr: np.ndarray, candidate: SymbolCandidate) -> SymbolResult:
        if not candidate.found or candidate.bbox is None:
            return SymbolResult(enabled=self.enabled, accepted=False, reason="No candidate contour found.")

        x, y, w, h = candidate.bbox
        frame_h, frame_w = frame_bgr.shape[:2]
        x1 = int(clamp(float(x), 0.0, float(max(0, frame_w - 1))))
        y1 = int(clamp(float(y), 0.0, float(max(0, frame_h - 1))))
        x2 = int(clamp(float(x + w), float(x1 + 1), float(frame_w)))
        y2 = int(clamp(float(y + h), float(y1 + 1), float(frame_h)))

        center_y = y1 + ((y2 - y1) / 2.0)
        if center_y > (frame_h * self.config.fast_filter.max_center_y_ratio):
            return SymbolResult(
                enabled=self.enabled,
                bbox=(x1, y1, x2 - x1, y2 - y1),
                accepted=False,
                reason="Candidate too low in frame (likely line).",
            )

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0 or crop.ndim != 3 or crop.shape[0] < 16 or crop.shape[1] < 16:
            return SymbolResult(enabled=self.enabled, bbox=candidate.bbox, reason="Candidate crop is empty.")
        if self._is_line_like_region(crop):
            return SymbolResult(
                enabled=self.enabled,
                bbox=(x1, y1, x2 - x1, y2 - y1),
                accepted=False,
                reason="Candidate geometry looks like tracking line.",
            )

        fp_score = self.detector._match_fingerprint(crop)
        if fp_score is None:
            pad_w = int(0.25 * (x2 - x1))
            pad_h = int(0.25 * (y2 - y1))
            ex1 = max(0, x1 - pad_w)
            ey1 = max(0, y1 - pad_h)
            ex2 = min(frame_w, x2 + pad_w)
            ey2 = min(frame_h, y2 + pad_h)
            expanded = frame_bgr[ey1:ey2, ex1:ex2]
            if expanded.size > 0:
                fp_score = self.detector._match_fingerprint(expanded)
        if fp_score is not None:
            good_matches = max(0.0, float(-fp_score))
            confidence = float(clamp(good_matches / 20.0, 0.0, 1.0))
            return SymbolResult(
                enabled=self.enabled,
                label="FINGERPRINT",
                confidence=confidence,
                detected_shape="FINGERPRINT",
                shape_confidence=confidence,
                action_label="FINGERPRINT",
                bbox=(x1, y1, x2 - x1, y2 - y1),
                accepted=True,
                reason="Fingerprint detected by AKAZE.",
            )

        contour_label, contour_score, contour_margin = self.detector._contour_template_match(crop)

        rotated_label, rotated_score = self.detector._rotated_template_match(crop, {"QR"})
        if rotated_label == "QR" and rotated_score <= 0.26 and contour_label != "RECYCLE":
            confidence = float(clamp(1.0 - rotated_score, 0.0, 1.0))
            return SymbolResult(
                enabled=self.enabled,
                label="QR",
                confidence=confidence,
                detected_shape="QR",
                shape_confidence=confidence,
                action_label="QR",
                bbox=(x1, y1, x2 - x1, y2 - y1),
                accepted=True,
                reason="Rotated QR template match.",
            )

        rotated_label, rotated_score = self.detector._rotated_template_match(crop, {"STOP"})
        if (
            rotated_label == "STOP"
            and rotated_score <= 0.24
            and (contour_label != "RECYCLE" or contour_score > 0.35)
            and not self._looks_like_arrow(crop)
        ):
            confidence = float(clamp(1.0 - rotated_score, 0.0, 1.0))
            return SymbolResult(
                enabled=self.enabled,
                label="STOP",
                confidence=confidence,
                detected_shape="STOP",
                shape_confidence=confidence,
                action_label="STOP",
                bbox=(x1, y1, x2 - x1, y2 - y1),
                accepted=True,
                reason="Rotated STOP template match.",
            )

        if contour_label in {"LEFT", "RIGHT", "RECYCLE", "STOP"}:
            score_limits = {
                "LEFT": 0.42,
                "RIGHT": 0.42,
                "RECYCLE": 0.44,
                "STOP": 0.44,
            }
            stop_ok = True
            if contour_label == "STOP":
                stop_mask = preprocess_bin(crop)
                stop_cnt = largest_contour(stop_mask)
                stop_ok = False
                if stop_cnt is not None and cv2.contourArea(stop_cnt) >= 180.0:
                    peri = cv2.arcLength(stop_cnt, True)
                    approx = cv2.approxPolyDP(stop_cnt, 0.02 * peri, True) if peri > 1e-6 else stop_cnt
                    stop_ok = 6 <= len(approx) <= 10

            if contour_label in {"LEFT", "RIGHT"} and not self._arrow_direction_ok(crop, contour_label):
                stop_ok = False

            if stop_ok and contour_score <= score_limits.get(contour_label, 0.44) and contour_margin >= 0.015:
                confidence = float(clamp(1.0 - min(1.0, contour_score), 0.0, 1.0))
                return SymbolResult(
                    enabled=self.enabled,
                    label=contour_label,
                    confidence=confidence,
                    detected_shape=contour_label,
                    shape_confidence=confidence,
                    action_label=contour_label,
                    bbox=(x1, y1, x2 - x1, y2 - y1),
                    accepted=True,
                    reason="Contour template match.",
                )

        direct_label, direct_score = self.detector._direct_template_match(crop)
        if direct_label in {"LEFT", "RIGHT", "STOP", "QR"}:
            if direct_label in {"LEFT", "RIGHT"} and direct_score <= 0.16 and self._arrow_direction_ok(crop, direct_label):
                confidence = float(clamp(1.0 - direct_score, 0.0, 1.0))
                return SymbolResult(
                    enabled=self.enabled,
                    label=direct_label,
                    confidence=confidence,
                    detected_shape=direct_label,
                    shape_confidence=confidence,
                    action_label=direct_label,
                    bbox=(x1, y1, x2 - x1, y2 - y1),
                    accepted=True,
                    reason="Direct template match to arrow.",
                )

            if direct_label == "QR" and direct_score <= 0.18:
                confidence = float(clamp(1.0 - direct_score, 0.0, 1.0))
                return SymbolResult(
                    enabled=self.enabled,
                    label="QR",
                    confidence=confidence,
                    detected_shape="QR",
                    shape_confidence=confidence,
                    action_label="QR",
                    bbox=(x1, y1, x2 - x1, y2 - y1),
                    accepted=True,
                    reason="Direct template match to QR.",
                )

            if direct_label == "STOP" and direct_score <= 0.23:
                confidence = float(clamp(1.0 - direct_score, 0.0, 1.0))
                return SymbolResult(
                    enabled=self.enabled,
                    label="STOP",
                    confidence=confidence,
                    detected_shape="STOP",
                    shape_confidence=confidence,
                    action_label="STOP",
                    bbox=(x1, y1, x2 - x1, y2 - y1),
                    accepted=True,
                    reason="Direct template match.",
                )

        results, _edges = self.detector.detect(crop)
        if not results:
            return SymbolResult(enabled=self.enabled, bbox=(x1, y1, x2 - x1, y2 - y1), accepted=False, reason="No template match.")

        best = results[0]
        label = best["label"]
        score = float(best.get("score", 0.0))
        action_label = _canonical_label(label)
        confidence = float(clamp(1.0 - score, 0.0, 1.0))

        if label == "QR":
            confidence = 0.98
            return SymbolResult(
                enabled=self.enabled,
                label="QR",
                confidence=confidence,
                detected_shape="QR",
                shape_confidence=confidence,
                action_label="QR",
                bbox=(x1, y1, x2 - x1, y2 - y1),
                accepted=True,
                reason="QR detected by OpenCV.",
            )

        if label == "FINGERPRINT":
            good_matches = max(0.0, float(-score))
            confidence = float(clamp(good_matches / 20.0, 0.0, 1.0))
            return SymbolResult(
                enabled=self.enabled,
                label="FINGERPRINT",
                confidence=confidence,
                detected_shape="FINGERPRINT",
                shape_confidence=confidence,
                action_label="FINGERPRINT",
                bbox=(x1, y1, x2 - x1, y2 - y1),
                accepted=True,
                reason="Fingerprint detected by AKAZE.",
            )

        if label in {"LEFT", "RIGHT"} and not self._arrow_direction_ok(crop, label):
            return SymbolResult(
                enabled=self.enabled,
                bbox=(x1, y1, x2 - x1, y2 - y1),
                accepted=False,
                reason="Arrow direction check failed.",
            )

        return SymbolResult(
            enabled=self.enabled,
            label=label,
            confidence=confidence,
            detected_shape=label,
            shape_confidence=confidence,
            action_label=action_label,
            bbox=(x1, y1, x2 - x1, y2 - y1),
            accepted=True,
            reason="Template match accepted.",
        )

    def probe_fingerprint(self, frame_bgr: np.ndarray) -> SymbolResult:
        frame_h, frame_w = frame_bgr.shape[:2]
        rois: list[tuple[int, int, int, int]] = [
            (0, int(frame_h * 0.30), frame_w, max(1, frame_h - int(frame_h * 0.30))),
            (int(frame_w * 0.10), int(frame_h * 0.35), int(frame_w * 0.80), max(1, int(frame_h * 0.65))),
        ]

        best_score: float | None = None
        best_bbox: tuple[int, int, int, int] | None = None
        for x, y, w, h in rois:
            if w < 16 or h < 16:
                continue
            crop = frame_bgr[y : y + h, x : x + w]
            score = self.detector._match_fingerprint(crop)
            if score is None:
                continue
            if best_score is None or score < best_score:
                best_score = float(score)
                best_bbox = (x, y, w, h)

        if best_score is None or best_bbox is None:
            return SymbolResult(enabled=self.enabled, accepted=False, reason="No fingerprint probe match.")

        good_matches = max(0.0, float(-best_score))
        if good_matches < 8.0:
            return SymbolResult(enabled=self.enabled, accepted=False, reason="Fingerprint probe below match threshold.")

        confidence = float(clamp(good_matches / 20.0, 0.0, 1.0))
        return SymbolResult(
            enabled=self.enabled,
            label="FINGERPRINT",
            confidence=confidence,
            detected_shape="FINGERPRINT",
            shape_confidence=confidence,
            action_label="FINGERPRINT",
            bbox=best_bbox,
            accepted=True,
            reason="Fingerprint probe matched by AKAZE.",
        )

    def probe_symbol(self, frame_bgr: np.ndarray) -> SymbolResult:
        frame_h, frame_w = frame_bgr.shape[:2]
        rois: list[tuple[int, int, int, int]] = [
            (int(frame_w * 0.08), int(frame_h * 0.10), int(frame_w * 0.84), int(frame_h * 0.62)),
            (0, int(frame_h * 0.14), frame_w, max(1, int(frame_h * 0.62))),
        ]

        best_label: str | None = None
        best_score = float("inf")
        best_bbox: tuple[int, int, int, int] | None = None

        for x, y, w, h in rois:
            if w < 20 or h < 20:
                continue
            crop = frame_bgr[y : y + h, x : x + w]
            if self._is_line_like_region(crop):
                continue

            contour_label, contour_score, contour_margin = self.detector._contour_template_match(crop)
            rotated_qr_label, rotated_qr_score = self.detector._rotated_template_match(crop, {"QR"})
            if rotated_qr_label == "QR" and rotated_qr_score <= 0.26 and contour_label == "QR":
                confidence = float(clamp(1.0 - rotated_qr_score, 0.0, 1.0))
                return SymbolResult(
                    enabled=self.enabled,
                    label="QR",
                    confidence=confidence,
                    detected_shape="QR",
                    shape_confidence=confidence,
                    action_label="QR",
                    bbox=(x, y, w, h),
                    accepted=True,
                    reason="Symbol probe rotated QR match.",
                )

            rotated_stop_label, rotated_stop_score = self.detector._rotated_template_match(crop, {"STOP"})
            if (
                rotated_stop_label == "STOP"
                and rotated_stop_score <= 0.24
                and (contour_label != "RECYCLE" or contour_score > 0.35)
                and not self._looks_like_arrow(crop)
            ):
                confidence = float(clamp(1.0 - rotated_stop_score, 0.0, 1.0))
                return SymbolResult(
                    enabled=self.enabled,
                    label="STOP",
                    confidence=confidence,
                    detected_shape="STOP",
                    shape_confidence=confidence,
                    action_label="STOP",
                    bbox=(x, y, w, h),
                    accepted=True,
                    reason="Symbol probe rotated STOP match.",
                )

            if contour_label in {"RECYCLE", "STOP"}:
                contour_limits = {"RECYCLE": 0.46, "STOP": 0.45}
                if contour_score <= contour_limits[contour_label] and contour_margin >= 0.01:
                    confidence = float(clamp(1.0 - min(1.0, contour_score), 0.0, 1.0))
                    return SymbolResult(
                        enabled=self.enabled,
                        label=contour_label,
                        confidence=confidence,
                        detected_shape=contour_label,
                        shape_confidence=confidence,
                        action_label=contour_label,
                        bbox=(x, y, w, h),
                        accepted=True,
                        reason="Symbol probe contour match.",
                    )

            direct_label, direct_score = self.detector._direct_template_match(crop)
            if direct_label in {"LEFT", "RIGHT"} and direct_score <= 0.16 and self._arrow_direction_ok(crop, direct_label):
                confidence = float(clamp(1.0 - direct_score, 0.0, 1.0))
                return SymbolResult(
                    enabled=self.enabled,
                    label=direct_label,
                    confidence=confidence,
                    detected_shape=direct_label,
                    shape_confidence=confidence,
                    action_label=direct_label,
                    bbox=(x, y, w, h),
                    accepted=True,
                    reason="Symbol probe direct arrow match.",
                )

            fp_score = self.detector._match_fingerprint(crop)
            if fp_score is not None:
                good_matches = max(0.0, float(-fp_score))
                confidence = float(clamp(good_matches / 20.0, 0.0, 1.0))
                return SymbolResult(
                    enabled=self.enabled,
                    label="FINGERPRINT",
                    confidence=confidence,
                    detected_shape="FINGERPRINT",
                    shape_confidence=confidence,
                    action_label="FINGERPRINT",
                    bbox=(x, y, w, h),
                    accepted=True,
                    reason="Symbol probe matched fingerprint.",
                )

            results, _ = self.detector.detect(crop)
            if not results:
                continue

            candidate = results[0]
            label = candidate.get("label")
            if label not in {"LEFT", "RIGHT", "RECYCLE", "STOP", "QR", "FINGERPRINT"}:
                continue
            score = float(candidate.get("score", 1.0))
            if score < best_score:
                best_score = score
                best_label = label
                best_bbox = (x, y, w, h)

        if best_label is None or best_bbox is None:
            return SymbolResult(enabled=self.enabled, accepted=False, reason="No symbol probe match.")

        x, y, w, h = best_bbox
        best_crop = frame_bgr[y : y + h, x : x + w]

        if best_label in {"LEFT", "RIGHT"}:
            if not self._arrow_direction_ok(best_crop, best_label):
                return SymbolResult(enabled=self.enabled, accepted=False, reason="Arrow probe direction check failed.")

        # Prevent dark/glare arrow crops from falling through as recycle.
        if best_label == "RECYCLE" and self._looks_like_arrow(best_crop):
            return SymbolResult(enabled=self.enabled, accepted=False, reason="Recycle fallback vetoed by arrow geometry.")

        if best_label == "QR":
            conf = 0.98
        elif best_label == "FINGERPRINT":
            conf = float(clamp(max(0.0, -best_score) / 20.0, 0.0, 1.0))
        else:
            conf = float(clamp(1.0 - best_score, 0.0, 1.0))

        return SymbolResult(
            enabled=self.enabled,
            label=best_label,
            confidence=conf,
            detected_shape=best_label,
            shape_confidence=conf,
            action_label=best_label,
            bbox=best_bbox,
            accepted=True,
            reason="Symbol probe fallback match.",
        )

    def detect(self, frame_bgr: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None, debug: bool = False):
        return self.detector.detect(frame_bgr, roi=roi, debug=debug)


def draw_symbol_debug(frame_bgr: np.ndarray, candidate: SymbolCandidate, result: SymbolResult) -> None:
    if candidate.roi_bbox is not None:
        x, y, w, h = candidate.roi_bbox
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (255, 200, 0), 1)

    if candidate.bbox is not None:
        x, y, w, h = candidate.bbox
        color = (0, 255, 0) if result.accepted else (0, 165, 255)
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)
        symbol_text = f"S:{result.label or '-'} {result.confidence:.2f}"
        cv2.putText(
            frame_bgr,
            symbol_text,
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
        )


SymbolDetector = TemplateSymbolDetector