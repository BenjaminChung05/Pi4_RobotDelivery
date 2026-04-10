"""
Symbol detector for Final robot controller.

Based on Week2's proven multi-stage detection pipeline with additions:
- Canonical label mapping (BUTTON/HAZARD/OCTAGON → STOP, arrows → LEFT/RIGHT)
- Fast-filter + classify API for integration with the main loop
- AKAZE feature matching for BUTTON, HAZARD, FINGERPRINT (rotation-tolerant)
- Triangle-counting heuristic for RECYCLE
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from utils import clamp

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical label mapping
# ---------------------------------------------------------------------------
ACTIVE_SYMBOL_LABELS = {"STOP", "LEFT", "RIGHT", "RECYCLE", "QR", "FINGERPRINT"}

_CANONICAL = {
    "STOP_SIGN": "STOP", "OCTAGON": "STOP", "BUTTON": "STOP", "HAZARD": "STOP",
    "STOP": "STOP",
    "ARROW_LEFT": "LEFT", "LEFT_ARROW": "LEFT", "LEFTTURN": "LEFT", "LEFT": "LEFT",
    "ARROW_RIGHT": "RIGHT", "RIGHT_ARROW": "RIGHT", "RIGHTTURN": "RIGHT", "RIGHT": "RIGHT",
    "FINGERPRINT": "FINGERPRINT", "RECYCLE": "RECYCLE", "QR": "QR",
}


def _canonical_label(label: str) -> str:
    token = label.strip().upper().replace("-", "_").replace(" ", "_")
    return _CANONICAL.get(token, token)


# ---------------------------------------------------------------------------
# Preprocessing (identical to Week2)
# ---------------------------------------------------------------------------
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
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, c_val,
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


# ---------------------------------------------------------------------------
# Arrow / defect helpers
# ---------------------------------------------------------------------------
def is_arrow(cnt: np.ndarray, defect_depth: int = 1000) -> bool:
    peri = cv2.arcLength(cnt, True)
    if peri <= 1e-6:
        return False
    approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)
    if approx is None or len(approx) < 6 or len(approx) > 14:
        return False
    try:
        hull_idx = cv2.convexHull(approx, returnPoints=False)
        if hull_idx is None or len(hull_idx) < 3:
            return False
        defects = cv2.convexityDefects(approx, hull_idx)
        if defects is None:
            return False
        # Use min so small contours get a lenient threshold
        adaptive_depth = min(int(0.012 * peri * 256.0), int(defect_depth))
        meaningful = []
        for i in range(defects.shape[0]):
            _, _, _, depth = defects[i, 0]
            if depth > adaptive_depth:
                meaningful.append(int(depth))
        if len(meaningful) != 2:
            return False
        # Both defects must be of comparable depth. True arrows have two
        # tail-fin defects of similar depth.  Non-arrows (e.g. Button) may
        # have one deep + one barely-above-threshold defect.
        ratio = min(meaningful) / max(1, max(meaningful))
        return ratio > 0.30
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
        tail_ref = np.array(
            [float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"])],
            dtype=np.float32,
        )
    d = np.sum((pts - tail_ref) ** 2, axis=1)
    tip = pts[int(np.argmax(d))]
    dx = float(tail_ref[0] - tip[0])
    dy = float(tail_ref[1] - tip[1])
    if abs(dx) > abs(dy):
        return "RIGHT" if dx > 0 else "LEFT"
    return "DOWN" if dy > 0 else "UP"


# ---------------------------------------------------------------------------
# Template data classes
# ---------------------------------------------------------------------------
@dataclass
class ShapeTemplate:
    label: str
    contour: np.ndarray
    verts: int
    extent: float
    solidity: float
    circularity: float
    bin_img: np.ndarray


@dataclass
class FeatureTemplate:
    label: str
    kp: list
    des: np.ndarray


# ---------------------------------------------------------------------------
# Template library
# ---------------------------------------------------------------------------
class TemplateLibrary:
    def __init__(self, templates_dir: str) -> None:
        self.templates_dir = templates_dir
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

    def load(self) -> "TemplateLibrary":
        if not os.path.isdir(self.templates_dir):
            raise RuntimeError(f"Templates dir not found: {self.templates_dir}")

        files = [
            f for f in os.listdir(self.templates_dir)
            if f.lower().endswith((".png", ".jpg")) and not f.startswith("._")
        ]
        by_base: dict[str, str] = {}
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
            raw_label = os.path.splitext(fn)[0].strip().upper()
            label = _canonical_label(raw_label)
            if label not in ACTIVE_SYMBOL_LABELS:
                continue
            path = os.path.join(self.templates_dir, fn)
            img = cv2.imread(path)
            if img is None:
                continue

            # AKAZE feature templates for shapes that need rotation tolerance
            # Week2 loaded these for BUTTON, HAZARD, FINGERPRINT — critical for detection
            if raw_label in {"FINGERPRINT", "BUTTON", "HAZARD"}:
                gray = preprocess_gray(img)
                kp, des = akaze.detectAndCompute(gray, None)
                if des is not None and len(kp) >= 16:
                    self.feature_templates.append(
                        FeatureTemplate(raw_label, kp, des)
                    )
                if raw_label == "FINGERPRINT":
                    continue  # fingerprint is feature-only

            th = preprocess_bin(img)
            cnt = largest_contour(th)
            if cnt is None or cv2.contourArea(cnt) < 200:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            roi = th[y : y + h, x : x + w]
            roi = _normalize_bin(roi)
            roi = cv2.resize(roi, (50, 50), interpolation=cv2.INTER_NEAREST)
            verts, extent, solidity, circularity = self._shape_descriptor(cnt)
            self.shape_templates.append(
                ShapeTemplate(label, cnt, verts, extent, solidity, circularity, roi)
            )

        if not self.shape_templates and not self.feature_templates:
            raise RuntimeError("No templates loaded successfully.")
        return self


# ---------------------------------------------------------------------------
# Core detector — Week2 multi-stage pipeline
# ---------------------------------------------------------------------------
class TemplateSymbolDetector:
    """
    Multi-stage symbol detector from Week2.

    Stages:
      0.  QR code detection (OpenCV)
      0b. Circle-like quick-accept for STOP (Button)
      0c. Octagon vs Hazard discrimination
      1.  AKAZE feature matching for HAZARD, BUTTON, FINGERPRINT
      1d. Triangle-count heuristic for RECYCLE
      2.  Arrow binary-template matching
      3.  Non-arrow binary IoU matching
      4.  Silhouette contour matching (matchShapes + geometry penalty)
      5.  Geometry-only arrow fallback
    """

    def __init__(
        self,
        lib: TemplateLibrary,
        min_area: int = 1100,
        match_thresh: float = 0.20,
        cross_shape_thresh: float = 0.72,
        fp_min_good: int = 12,
        fp_ratio: float = 0.75,
    ) -> None:
        self.shape_templates = lib.shape_templates
        self.feature_templates = lib.feature_templates
        self.arrow_templates = [
            t for t in self.shape_templates if t.label in {"LEFT", "RIGHT"}
        ]
        self.non_arrow_templates = [
            t for t in self.shape_templates if t.label not in {"LEFT", "RIGHT"}
        ]

        self.min_area = int(min_area)
        self.match_thresh = float(match_thresh)
        self.cross_shape_thresh = float(cross_shape_thresh)
        self.arrow_defect_depth = 1000

        self.fp_min_good = int(fp_min_good)
        self.fp_ratio = float(fp_ratio)
        self.button_min_good = 5
        self.hazard_min_good = 5
        self.feature_ratio = 0.75

        self.akaze = cv2.AKAZE_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.qr = cv2.QRCodeDetector()

        self.bin_match_thresh = 0.45
        self.arrow_bin_thresh = 0.40
        self.arrow_bin_strict = 0.50

        self.button_template = next(
            (t for t in self.shape_templates if t.label == "STOP"), None
        )

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
    def _descriptor_penalty(
        cand_desc: Tuple[int, float, float, float], templ: ShapeTemplate,
    ) -> float:
        c_verts, c_extent, c_solidity, c_circularity = cand_desc
        p_verts = 0.02 * abs(float(c_verts - templ.verts))
        p_extent = 0.35 * abs(c_extent - templ.extent)
        p_solidity = 0.25 * abs(c_solidity - templ.solidity)
        p_circularity = 0.20 * abs(c_circularity - templ.circularity)
        return float(p_verts + p_extent + p_solidity + p_circularity)

    @staticmethod
    def _count_triangle_contours(image_bgr: np.ndarray) -> int:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        _, mask_norm = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), dtype=np.uint8)
        # Minimum triangle area = 2% of image area to reject noise
        img_area = float(image_bgr.shape[0] * image_bgr.shape[1])
        min_tri_area = max(40.0, img_area * 0.02)
        triangle_count = 0
        seen: list[tuple[int, int, int, int]] = []
        for mask in (mask_inv, mask_norm):
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < min_tri_area:
                    continue
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * peri, True) if peri > 1e-6 else contour
                if len(approx) != 3:
                    continue
                x, y, w, h = cv2.boundingRect(approx)
                if w < 8 or h < 8:
                    continue
                if any(abs(x - bx) < 4 and abs(y - by) < 4 for bx, by, _, _ in seen):
                    continue
                seen.append((x, y, w, h))
                triangle_count += 1
        return triangle_count

    def detect(
        self,
        frame_bgr: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> list[dict]:
        """Week2-style multi-stage detection. Returns list of {label, bbox, score}."""
        x_off = y_off = 0
        work = frame_bgr
        if roi is not None:
            x1, y1, x2, y2 = roi
            work = frame_bgr[y1:y2, x1:x2]
            x_off, y_off = x1, y1

        wh, ww = work.shape[:2]
        results: list[dict] = []

        # ================================================================
        # WHOLE-IMAGE stages (run BEFORE per-contour, catches fragmented
        # symbols like QR, Fingerprint, Button, Recycle)
        # ================================================================

        # ---- Stage W0: QR detection on whole image ----
        try:
            qr_ok, points = self.qr.detect(work)
        except cv2.error:
            qr_ok, points = False, None
        if qr_ok and points is not None and len(points) > 0:
            pts = points[0].astype(int)
            x, y, w, h = cv2.boundingRect(pts)
            if w * h >= 200:
                _log.debug("W0-QR: detected QR code")
                results.append({
                    "label": "QR", "bbox": (x + x_off, y + y_off, w, h),
                    "score": -1.0,
                })

        # ---- Stage W1: AKAZE feature matching on whole image ----
        # This catches FINGERPRINT, BUTTON, HAZARD even when their contour
        # is fragmented or below min_area.
        # Check ALL three types and pick the best match (most good features).
        if self.feature_templates:
            gray_w = preprocess_gray(work)
            kp_w, des_w = self.akaze.detectAndCompute(gray_w, None)
            if des_w is not None and len(kp_w) >= 5:
                best_feat_label = None
                best_feat_good = 0
                best_feat_min = 0
                for feat_label in ("FINGERPRINT", "BUTTON", "HAZARD"):
                    feats = [t for t in self.feature_templates if t.label == feat_label]
                    if not feats:
                        continue
                    ratio = self.fp_ratio if feat_label == "FINGERPRINT" else self.feature_ratio
                    min_g = 6 if feat_label == "FINGERPRINT" else (2 if feat_label == "BUTTON" else 3)
                    for t in feats:
                        knn = self.bf.knnMatch(des_w, t.des, k=2)
                        good = sum(1 for m, n in knn if m.distance < ratio * n.distance)
                        if good >= min_g and good > best_feat_good:
                            best_feat_good = good
                            best_feat_label = feat_label
                            best_feat_min = min_g
                if best_feat_label is not None:
                    _log.debug("W1-AKAZE: best=%s good=%d", best_feat_label, best_feat_good)
                    # Guard: check dominant contour shape to avoid false
                    # FINGERPRINT on round/solid shapes like stop/octagon.
                    skip_feat = False
                    if best_feat_label == "FINGERPRINT":
                        # Require minimum edge density — fingerprints
                        # are uniquely dense in edges.
                        fp_edges = preprocess_edges(work)
                        fp_edensity = float(np.count_nonzero(fp_edges)) / max(1.0, float(fp_edges.size))
                        if fp_edensity < 0.14:
                            skip_feat = True
                        feat_bin = preprocess_bin(work)
                        feat_cnt = largest_contour(feat_bin)
                        if feat_cnt is not None and cv2.contourArea(feat_cnt) > 200:
                            feat_peri = cv2.arcLength(feat_cnt, True)
                            feat_area = float(cv2.contourArea(feat_cnt))
                            feat_circ = (4.0 * np.pi * feat_area) / max(1e-6, feat_peri * feat_peri)
                            feat_hull = float(cv2.contourArea(cv2.convexHull(feat_cnt)))
                            feat_sol = feat_area / max(1e-6, feat_hull)
                            # Highly round/solid shapes are stop signs, not fingerprints
                            if feat_circ > 0.70 or feat_sol > 0.90:
                                skip_feat = True
                    if not skip_feat:
                        out_label = "FINGERPRINT" if best_feat_label == "FINGERPRINT" else "STOP"
                        _log.debug("W1-AKAZE: accepted → %s (from %s, good=%d)", out_label, best_feat_label, best_feat_good)
                        results.append({
                            "label": out_label,
                            "bbox": (x_off, y_off, ww, wh),
                            "score": float(-best_feat_good),
                        })

        # ---- Stage W2: RECYCLE triangle count on whole image ----
        # Guard: skip if the dominant contour looks like an arrow or has
        # high IoU with an arrow template.
        # Also guard against QR codes which have many angular sub-shapes
        # that the triangle counter can mistake for RECYCLE triangles.
        if not results:
            # Quick QR-finder-pattern check: QR codes contain 3 concentric
            # square finder patterns. Detect nested squares to reject.
            _has_qr_features = False
            gray_qr = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
            _, mask_qr = cv2.threshold(gray_qr, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            _qr_cnts, _qr_hier = cv2.findContours(mask_qr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if _qr_hier is not None:
                _nested_sq = 0
                for _qi, _qh in enumerate(_qr_hier[0]):
                    if _qh[2] < 0:  # no child
                        continue
                    _qc = _qr_cnts[_qi]
                    _q_peri = cv2.arcLength(_qc, True)
                    _q_app = cv2.approxPolyDP(_qc, 0.04 * _q_peri, True) if _q_peri > 1 else _qc
                    if 4 <= len(_q_app) <= 6:
                        # Check if the child is also square-ish
                        _child_idx = _qh[2]
                        _cc = _qr_cnts[_child_idx]
                        _c_peri = cv2.arcLength(_cc, True)
                        _c_app = cv2.approxPolyDP(_cc, 0.04 * _c_peri, True) if _c_peri > 1 else _cc
                        if 4 <= len(_c_app) <= 6:
                            _nested_sq += 1
                if _nested_sq >= 2:
                    _has_qr_features = True

            has_arrow_blob = False
            bin_w = preprocess_bin(work)
            cnt_w = largest_contour(bin_w)
            if cnt_w is not None and cv2.contourArea(cnt_w) > 300:
                if is_arrow(cnt_w, defect_depth=800):
                    has_arrow_blob = True
                else:
                    # Check binary IoU with arrow templates.
                    # If BOTH LEFT and RIGHT have similar IoU (within 0.08),
                    # the blob is symmetric (likely Recycle), not directional.
                    cand_bin_w = _contour_roi_bin(bin_w, cnt_w)
                    if cand_bin_w is not None:
                        arrow_ious = {}
                        for at in self.arrow_templates:
                            iou = _bin_iou(cand_bin_w, at.bin_img)
                            arrow_ious[at.label] = max(arrow_ious.get(at.label, 0.0), iou)
                        l_iou = arrow_ious.get("LEFT", 0.0)
                        r_iou = arrow_ious.get("RIGHT", 0.0)
                        max_iou = max(l_iou, r_iou)
                        if max_iou >= 0.20 and abs(l_iou - r_iou) > 0.08:
                            has_arrow_blob = True
            if not has_arrow_blob:
                # Additional guard: check dominant contour solidity.
                # Recycle has low solidity (< 0.80). Round/solid shapes
                # like stop sign or octagon should not trigger RECYCLE.
                recycle_shape_ok = True
                if cnt_w is not None and cv2.contourArea(cnt_w) > 200:
                    rh = float(cv2.contourArea(cv2.convexHull(cnt_w)))
                    r_sol = float(cv2.contourArea(cnt_w)) / max(1e-6, rh)
                    if r_sol > 0.85:
                        recycle_shape_ok = False
                tri = self._count_triangle_contours(work)
                if tri >= 2 and recycle_shape_ok and not _has_qr_features:
                    _log.debug("W2-RECYCLE: tri=%d sol=%.2f", tri, r_sol if cnt_w is not None else -1)
                    results.append({
                        "label": "RECYCLE",
                        "bbox": (x_off, y_off, ww, wh),
                        "score": 0.08,
                    })

        # If whole-image stages found something, return early
        if results:
            results.sort(key=lambda r: r["score"])
            _log.info("detect → %s (whole-image stage, score=%.3f)", results[0]["label"], results[0]["score"])
            return results

        # ================================================================
        # PER-CONTOUR stages (for well-defined single-blob symbols:
        # arrows, octagon, hazard sign, stop sign)
        # ================================================================
        bin_img = preprocess_bin(work)
        cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            if cv2.contourArea(c) < 400:
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

            c_verts, c_extent, c_solidity, c_circularity = cand_desc
            defect_count = convexity_defect_count(cnt)

            # ---- Stage 0a: Circle/round quick-accept for STOP (Button) ----
            # Run before fingerprint check to prevent round symbols with
            # moderate edge density from being misclassified as FINGERPRINT.
            # Also accept via high Button-template IoU even with lower
            # circularity (Button at small scale can have circ ~0.40).
            if self.button_template is not None:
                btn_iou = _bin_iou(cand_bin, self.button_template.bin_img)
            else:
                btn_iou = 0.0
            if (
                self.button_template is not None
                and c_solidity > 0.65
                and c_extent > 0.45
                and (
                    (c_circularity > 0.38 and c_verts >= 5 and btn_iou >= 0.22)
                    or btn_iou >= 0.40
                )
            ):
                results.append({
                    "label": "STOP",
                    "bbox": (x + x_off, y + y_off, w, h),
                    "score": float(1.0 - btn_iou),
                })
                continue

            # ---- Stage 0b: Fingerprint edge-density check ----
            # Fingerprints have uniquely HIGH edge density combined with
            # LOW circularity AND HIGH solidity (> 0.82).  Recycle also has
            # high edge density + low circularity but much LOWER solidity
            # (< 0.75), so the solidity guard excludes it.
            edges_cand = preprocess_edges(cand)
            cand_edge_density = float(np.count_nonzero(edges_cand)) / float(max(1, edges_cand.size))
            cand_aspect = float(w) / max(1.0, float(h))
            if (
                cand_edge_density > 0.18
                and c_circularity < 0.70
                and c_solidity > 0.78
                and c_solidity < 0.95
                and c_extent > 0.45
                and 0.5 < cand_aspect < 2.0
            ):
                results.append({
                    "label": "FINGERPRINT",
                    "bbox": (x + x_off, y + y_off, w, h),
                    "score": float(0.10 - cand_edge_density),
                })
                continue

            # ---- Stage 0c: Octagon/Hazard discrimination ----
            if (
                7 <= c_verts <= 10
                and c_solidity > 0.97
                and c_circularity > 0.85
                and c_extent > 0.70
            ):
                best_label = None
                best_iou = 0.0
                best_score = 1e9
                for t in self.non_arrow_templates:
                    if t.label != "STOP":
                        continue
                    iou = _bin_iou(cand_bin, t.bin_img)
                    score = (1.0 - iou) + 0.35 * abs(c_circularity - t.circularity)
                    if score < best_score:
                        best_score = score
                        best_label = t.label
                        best_iou = iou
                if best_label is not None and best_iou >= 0.22:
                    results.append({
                        "label": "STOP",
                        "bbox": (x + x_off, y + y_off, w, h),
                        "score": float(1.0 - best_iou),
                    })
                    continue

            # ---- Stage 2: Arrow by binary template matching ----
            is_quad_like = c_solidity > 0.86 and c_extent > 0.50 and c_verts <= 8
            if self.arrow_templates and not is_quad_like:
                best_arrow_iou = 0.0
                best_arrow_label = None
                for t in self.arrow_templates:
                    iou = _bin_iou(cand_bin, t.bin_img)
                    if iou > best_arrow_iou:
                        best_arrow_iou = iou
                        best_arrow_label = t.label
                if best_arrow_iou >= self.arrow_bin_thresh:
                    is_arrow_like = (
                        is_arrow(cnt, defect_depth=self.arrow_defect_depth)
                        and c_solidity < 0.85
                        and c_verts >= 7
                    )
                    # Require BOTH high IoU AND geometry confirmation
                    if best_arrow_iou >= self.arrow_bin_strict and is_arrow_like:
                        # Determine direction from mass distribution, not
                        # template label (L/R templates are near-mirrors).
                        arrow_label = self._mass_arrow_direction(th, cnt)
                        results.append({
                            "label": arrow_label,
                            "bbox": (x + x_off, y + y_off, w, h),
                            "score": float(1.0 - best_arrow_iou),
                        })
                        continue

            # ---- Stage 3: Non-arrow binary IoU ----
            # Exclude RECYCLE from per-contour matching — it should only
            # come from the whole-image triangle-counting stage (W2).
            bin_scores: dict[str, float] = {}
            for t in self.non_arrow_templates:
                if t.label == "RECYCLE":
                    continue
                iou = _bin_iou(cand_bin, t.bin_img)
                prev = bin_scores.get(t.label, 0.0)
                if iou > prev:
                    bin_scores[t.label] = iou
            if bin_scores:
                best_bin_label = max(bin_scores, key=lambda k: bin_scores[k])
                best_bin_iou = float(bin_scores[best_bin_label])
                if best_bin_iou >= self.bin_match_thresh:
                    results.append({
                        "label": best_bin_label,
                        "bbox": (x + x_off, y + y_off, w, h),
                        "score": float(1.0 - best_bin_iou),
                    })
                    continue

            # ---- Stage 4: Silhouette contour matching ----
            best_label = None
            best_score = 1e9
            label_scores: dict[str, float] = {}
            for t in self.non_arrow_templates:
                if t.label == "RECYCLE":
                    continue
                s = cv2.matchShapes(cnt, t.contour, cv2.CONTOURS_MATCH_I1, 0.0)
                s += self._descriptor_penalty(cand_desc, t)
                prev = label_scores.get(t.label)
                if prev is None or s < prev:
                    label_scores[t.label] = s
            for label, s in label_scores.items():
                if s < best_score:
                    best_score = s
                    best_label = label
            if best_label is not None and best_score <= self.match_thresh:
                results.append({
                    "label": best_label,
                    "bbox": (x + x_off, y + y_off, w, h),
                    "score": float(best_score),
                })
                continue

            # ---- Stage 5: Geometry-only arrow fallback ----
            # Guard: only use arrow fallback if no non-arrow template has
            # reasonable IoU — prevents Button/other complex shapes from
            # being misclassified as arrows.
            best_nonarrow_iou = max(
                (_bin_iou(cand_bin, t.bin_img) for t in self.non_arrow_templates),
                default=0.0,
            )
            if (
                not is_quad_like
                and c_solidity < 0.82
                and c_verts >= 7
                and best_nonarrow_iou < 0.20
                and is_arrow(cnt, defect_depth=self.arrow_defect_depth)
            ):
                d = self._mass_arrow_direction(th, cnt)
                results.append({
                    "label": d,
                    "bbox": (x + x_off, y + y_off, w, h),
                    "score": 0.10,
                })
                continue

        results.sort(key=lambda r: r["score"])
        if results:
            _log.info("detect → %s (per-contour, score=%.3f)", results[0]["label"], results[0]["score"])
        return results

    @staticmethod
    def _mass_arrow_direction(bin_img: np.ndarray, cnt: np.ndarray) -> str:
        """Determine arrow direction from pixel-mass distribution.

        Primary: column-sum asymmetry of the binary crop.
          A LEFT arrow has its body/tail on the RIGHT half → more pixels there.
          A RIGHT arrow has its body/tail on the LEFT half → more pixels there.
        Fallback: farthest-from-centroid point (tail heuristic).
        """
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            return "LEFT"
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])

        # --- Primary: binary column-sum asymmetry ---
        if bin_img is not None and bin_img.size > 0:
            h, w = bin_img.shape[:2]
            mid = w // 2
            if mid > 0:
                left_sum = float(np.count_nonzero(bin_img[:, :mid]))
                right_sum = float(np.count_nonzero(bin_img[:, mid:]))
                total = left_sum + right_sum
                if total > 0 and abs(left_sum - right_sum) / total > 0.06:
                    # More pixels on right → body/tail right → arrow points LEFT
                    return "LEFT" if right_sum > left_sum else "RIGHT"

        # --- Fallback: farthest-from-centroid point (tail heuristic) ---
        pts = cnt.reshape(-1, 2).astype(np.float32)
        dists = np.sum((pts - np.array([cx, cy])) ** 2, axis=1)
        tail = pts[int(np.argmax(dists))]
        dx = float(tail[0] - cx)
        if abs(dx) >= abs(float(tail[1] - cy)) * 0.5:
            return "LEFT" if dx > 0 else "RIGHT"
        return "LEFT"

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
        fp_templates = [t for t in self.feature_templates if t.label == "FINGERPRINT"]
        templates = fp_templates if fp_templates else self.feature_templates
        for t in templates:
            knn = self.bf.knnMatch(des, t.des, k=2)
            good = sum(1 for m, n in knn if m.distance < self.fp_ratio * n.distance)
            best_good = max(best_good, good)
        if best_good >= max(8, self.fp_min_good - 4):
            return float(-best_good)
        return None

    def _match_feature(
        self, cand_bgr: np.ndarray, label: str, min_good: int, ratio: float,
    ) -> Optional[float]:
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
            good = sum(1 for m, n in knn if m.distance < ratio * n.distance)
            best_good = max(best_good, good)
        if best_good >= min_good:
            return float(-best_good)
        return None


# ---------------------------------------------------------------------------
# Fast-filter config + wrapper data classes
# ---------------------------------------------------------------------------
@dataclass
class FastFilterConfig:
    roi_top_ratio: float = 0.02
    roi_bottom_ratio: float = 0.68   # was 0.62 — catch symbols when robot is closer
    roi_left_ratio: float = 0.02
    roi_right_ratio: float = 0.98
    min_area: float = 70.0
    padding: int = 14
    morph_kernel_size: int = 3
    min_bbox_size: int = 14
    max_bbox_aspect_ratio: float = 3.8
    min_fill_ratio: float = 0.12
    max_center_y_ratio: float = 0.72  # must be >= roi_bottom_ratio


@dataclass
class SymbolDetectorConfig:
    model_path: str = "models/symbol_classifier.tflite"
    labels_path: str = "models/labels.txt"
    templates_dir: str = "templates"
    input_size: tuple[int, int] = (224, 224)
    confidence_threshold: float = 0.60
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
    action_label: str | None = None
    bbox: tuple[int, int, int, int] | None = None
    accepted: bool = False
    reason: str | None = None


# ---------------------------------------------------------------------------
# TFLiteSymbolDetector — main API used by main.py
# ---------------------------------------------------------------------------
class TFLiteSymbolDetector:
    """
    Wrapper providing fast_filter → classify/probe API over the Week2 detector.
    """

    def __init__(self, config: SymbolDetectorConfig | None = None) -> None:
        self.config = config or SymbolDetectorConfig()
        self.enabled = True
        self.reason = "Template detector loaded."

        try:
            library = TemplateLibrary(self.config.templates_dir).load()
            self.detector = TemplateSymbolDetector(library)
        except Exception as exc:
            self.enabled = False
            self.reason = str(exc)
            self.detector = None

    # ----- fast filter: find candidate blob in frame -----
    def fast_filter(self, frame_bgr: np.ndarray) -> SymbolCandidate:
        cfg = self.config.fast_filter
        fh, fw = frame_bgr.shape[:2]
        x1 = int(fw * cfg.roi_left_ratio)
        y1 = int(fh * cfg.roi_top_ratio)
        x2 = int(fw * cfg.roi_right_ratio)
        y2 = int(fh * cfg.roi_bottom_ratio)
        roi = frame_bgr[y1:y2, x1:x2]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        _, mask_norm = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        ks = cfg.morph_kernel_size
        kernel = np.ones((ks, ks), dtype=np.uint8)
        for m in (mask_inv, mask_norm):
            cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, dst=m, iterations=1)
            cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, dst=m, iterations=2)

        roi_area = float((y2 - y1) * (x2 - x1))
        best_contour = None
        best_area = 0.0
        best_mask = mask_inv
        for mask in (mask_inv, mask_norm):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in sorted(contours, key=cv2.contourArea, reverse=True):
                area = float(cv2.contourArea(contour))
                if area < cfg.min_area or area > roi_area * 0.70:
                    continue
                bx, by, bw, bh = cv2.boundingRect(contour)
                # Only reject blobs touching the BOTTOM edge of the ROI
                # (those are almost certainly the track line extending down).
                # Top/left/right touching is fine — symbols can appear near
                # frame edges as the robot approaches them.
                if by + bh >= mask.shape[0] - 1:
                    continue
                if area > best_area:
                    best_area = area
                    best_contour = contour
                    best_mask = mask
                break

        roi_bbox = (x1, y1, x2 - x1, y2 - y1)
        if best_contour is None:
            return SymbolCandidate(found=False, roi_bbox=roi_bbox, mask=best_mask)

        bx, by, bw, bh = cv2.boundingRect(best_contour)
        if bw < cfg.min_bbox_size or bh < cfg.min_bbox_size:
            return SymbolCandidate(found=False, area=best_area, roi_bbox=roi_bbox)

        fill = best_area / float(max(1, bw * bh))
        if fill < cfg.min_fill_ratio:
            return SymbolCandidate(found=False, area=best_area, roi_bbox=roi_bbox)

        center_y = y1 + by + bh / 2.0
        if center_y > fh * cfg.max_center_y_ratio:
            return SymbolCandidate(found=False, area=best_area, roi_bbox=roi_bbox)

        aspect = max(bw, bh) / max(1, min(bw, bh))
        if aspect > cfg.max_bbox_aspect_ratio:
            return SymbolCandidate(found=False, area=best_area, roi_bbox=roi_bbox)

        pad = cfg.padding
        box_x = max(0, x1 + bx - pad)
        box_y = max(0, y1 + by - pad)
        box_w = min(fw - box_x, bw + 2 * pad)
        box_h = min(fh - box_y, bh + 2 * pad)
        return SymbolCandidate(
            found=True,
            bbox=(box_x, box_y, box_w, box_h),
            area=best_area,
            roi_bbox=roi_bbox,
            mask=best_mask,
        )

    # ----- classify: run full pipeline on candidate -----
    def classify(self, frame_bgr: np.ndarray, candidate: SymbolCandidate) -> SymbolResult:
        if not self.enabled or self.detector is None:
            return SymbolResult(enabled=False, reason="Detector not loaded.")
        if not candidate.found or candidate.bbox is None:
            return SymbolResult(enabled=True, accepted=False, reason="No candidate.")

        x, y, w, h = candidate.bbox
        fh, fw = frame_bgr.shape[:2]
        x1 = int(clamp(float(x), 0.0, float(max(0, fw - 1))))
        y1 = int(clamp(float(y), 0.0, float(max(0, fh - 1))))
        x2 = int(clamp(float(x + w), float(x1 + 1), float(fw)))
        y2 = int(clamp(float(y + h), float(y1 + 1), float(fh)))
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 16:
            return SymbolResult(enabled=True, bbox=candidate.bbox, reason="Crop too small.")

        # Reject line-like blobs
        if self._is_line_like(crop):
            return SymbolResult(enabled=True, bbox=candidate.bbox, accepted=False, reason="Looks like track line.")

        # Run the full Week2 detection pipeline on the crop
        results = self.detector.detect(crop)
        if not results:
            # Try expanded ROI
            pad_w = int(0.25 * (x2 - x1))
            pad_h = int(0.25 * (y2 - y1))
            ex1 = max(0, x1 - pad_w)
            ey1 = max(0, y1 - pad_h)
            ex2 = min(fw, x2 + pad_w)
            ey2 = min(fh, y2 + pad_h)
            expanded = frame_bgr[ey1:ey2, ex1:ex2]
            if expanded.size > 0 and expanded.shape[0] >= 20 and expanded.shape[1] >= 20:
                results = self.detector.detect(expanded)

        if not results:
            return SymbolResult(enabled=True, bbox=candidate.bbox, accepted=False, reason="No template match.")

        best = results[0]
        label = best["label"]
        score = float(best.get("score", 0.0))

        # Verify arrow direction
        if label in {"LEFT", "RIGHT"}:
            # Direction already determined by mass method in detect(),
            # so no secondary check needed here
            pass

        if label == "QR":
            confidence = 0.98
        elif label == "FINGERPRINT":
            confidence = float(clamp(max(0.0, -score) / 8.0 + 0.40, 0.0, 1.0))
        else:
            confidence = float(clamp(1.0 - score, 0.0, 1.0))

        return SymbolResult(
            enabled=True,
            label=label,
            confidence=confidence,
            action_label=label,
            bbox=(x1, y1, x2 - x1, y2 - y1),
            accepted=True,
            reason="Matched.",
        )

    # ----- probe: full-frame fallback when fast filter rejects -----
    def probe_symbol(self, frame_bgr: np.ndarray) -> SymbolResult:
        if not self.enabled or self.detector is None:
            return SymbolResult(enabled=False, reason="Detector not loaded.")

        fh, fw = frame_bgr.shape[:2]
        rois = [
            # Upper-middle zone where approaching symbols appear
            (int(fw * 0.02), int(fh * 0.02), int(fw * 0.96), int(fh * 0.55)),
            # Slightly wider fallback
            (0, int(fh * 0.05), fw, max(1, int(fh * 0.58))),
            # Closer-range zone — catches symbols when robot is nearly on top of them
            (int(fw * 0.05), int(fh * 0.02), int(fw * 0.90), max(1, int(fh * 0.68))),
        ]

        for rx, ry, rw, rh in rois:
            if rw < 20 or rh < 20:
                continue
            crop = frame_bgr[ry : ry + rh, rx : rx + rw]
            if self._is_line_like(crop):
                continue

            results = self.detector.detect(crop)
            if not results:
                continue

            best = results[0]
            label = best["label"]
            if label not in ACTIVE_SYMBOL_LABELS:
                continue

            score = float(best.get("score", 1.0))

            if label in {"LEFT", "RIGHT"} and not self._arrow_direction_ok(crop, label):
                continue

            if label == "QR":
                conf = 0.98
            elif label == "FINGERPRINT":
                conf = float(clamp(max(0.0, -score) / 8.0 + 0.40, 0.0, 1.0))
            else:
                conf = float(clamp(1.0 - score, 0.0, 1.0))

            return SymbolResult(
                enabled=True,
                label=label,
                confidence=conf,
                action_label=label,
                bbox=(rx, ry, rw, rh),
                accepted=True,
                reason="Probe matched.",
            )

        return SymbolResult(enabled=True, accepted=False, reason="No probe match.")

    # ----- helpers -----
    @staticmethod
    def _is_line_like(crop_bgr: np.ndarray) -> bool:
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
        fill = area / float(max(1, w * h))
        aspect = max(w, h) / float(max(1, min(w, h)))
        width_ratio = w / float(max(1, bin_img.shape[1]))
        height_ratio = h / float(max(1, bin_img.shape[0]))
        touches_bottom = (y + h) >= (bin_img.shape[0] - 2)
        touches_top = y <= 1
        touches_left = x <= 1
        touches_right = (x + w) >= (bin_img.shape[1] - 2)

        # Only reject very elongated, line-dominant blobs. Earlier logic could
        # reject valid symbols when the crop was tight and touched ROI borders.
        if aspect >= 6.0 and fill <= 0.38:
            return True
        if touches_top and touches_bottom and width_ratio <= 0.24 and aspect >= 3.8:
            return True
        if touches_left and touches_right and height_ratio <= 0.24 and aspect >= 3.8:
            return True
        if touches_bottom and aspect >= 5.2 and fill < 0.30:
            return True
        if touches_bottom and touches_top and touches_left and touches_right and fill < 0.20:
            return True
        return False

    @staticmethod
    def _arrow_direction_ok(crop_bgr: np.ndarray, expected: str) -> bool:
        if expected not in {"LEFT", "RIGHT"}:
            return True
        if crop_bgr is None or crop_bgr.size == 0:
            return False
        mask = preprocess_bin(crop_bgr)
        cnt = largest_contour(mask)
        if cnt is None or cv2.contourArea(cnt) < 150.0:
            return False

        # Centroid-to-tip direction check
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            return True
        cx = float(M["m10"] / M["m00"])
        pts = cnt.reshape(-1, 2).astype(np.float32)
        dists = np.sum((pts - np.array([cx, float(M["m01"] / M["m00"])])) ** 2, axis=1)
        tip = pts[int(np.argmax(dists))]
        dx = float(tip[0] - cx)
        if abs(dx) < 3.0:
            return True  # ambiguous, accept
        detected = "RIGHT" if dx > 0 else "LEFT"
        return detected == expected


# ---------------------------------------------------------------------------
# Debug overlay
# ---------------------------------------------------------------------------
def draw_symbol_debug(
    frame_bgr: np.ndarray,
    candidate: SymbolCandidate,
    result: SymbolResult,
) -> None:
    if candidate.roi_bbox is not None:
        x, y, w, h = candidate.roi_bbox
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (255, 200, 0), 1)
    if candidate.bbox is not None:
        x, y, w, h = candidate.bbox
        color = (0, 255, 0) if result.accepted else (0, 165, 255)
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)
        text = f"S:{result.label or '-'} {result.confidence:.2f}"
        cv2.putText(
            frame_bgr, text, (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2,
        )
