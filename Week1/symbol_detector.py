import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ---------- preprocess ----------
def preprocess_gray(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray


def preprocess_bin(bgr: np.ndarray) -> np.ndarray:
    gray = preprocess_gray(bgr)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
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
    # Arrows in this symbol set are moderately complex concave polygons.
    if len(approx) < 7 or len(approx) > 12:
        return False
    try:
        hull_idx = cv2.convexHull(approx, returnPoints=False)
        if hull_idx is None or len(hull_idx) < 3:
            return False
        defects = cv2.convexityDefects(approx, hull_idx)
        if defects is None:
            return False
        # Convexity defect depth is fixed-point (depth/256 ~= pixels). Use adaptive floor.
        adaptive_depth = max(int(0.015 * peri * 256.0), int(defect_depth))
        meaningful = 0
        for i in range(defects.shape[0]):
            _, _, _, depth = defects[i, 0]
            if depth > adaptive_depth:
                meaningful += 1
        # In these templates arrows consistently have two significant defects.
        return meaningful == 2
    except cv2.error:
        return False


def arrow_direction(cnt: np.ndarray) -> Optional[str]:
    pts = cnt.reshape(-1, 2).astype(np.float32)
    if len(pts) < 3:
        return None

    # Estimate tail notch from deepest convexity defect; the tip is usually farthest from that notch.
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
    # tip/tail inference from defects is occasionally swapped; use the opposite vector.
    dx = float(tail_ref[0] - tip[0])
    dy = float(tail_ref[1] - tip[1])
    if abs(dx) > abs(dy):
        return "RIGHT" if dx > 0 else "LEFT"
    return "DOWN" if dy > 0 else "UP"


# ---------- templates ----------
@dataclass
class ShapeTemplate:
    label: str
    contour: np.ndarray
    verts: int
    extent: float
    solidity: float
    circularity: float


@dataclass
class FeatureTemplate:
    label: str
    kp: list
    des: np.ndarray


class TemplateLibrary:
    """
    Loads templates from templates/ folder.

    - Silhouette shapes (everything except FINGERPRINT) are loaded as contours for matchShapes
    - FINGERPRINT is loaded as AKAZE features (more stable than ORB for prints)
    """
    def __init__(self, templates_dir: str):
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
            if f.lower().endswith((".png", ".jpg", ".jpeg")) and not f.startswith("._")
        ]
        if not files:
            raise RuntimeError(f"No template images found in {self.templates_dir}")

        akaze = cv2.AKAZE_create()

        self.shape_templates = []
        self.feature_templates = []

        for fn in sorted(files):
            label = os.path.splitext(fn)[0].strip().upper()
            path = os.path.join(self.templates_dir, fn)

            img = cv2.imread(path)
            if img is None:
                continue

            if label == "FINGERPRINT":
                gray = preprocess_gray(img)
                kp, des = akaze.detectAndCompute(gray, None)
                if des is not None and len(kp) >= 20:
                    self.feature_templates.append(FeatureTemplate(label, kp, des))
                continue

            th = preprocess_bin(img)
            cnt = largest_contour(th)
            if cnt is None or cv2.contourArea(cnt) < 200:
                continue

            verts, extent, solidity, circularity = self._shape_descriptor(cnt)
            self.shape_templates.append(
                ShapeTemplate(label, cnt, verts, extent, solidity, circularity)
            )

        if not self.shape_templates and not self.feature_templates:
            raise RuntimeError("No templates loaded successfully.")
        return self


# ---------- detector ----------
class SymbolDetector:
    def __init__(
        self,
        lib: TemplateLibrary,
        min_area: int = 1800,
        match_thresh: float = 0.30,
        cross_shape_thresh: float = 0.72,
        fp_min_good: int = 18,
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
        self.arrow_shape_thresh = 0.22
        self.arrow_defect_depth = 2200

        self.fp_min_good = int(fp_min_good)
        self.fp_ratio = float(fp_ratio)

        self.akaze = cv2.AKAZE_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

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
        cand_desc: Tuple[int, float, float, float], templ: ShapeTemplate
    ) -> float:
        c_verts, c_extent, c_solidity, c_circularity = cand_desc
        p_verts = 0.02 * abs(float(c_verts - templ.verts))
        p_extent = 0.35 * abs(c_extent - templ.extent)
        p_solidity = 0.25 * abs(c_solidity - templ.solidity)
        p_circularity = 0.20 * abs(c_circularity - templ.circularity)
        return float(p_verts + p_extent + p_solidity + p_circularity)

    def detect(self, frame_bgr: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None):
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

        for c in cnts:
            if cv2.contourArea(c) < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(c)
            if w <= 0 or h <= 0:
                continue

            cand = work[y:y+h, x:x+w]

            th = preprocess_bin(cand)
            cnt = largest_contour(th)
            if cnt is None or cv2.contourArea(cnt) < 200:
                continue
            cand_desc = self._shape_descriptor(cnt)

            # 1) fingerprint (AKAZE ratio test) first for texture symbol
            fp_score = self._match_fingerprint(cand)
            if fp_score is not None:
                results.append({"label": "FINGERPRINT", "bbox": (x + x_off, y + y_off, w, h), "score": fp_score})
                continue

            # 2) arrow by dedicated arrow-template gate, then geometry direction.
            if self.arrow_templates:
                arrow_score = min(
                    cv2.matchShapes(cnt, t.contour, cv2.CONTOURS_MATCH_I1, 0.0)
                    for t in self.arrow_templates
                )
                if arrow_score <= self.arrow_shape_thresh:
                    d = arrow_direction(cnt)
                    if d:
                        results.append({"label": d, "bbox": (x + x_off, y + y_off, w, h), "score": float(arrow_score)})
                        continue

            # 3) silhouette matching
            best_label = None
            best_score = 1e9
            cross_score = 1e9

            for t in self.non_arrow_templates:
                s = cv2.matchShapes(cnt, t.contour, cv2.CONTOURS_MATCH_I1, 0.0)
                s += self._descriptor_penalty(cand_desc, t)
                if t.label in {"PLUS", "CROSS"} and s < cross_score:
                    cross_score = s
                if s < best_score:
                    best_score = s
                    best_label = t.label

            accept = False
            if best_label is not None:
                if best_label in {"PLUS", "CROSS"}:
                    accept = best_score <= self.cross_shape_thresh
                else:
                    accept = best_score <= self.match_thresh

            if accept:
                if best_label in {"PLUS", "CROSS"}:
                    best_label = "CROSS"
                results.append({"label": best_label, "bbox": (x + x_off, y + y_off, w, h), "score": float(best_score)})
                continue

            # 2b) dedicated cross fallback when cross is close but not top-1
            if cross_score <= self.cross_fallback_thresh:
                results.append({"label": "CROSS", "bbox": (x + x_off, y + y_off, w, h), "score": float(cross_score)})
                continue

            # 4) geometry-only arrow fallback
            if is_arrow(cnt, defect_depth=self.arrow_defect_depth):
                d = arrow_direction(cnt)
                if d:
                    results.append({"label": d, "bbox": (x + x_off, y + y_off, w, h), "score": 0.10})
                    continue

        results.sort(key=lambda r: r["score"])
        return results, edges

    def _match_fingerprint(self, cand_bgr: np.ndarray) -> Optional[float]:
        if not self.feature_templates:
            return None

        edges = preprocess_edges(cand_bgr)
        edge_density = float(np.count_nonzero(edges)) / float(max(1, edges.size))
        if edge_density < 0.06:
            return None

        gray = preprocess_gray(cand_bgr)
        kp, des = self.akaze.detectAndCompute(gray, None)
        if des is None or len(kp) < 10:
            return None

        best_good = 0
        for t in self.feature_templates:
            knn = self.bf.knnMatch(des, t.des, k=2)
            good = 0
            for m, n in knn:
                if m.distance < self.fp_ratio * n.distance:
                    good += 1
            best_good = max(best_good, good)

        if best_good >= self.fp_min_good:
            return float(-best_good)  # negative = better rank
        return None
