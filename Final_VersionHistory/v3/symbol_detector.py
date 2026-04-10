import os
import cv2
import numpy as np


# ============================================================
# FAST WEEK 3 SYMBOL DETECTOR
# Only supports:
#   Arrows: LEFT RIGHT UP DOWN
#   Symbols: BUTTON HAZARD RECYCLE QR FINGERPRINT
# Optimised for speed + less false RECYCLE matches
# ============================================================


def preprocess_gray(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray


def preprocess_bin(bgr):
    gray = preprocess_gray(bgr)
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        51, 10
    )
    k = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
    return th


def largest_contour(bin_img):
    cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


def contour_roi_bin(bin_img, cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    if w <= 1 or h <= 1:
        return None
    roi = bin_img[y:y+h, x:x+w]
    white = int(np.count_nonzero(roi))
    if white > roi.size // 2:
        roi = cv2.bitwise_not(roi)
    roi = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_NEAREST)
    return roi


def bin_iou(a, b):
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


def shape_desc(cnt):
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
    aspect = w / float(max(1, h))
    return {
        "verts": verts,
        "area": area,
        "extent": extent,
        "solidity": solidity,
        "circularity": circularity,
        "aspect": aspect,
        "bbox": (x, y, w, h),
    }


def convexity_defect_count(cnt, defect_depth=1400):
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


def is_arrow_shape(cnt):
    peri = cv2.arcLength(cnt, True)
    if peri <= 1e-6:
        return False
    approx = cv2.approxPolyDP(cnt, 0.015 * peri, True)
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

        meaningful = 0
        adaptive_depth = max(int(0.015 * peri * 256.0), 2200)
        for i in range(defects.shape[0]):
            _, _, _, depth = defects[i, 0]
            if depth > adaptive_depth:
                meaningful += 1
        return meaningful == 2
    except cv2.error:
        return False


class TemplateLibrary:
    def __init__(self, templates_dir):
        self.templates_dir = templates_dir
        self.templates = {}

    def load(self):
        if not os.path.isdir(self.templates_dir):
            raise RuntimeError(f"Templates dir not found: {self.templates_dir}")

        wanted = {
            "LEFT", "RIGHT", "UP", "DOWN",
            "BUTTON", "HAZARD", "RECYCLE",
            "QR", "FINGERPRINT"
        }

        for fn in os.listdir(self.templates_dir):
            if not fn.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            label = os.path.splitext(fn)[0].strip().upper()
            if label not in wanted:
                continue

            img = cv2.imread(os.path.join(self.templates_dir, fn))
            if img is None:
                continue

            th = preprocess_bin(img)
            cnt = largest_contour(th)
            if cnt is None or cv2.contourArea(cnt) < 100:
                continue

            roi = contour_roi_bin(th, cnt)
            self.templates[label] = {
                "img": img,
                "bin": roi,
                "cnt": cnt,
            }

        return self


class SymbolDetector:
    def __init__(self, lib):
        self.templates = lib.templates
        self.qr_detector = cv2.QRCodeDetector()

    def detect(self, frame_bgr, roi=None, debug=False):
        x_off = y_off = 0
        work = frame_bgr

        if roi is not None:
            x1, y1, x2, y2 = roi
            work = frame_bgr[y1:y2, x1:x2]
            x_off, y_off = x1, y1

        bin_img = preprocess_bin(work)
        cnt = largest_contour(bin_img)

        if cnt is None or cv2.contourArea(cnt) < 1200:
            return ([], None) if not debug else ([], None, [])

        desc = shape_desc(cnt)
        cand_bin = contour_roi_bin(bin_img, cnt)
        if cand_bin is None:
            return ([], None) if not debug else ([], None, [])

        x, y, w, h = desc["bbox"]
        cand = work[y:y+h, x:x+w]

        results = []
        debug_items = []

        # ============================================================
        # 1. QR: only if QR detector explicitly sees QR
        # ============================================================
        try:
            qr_ok, points = self.qr_detector.detect(cand)
        except cv2.error:
            qr_ok, points = False, None

        if qr_ok and points is not None:
            results.append({
                "label": "QR",
                "bbox": (x + x_off, y + y_off, w, h),
                "score": -5.0
            })
            return (results, None) if not debug else (results, None, debug_items)

        # ============================================================
        # 2. FINGERPRINT: only if texture is high
        # Fast texture gate to avoid false positives
        # ============================================================
        gray = preprocess_gray(cand)
        edges = cv2.Canny(gray, 50, 140)
        edge_density = float(np.count_nonzero(edges)) / float(max(1, edges.size))
        if edge_density > 0.18:
            if "FINGERPRINT" in self.templates:
                fp_iou = bin_iou(cand_bin, self.templates["FINGERPRINT"]["bin"])
                if fp_iou >= 0.18:
                    results.append({
                        "label": "FINGERPRINT",
                        "bbox": (x + x_off, y + y_off, w, h),
                        "score": 1.0 - fp_iou
                    })
                    return (results, None) if not debug else (results, None, debug_items)

        # ============================================================
        # 3. ARROW: only if arrow geometry looks right
        # ============================================================
        if is_arrow_shape(cnt):
            best_label = None
            best_iou = 0.0
            for label in ("LEFT", "RIGHT", "UP", "DOWN"):
                if label not in self.templates:
                    continue
                iou = bin_iou(cand_bin, self.templates[label]["bin"])
                if iou > best_iou:
                    best_iou = iou
                    best_label = label

            if best_label is not None and best_iou >= 0.28:
                results.append({
                    "label": best_label,
                    "bbox": (x + x_off, y + y_off, w, h),
                    "score": 1.0 - best_iou
                })
                return (results, None) if not debug else (results, None, debug_items)

        # ============================================================
        # 4. BUTTON / HAZARD / RECYCLE
        # Use shape gate first, then template IoU
        # ============================================================
        defects = convexity_defect_count(cnt)
        verts = desc["verts"]
        circularity = desc["circularity"]
        solidity = desc["solidity"]
        extent = desc["extent"]

        # BUTTON: round-ish
        if "BUTTON" in self.templates:
            if circularity > 0.72 and solidity > 0.88 and extent > 0.55:
                btn_iou = bin_iou(cand_bin, self.templates["BUTTON"]["bin"])
                if btn_iou >= 0.26:
                    results.append({
                        "label": "BUTTON",
                        "bbox": (x + x_off, y + y_off, w, h),
                        "score": 1.0 - btn_iou
                    })
                    return (results, None) if not debug else (results, None, debug_items)

        # HAZARD: not round, not arrow, moderate symmetry
        if "HAZARD" in self.templates:
            if 5 <= verts <= 10 and solidity > 0.78 and extent > 0.32 and circularity < 0.88:
                haz_iou = bin_iou(cand_bin, self.templates["HAZARD"]["bin"])
                if haz_iou >= 0.18:
                    results.append({
                        "label": "HAZARD",
                        "bbox": (x + x_off, y + y_off, w, h),
                        "score": 1.0 - haz_iou
                    })
                    return (results, None) if not debug else (results, None, debug_items)

        # RECYCLE: only allow when defect count is high enough
        if "RECYCLE" in self.templates:
            if defects >= 3 and solidity < 0.92:
                rec_iou = bin_iou(cand_bin, self.templates["RECYCLE"]["bin"])
                if rec_iou >= 0.20:
                    results.append({
                        "label": "RECYCLE",
                        "bbox": (x + x_off, y + y_off, w, h),
                        "score": 1.0 - rec_iou
                    })
                    return (results, None) if not debug else (results, None, debug_items)

        return (results, None) if not debug else (results, None, debug_items)