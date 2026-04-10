"""MacBook detection server — uses the proven multi-stage pipeline from Final/.

Runs as a TCP server (via ``MacServer``).  For every incoming JPEG frame
from the Raspberry Pi, it runs the ``TFLiteSymbolDetector`` pipeline
(fast_filter → classify → probe) and returns a command string.

A ``cv2.imshow`` debug window shows the annotated feed in real time.
"""

import logging
import os
import sys

import cv2
import numpy as np

# Enable debug logging for the symbol detector pipeline
logging.basicConfig(level=logging.DEBUG,
                    format="%(name)s %(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Add Final/ to the Python path so we can import symbol_detector & utils
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(SCRIPT_DIR)  # Week3/
FINAL_DIR = os.path.join(WORKSPACE_DIR, "Final")
if FINAL_DIR not in sys.path:
    sys.path.insert(0, FINAL_DIR)

from symbol_detector import (
    TFLiteSymbolDetector,
    SymbolDetectorConfig,
    FastFilterConfig,
    draw_symbol_debug,
)
from mac_server import MacServer

# ---------------------------------------------------------------------------
# Templates path — use the existing templates/ in the workspace root
# ---------------------------------------------------------------------------
TEMPLATES_DIR = os.path.join(WORKSPACE_DIR, "templates")

# ---------------------------------------------------------------------------
# Symbol label → robot command mapping
# ---------------------------------------------------------------------------
LABEL_TO_COMMAND = {
    "LEFT":        "LEFT",
    "RIGHT":       "RIGHT",
    "STOP":        "STOP",
    "RECYCLE":     "ROTATE_360",
    "QR":          "PRINT_QR",
    "FINGERPRINT": "PRINT_FINGERPRINT",
    "LEFT_ARROW":  "LEFT",
    "ARROW_LEFT":  "LEFT",
    "RIGHT_ARROW": "RIGHT",
    "ARROW_RIGHT": "RIGHT",
    "BUTTON":      "STOP",
    "HAZARD":      "STOP",
    "OCTAGON":     "STOP",
    "STOP_SIGN":   "STOP",
}

DEFAULT_COMMAND = "FORWARD"
MIN_CLASSIFY_CONFIDENCE = 0.70   # was 0.58 — stricter to avoid false STOP
MIN_PROBE_CONFIDENCE = 0.74      # was 0.62
MIN_DIRECT_CONFIDENCE = 0.82     # was 0.78
MIN_ARROW_CLASSIFY_CONFIDENCE = 0.76   # was 0.72
MIN_ARROW_PROBE_CONFIDENCE = 0.82      # was 0.78
MIN_ARROW_DIRECT_CONFIDENCE = 0.90     # was 0.88


# ---------------------------------------------------------------------------
# Build the detector — with a much wider ROI than the on-robot default,
# since the Pi sends a full frame specifically containing a symbol.
# ---------------------------------------------------------------------------
def _build_detector() -> TFLiteSymbolDetector:
    config = SymbolDetectorConfig(
        templates_dir=TEMPLATES_DIR,
        confidence_threshold=0.60,
        fast_filter=FastFilterConfig(
            roi_top_ratio=0.02,
            roi_bottom_ratio=0.78,   # ignore the lower area where track clutter lives
            roi_left_ratio=0.02,
            roi_right_ratio=0.98,
            min_area=70.0,
            padding=16,
            min_bbox_size=14,
            max_bbox_aspect_ratio=3.8,
            min_fill_ratio=0.12,
            max_center_y_ratio=0.76,
        ),
    )
    det = TFLiteSymbolDetector(config)
    if det.enabled:
        print(f"[mac_detector] Detector loaded from {TEMPLATES_DIR}")
    else:
        print(f"[mac_detector] WARNING: detector failed to load: {det.reason}")
    return det


# ---------------------------------------------------------------------------
# Detection callback wired to MacServer
# ---------------------------------------------------------------------------
_detector: TFLiteSymbolDetector | None = None
_recycle_template_bin: np.ndarray | None = None


def _load_recycle_template() -> np.ndarray | None:
    """Load and preprocess Recycle template for binary IoU matching (Week2 approach)."""
    for ext in (".png", ".jpg"):
        path = os.path.join(TEMPLATES_DIR, f"Recycle{ext}")
        if os.path.isfile(path):
            img = cv2.imread(path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            block = min(141, max(21, (min(gray.shape[:2]) // 2) * 2 - 1))
            th = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV, block, 15,
            )
            k = np.ones((3, 3), np.uint8)
            th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
            th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
            cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                return None
            cnt = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            roi = th[y:y+h, x:x+w]
            # Normalize: foreground white
            if int(np.count_nonzero(roi)) > roi.size // 2:
                roi = cv2.bitwise_not(roi)
            roi = cv2.resize(roi, (50, 50), interpolation=cv2.INTER_NEAREST)
            print(f"[mac_detector] Loaded Recycle template from {path}")
            return roi
    print("[mac_detector] WARNING: Recycle template not found")
    return None


def _bin_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Binary IoU between two binary images (Week2 approach)."""
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


def _check_recycle_iou(frame: np.ndarray) -> float:
    """Check binary IoU of largest contour against Recycle template."""
    global _recycle_template_bin
    if _recycle_template_bin is None:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    block = min(141, max(21, (min(gray.shape[:2]) // 2) * 2 - 1))
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, block, 15,
    )
    k = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 200:
        return 0.0
    x, y, w, h = cv2.boundingRect(cnt)
    roi = th[y:y+h, x:x+w]
    if int(np.count_nonzero(roi)) > roi.size // 2:
        roi = cv2.bitwise_not(roi)
    roi = cv2.resize(roi, (50, 50), interpolation=cv2.INTER_NEAREST)
    return _bin_iou(roi, _recycle_template_bin)


def _get_solidity(frame: np.ndarray) -> float:
    """Get solidity of the largest contour in frame. RECYCLE ~0.5-0.7, STOP ~0.9+."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    k = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 1.0
    cnt = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    if area < 100:
        return 1.0
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    return area / max(1e-6, hull_area)


def _count_triangles(frame: np.ndarray) -> int:
    """Count triangle-like contours using Otsu + dual masks (from Final/)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask_inv = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _, mask_norm = cv2.threshold(gray, 0, 255,
                                 cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), dtype=np.uint8)
    img_area = float(frame.shape[0] * frame.shape[1])
    min_tri_area = max(40.0, img_area * 0.02)
    tri = 0
    seen = []
    for mask in (mask_inv, mask_norm):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < min_tri_area:
                continue
            peri = cv2.arcLength(cnt, True)
            if peri < 1e-6:
                continue
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) != 3:
                continue
            x, y, w, h = cv2.boundingRect(approx)
            if w < 8 or h < 8:
                continue
            # Deduplicate nearby triangles
            if any(abs(x - bx) < 4 and abs(y - by) < 4
                   for bx, by, _, _ in seen):
                continue
            seen.append((x, y, w, h))
            tri += 1
    return tri


def detect(frame: np.ndarray) -> str:
    """Run detection on a full frame sent from the Pi.

    Tries three approaches and picks the best confident result:
    1. fast_filter → classify (standard pipeline)
    2. probe (whole-image fallback)
    3. Direct full-frame detector.detect() (bypass fast_filter entirely)
    """
    global _detector
    if _detector is None:
        _detector = _build_detector()

    if not _detector.enabled:
        _annotate(frame, DEFAULT_COMMAND, None, None)
        return DEFAULT_COMMAND

    # --- Quick triangle check: ≥2 triangles → likely RECYCLE ---
    tri_count = _count_triangles(frame)
    recycle_iou = _check_recycle_iou(frame)
    solidity = _get_solidity(frame)
    print(f"[detect] RECYCLE checks: tri={tri_count}, IoU={recycle_iou:.3f}, sol={solidity:.3f}")

    # Triangle shortcut: ≥2 triangles + low solidity = definitely RECYCLE
    if tri_count >= 2 and solidity < 0.82:
        print(f"[detect] Triangle shortcut: {tri_count} triangles + sol={solidity:.3f} → RECYCLE")
        command = LABEL_TO_COMMAND.get("RECYCLE", DEFAULT_COMMAND)
        _annotate(frame, command, None, None)
        return command

    # IoU shortcut: strong template match + not solid shape
    if recycle_iou >= 0.18 and solidity < 0.82:
        print(f"[detect] Recycle IoU shortcut: {recycle_iou:.3f} + sol={solidity:.3f} → RECYCLE")
        command = LABEL_TO_COMMAND.get("RECYCLE", DEFAULT_COMMAND)
        _annotate(frame, command, None, None)
        return command

    # Combined: even weak signals together = RECYCLE
    if tri_count >= 1 and recycle_iou >= 0.13 and solidity < 0.80:
        print(f"[detect] Combined RECYCLE: tri={tri_count}, IoU={recycle_iou:.3f}, sol={solidity:.3f}")
        command = LABEL_TO_COMMAND.get("RECYCLE", DEFAULT_COMMAND)
        _annotate(frame, command, None, None)
        return command

    best_result = None
    best_candidate = None

    def _passes_confidence_gate(label: str | None, confidence: float, stage: str) -> bool:
        if label in {"LEFT", "RIGHT"}:
            if stage == "classify":
                return confidence >= MIN_ARROW_CLASSIFY_CONFIDENCE
            if stage == "probe":
                return confidence >= MIN_ARROW_PROBE_CONFIDENCE
            return confidence >= MIN_ARROW_DIRECT_CONFIDENCE
        if stage == "classify":
            return confidence >= MIN_CLASSIFY_CONFIDENCE
        if stage == "probe":
            return confidence >= MIN_PROBE_CONFIDENCE
        return confidence >= MIN_DIRECT_CONFIDENCE

    # --- Approach 1: fast_filter → classify ---
    candidate = _detector.fast_filter(frame)
    print(f"[detect] fast_filter: found={candidate.found}, "
          f"area={candidate.area:.0f}, bbox={candidate.bbox}")

    if candidate.found:
        result = _detector.classify(frame, candidate)
        print(f"[detect] classify: accepted={result.accepted}, "
              f"label={result.label}, conf={result.confidence:.2f}, "
              f"reason={result.reason}")
        if (result.accepted and result.label
                and _passes_confidence_gate(
                    result.label, result.confidence, "classify"
                )):
            best_result = result
            best_candidate = candidate

    # --- Approach 2: probe (whole-image scan, includes triangle heuristic) ---
    if best_result is None or best_result.confidence < 0.70:
        probe_result = _detector.probe_symbol(frame)
        print(f"[detect] probe: accepted={probe_result.accepted}, "
              f"label={probe_result.label}, conf={probe_result.confidence:.2f}, "
              f"reason={probe_result.reason}")
        if (probe_result.accepted and probe_result.label
                and _passes_confidence_gate(
                    probe_result.label, probe_result.confidence, "probe"
                )):
            if best_result is None or probe_result.confidence > best_result.confidence:
                best_result = probe_result
                best_candidate = candidate

    # RECYCLE uses a very specific triangle-count heuristic that doesn't
    # false-positive easily.  If probe found RECYCLE, trust it and skip
    # approach 3 which often misclassifies as STOP or QR.
    recycle_locked = (best_result is not None
                      and best_result.label == "RECYCLE"
                      and best_result.confidence >= 0.45)

    # --- Approach 3: direct detect on full frame ---
    if not recycle_locked and (best_result is None or best_result.confidence < 0.60):
        direct_results = _detector.detector.detect(frame)
        if direct_results:
            d = direct_results[0]
            print(f"[detect] direct: label={d['label']}, score={d['score']:.3f}")
            # Convert score to confidence (lower score = better match)
            from symbol_detector import ACTIVE_SYMBOL_LABELS
            if d["label"] in ACTIVE_SYMBOL_LABELS:
                from utils import clamp
                if d["label"] == "FINGERPRINT":
                    conf = clamp(max(0.0, -d["score"]) / 8.0 + 0.40, 0.0, 1.0)
                elif d["label"] == "QR":
                    conf = 0.98
                else:
                    conf = clamp(1.0 - d["score"], 0.0, 1.0)
                if _passes_confidence_gate(d["label"], conf, "direct") and (
                        best_result is None or conf > best_result.confidence):
                    from symbol_detector import SymbolResult
                    best_result = SymbolResult(
                        enabled=True, label=d["label"], confidence=conf,
                        action_label=d["label"], bbox=d.get("bbox"),
                        accepted=True, reason="Direct detect.",
                    )
        else:
            print("[detect] direct: no results")

    # Map label → command
    command = DEFAULT_COMMAND
    if best_result is not None and best_result.accepted and best_result.label:
        # --- Solidity guard: RECYCLE has low solidity, STOP/arrows are solid ---
        # If pipeline says STOP/LEFT/RIGHT but shape is hollow, override to RECYCLE
        if best_result.label in {"STOP", "LEFT", "RIGHT"}:
            print(f"[detect] Solidity check: {solidity:.3f} (label={best_result.label}, "
                  f"tri={tri_count}, IoU={recycle_iou:.3f})")
            if solidity < 0.78 and (tri_count >= 1 or recycle_iou >= 0.13):
                print(f"[detect] Low solidity → overriding {best_result.label} to RECYCLE")
                best_result.label = "RECYCLE"
            elif solidity < 0.72:
                # Very low solidity — almost certainly not STOP/arrow
                print(f"[detect] Very low solidity → overriding {best_result.label} to RECYCLE")
                best_result.label = "RECYCLE"
        command = LABEL_TO_COMMAND.get(best_result.label, DEFAULT_COMMAND)

    print(f"[detect] → {command} (label={best_result.label if best_result else None})")
    _annotate(frame, command, best_candidate, best_result)
    return command


def _annotate(frame, command, candidate, result):
    """Draw detection overlay and show in a debug window."""
    display = frame.copy()

    if candidate is not None and result is not None:
        draw_symbol_debug(display, candidate, result)

    # Command text
    color = (0, 255, 0) if command != DEFAULT_COMMAND else (128, 128, 128)
    cv2.putText(display, command, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)

    cv2.imshow("MacBook Detector", display)
    cv2.waitKey(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MacBook detection server")
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()

    # Pre-build detector before starting the server
    _detector = _build_detector()
    _recycle_template_bin = _load_recycle_template()

    print("[mac_detector] Starting server with full detection pipeline")
    server = MacServer(detect_fn=detect, port=args.port)
    try:
        server.start()
    except KeyboardInterrupt:
        print("\n[mac_detector] Shutting down")
        server.stop()
        cv2.destroyAllWindows()
