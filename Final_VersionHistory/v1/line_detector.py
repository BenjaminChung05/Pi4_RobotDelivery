from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from utils import clamp


@dataclass
class LineDetectorConfig:
    roi_top_ratio: float = 0.30
    roi_bottom_ratio: float = 1.0
    threshold_value: int = 60
    use_adaptive_threshold: bool = True
    adaptive_block_size: int = 31
    adaptive_c: int = 8
    blur_kernel: tuple[int, int] = (5, 5)
    min_contour_area: float = 200.0
    morph_kernel_size: int = 3
    line_colors: tuple[str, ...] = ("BLACK", "BLUE")
    lost_search_flip_frames: int = 4
    alternate_search_on_loss: bool = False
    lost_turn_delay_frames: int = 3
    lost_forward_speed_scale: float = 0.65
    recovery_preferred_turn: str | None = None
    require_bottom_contact: bool = True
    bottom_contact_band_ratio: float = 0.28
    bottom_contact_min_cols_ratio: float = 0.012
    high_error_confirm_frames: int = 2
    min_black_width_px: int = 18       # ignore black contours narrower than this (~1cm)
    min_black_support_rows: int = 12   # reject black blobs that occupy too few image rows
    black_support_min_cols: int = 4    # minimum filled pixels for a row to count as support
    min_black_contrast_delta: float = 24.0  # ring_mean - contour_mean must exceed this
    black_contrast_ring_px: int = 7         # pixels of surrounding ring used for contrast
    # Center calibration: positive = robot physically sits left of camera centre
    # (robot hugs right side of line → increase this value, e.g. +10..+30)
    center_offset_px: float = 0.0
    # Lookahead: fraction of ROI height above which the centroid band starts.
    # 0.60 = bottom 40% only (no lookahead, stable but reacts late to curves)
    # 0.35 = bottom 65% (more lookahead, anticipates turns earlier)
    centroid_bottom_band_ratio: float = 0.60


@dataclass
class SteeringConfig:
    base_speed: float = 45.0
    steer_gain: float = 0.35
    max_adjust: float = 30.0
    deadband_px: float = 5.0
    lost_turn_speed: float = 62.0
    tank_turn_error_px: float = 80.0
    tank_turn_speed: float = 70.0
    error_filter_alpha: float = 0.40
    max_correction_step: float = 16.0
    invert_turn_direction: bool = False
    # Derivative term — damps overshoot on tight corners
    d_gain: float = 0.06
    # Adaptive speed — slows down on sharp turns, speeds up on straights
    speed_adapt: bool = True
    speed_straight_scale: float = 1.25   # multiplier when |error| < straight_threshold
    speed_curve_scale: float = 0.72      # multiplier when |error| > curve_threshold
    speed_straight_threshold: float = 20.0  # px below this → treat as straight
    speed_curve_threshold: float = 50.0     # px above this → treat as sharp curve


@dataclass
class LineDetectionResult:
    found: bool
    error_px: float
    centroid_x: int | None
    centroid_y: int | None
    contour_area: float
    roi_top: int
    roi_bottom: int
    is_t_junction: bool = False
    contour: np.ndarray | None = None
    mask: np.ndarray | None = None


@dataclass
class MotorCommand:
    left_speed: float
    right_speed: float
    correction: float
    recovery_mode: bool = False


class LineFollower:
    """
    Fast OpenCV line following.

    The line detector always runs every frame so steering remains responsive
    even when symbol classification is rate-limited.
    """

    def __init__(
        self,
        detector_config: LineDetectorConfig | None = None,
        steering_config: SteeringConfig | None = None,
    ) -> None:
        self.detector_config = detector_config or LineDetectorConfig()
        self.steering_config = steering_config or SteeringConfig()
        self.last_error = 0.0
        self.filtered_error = 0.0
        self.last_filtered_error = 0.0
        self.last_correction = 0.0
        self.last_centroid_x: int | None = None
        self.last_frame_width: int = 0
        self.lost_frames = 0
        self.search_direction = -1
        self._junction_latched = False
        self._high_error_frames = 0

    def reset_steering(self) -> None:
        """Clear error/correction memory (call after spins or long actions)."""
        self.filtered_error = 0.0
        self.last_filtered_error = 0.0
        self.last_error = 0.0
        self.last_correction = 0.0
        self._high_error_frames = 0
        self._junction_latched = False

    @staticmethod
    def _bottom_contact_cols(mask: np.ndarray, contour: np.ndarray, band_ratio: float) -> int:
        if mask is None or contour is None:
            return 0
        h, w = mask.shape[:2]
        if h < 4 or w < 4:
            return 0

        ratio = float(clamp(band_ratio, 0.05, 0.60))
        band_start = int(h * (1.0 - ratio))
        band_start = int(clamp(band_start, 0, h - 1))

        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
        bottom_band = contour_mask[band_start:, :]
        if bottom_band.size == 0:
            return 0
        return int(np.count_nonzero(np.any(bottom_band > 0, axis=0)))

    @staticmethod
    def _support_rows(mask: np.ndarray, contour: np.ndarray, min_cols_per_row: int) -> int:
        if mask is None or contour is None:
            return 0
        h, w = mask.shape[:2]
        if h < 4 or w < 4:
            return 0

        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
        row_counts = np.count_nonzero(contour_mask > 0, axis=1)
        return int(np.count_nonzero(row_counts >= max(1, int(min_cols_per_row))))

    @staticmethod
    def _contour_contrast(gray: np.ndarray, contour: np.ndarray, ring_px: int) -> float:
        if gray is None or contour is None:
            return 0.0
        h, w = gray.shape[:2]
        if h < 6 or w < 6:
            return 0.0

        contour_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
        inside_vals = gray[contour_mask > 0]
        if inside_vals.size < 10:
            return 0.0

        ring_px = int(clamp(ring_px, 1, 24))
        kernel_size = max(3, ring_px * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated = cv2.dilate(contour_mask, kernel, iterations=1)
        ring_mask = cv2.bitwise_and(dilated, cv2.bitwise_not(contour_mask))
        ring_vals = gray[ring_mask > 0]
        if ring_vals.size < 12:
            return 0.0

        return float(np.mean(ring_vals) - np.mean(inside_vals))

    @staticmethod
    def _is_t_junction(mask: np.ndarray, contour: np.ndarray) -> bool:
        if mask is None or contour is None:
            return False
        h, w = mask.shape[:2]
        if h < 20 or w < 20:
            return False

        # Focus on contour bounding area to avoid unrelated noise.
        x, y, cw, ch = cv2.boundingRect(contour)
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + cw)
        y2 = min(h, y + ch)
        roi = mask[y1:y2, x1:x2]
        if roi.size == 0:
            return False

        rh, rw = roi.shape[:2]
        if rh < 16 or rw < 16:
            return False

        top_band = roi[int(rh * 0.08):int(rh * 0.34), :]
        bottom_band = roi[int(rh * 0.72):int(rh * 0.96), :]
        if top_band.size == 0 or bottom_band.size == 0:
            return False

        top_cols = np.count_nonzero(np.any(top_band > 0, axis=0))
        bottom_cols = np.count_nonzero(np.any(bottom_band > 0, axis=0))
        if top_cols < int(0.32 * rw) or bottom_cols < int(0.08 * rw):
            return False

        left_top = np.count_nonzero(np.any(top_band[:, : max(1, rw // 2)] > 0, axis=0))
        right_top = np.count_nonzero(np.any(top_band[:, rw // 2 :] > 0, axis=0))
        center_bottom = np.count_nonzero(
            np.any(bottom_band[:, int(rw * 0.35): int(rw * 0.65)] > 0, axis=0)
        )

        # T-junction: clear left+right arms at top and a center stem at bottom.
        return (
            left_top >= int(0.12 * rw)
            and right_top >= int(0.12 * rw)
            and center_bottom >= max(2, int(0.08 * rw))
        )

    def _build_color_mask(
        self,
        roi_bgr: np.ndarray,
        *,
        hsv: np.ndarray | None = None,
        colors: tuple[str, ...] | list[str] | None = None,
    ) -> np.ndarray:
        if hsv is None:
            hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        cfg = self.detector_config
        tokens = colors if colors is not None else cfg.line_colors

        # Adaptive BLACK threshold: use the frame's median V to set the upper
        # V limit.  Multiplier 0.47 balances bright-light line detection vs
        # shadow rejection.  Shadows are typically V 100-130, black tape V 40-90.
        median_v = float(np.median(hsv[:, :, 2]))
        black_v_upper = int(clamp(median_v * 0.47, 45, 105))
        black_s_upper = 80 if median_v > 140 else 60

        # Built-in HSV ranges so we can extend from black-only to multi-color lines.
        color_ranges: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
            "BLACK": [((0, 0, 0), (180, black_s_upper, black_v_upper))],
            "BLUE": [
                ((85, 30, 40), (130, 255, 255)),   # Core blue: lower S for light blue
                ((70, 15, 40), (85, 255, 255)),    # Cyan / light blue
                ((130, 40, 30), (145, 255, 255)),  # Blue-violet fringe
                ((95, 10, 30), (130, 50, 200)),    # Dark blue: low saturation
            ],
            "YELLOW": [((18, 70, 70), (40, 255, 255))],
            "GREEN": [((35, 50, 40), (90, 255, 255))],
            "WHITE": [((0, 0, 170), (180, 70, 255))],
            "RED": [
                ((0, 60, 30), (12, 255, 255)),
                ((165, 60, 30), (180, 255, 255)),
            ],
        }

        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        b = roi_bgr[:, :, 0].astype(np.int16)
        g = roi_bgr[:, :, 1].astype(np.int16)
        r = roi_bgr[:, :, 2].astype(np.int16)
        for token in tokens:
            key = token.strip().upper()
            ranges = color_ranges.get(key)
            if not ranges:
                continue
            for lower, upper in ranges:
                lower_np = np.array(lower, dtype=np.uint8)
                upper_np = np.array(upper, dtype=np.uint8)
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower_np, upper_np))

            if key == "BLUE":
                # Supplement HSV with BGR blue-dominance checks so darker,
                # less saturated blue paint still registers without treating
                # neutral gray/black stains as blue.
                blue_dom = (
                    ((b >= 48) & (b >= g + 10) & (b >= r + 18))
                    | ((b >= 38) & (b * 10 >= g * 12) & (b * 10 >= r * 15) & (b - r >= 12))
                )
                blue_dom_mask = np.where(blue_dom, 255, 0).astype(np.uint8)
                mask = cv2.bitwise_or(mask, blue_dom_mask)

        return mask

    def detect_line(self, frame_bgr: np.ndarray) -> LineDetectionResult:
        frame_h, frame_w = frame_bgr.shape[:2]
        self.last_frame_width = frame_w
        roi_top = int(frame_h * self.detector_config.roi_top_ratio)
        roi_bottom = int(frame_h * self.detector_config.roi_bottom_ratio)
        roi = frame_bgr[roi_top:roi_bottom, :]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, self.detector_config.blur_kernel, 0)
        _, mask_basic = cv2.threshold(
            gray,
            self.detector_config.threshold_value,
            255,
            cv2.THRESH_BINARY_INV,
        )

        if self.detector_config.use_adaptive_threshold:
            block_size = max(3, self.detector_config.adaptive_block_size)
            if block_size % 2 == 0:
                block_size += 1
            mask_adaptive = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                block_size,
                self.detector_config.adaptive_c,
            )
        else:
            mask_adaptive = np.zeros_like(mask_basic)

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask_color = self._build_color_mask(roi, hsv=hsv)
        color_tokens_set = {t.strip().upper() for t in self.detector_config.line_colors}
        steering_bias_top = int(roi.shape[0] * 0.25)

        # Build a separate colour-only mask (no BLACK) so we can detect
        # coloured lines that should override the black line.
        non_black_colors = [c for c in self.detector_config.line_colors
                            if c.strip().upper() != "BLACK"]
        mask_nonblack = None
        cmask_morph = None
        if non_black_colors:
            mask_nonblack = self._build_color_mask(roi, hsv=hsv, colors=non_black_colors)

        candidates: list[np.ndarray] = []
        # Primary: unified color mask (BLACK HSV + coloured together)
        candidates.append(mask_color)
        # NOTE: grayscale backup removed — it picks up shadows and tile gaps.
        # The HSV-based BLACK mask is sufficient for black line detection.

        kernel_size = self.detector_config.morph_kernel_size
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

        # Pre-compute morphed colour mask once (used for colour-overlap check)
        if mask_nonblack is not None:
            cmask_morph = cv2.morphologyEx(mask_nonblack, cv2.MORPH_OPEN, kernel, iterations=1)
            cmask_morph = cv2.morphologyEx(cmask_morph, cv2.MORPH_CLOSE, kernel, iterations=2)

        best_contour = None
        best_area = 0.0
        best_contact_cols = 0
        best_center_dist = 999.0
        best_score = -1.0
        best_mask = None
        # Also track the best *coloured* contour separately so we can
        # prefer it when it represents a real line (area threshold).
        color_contour = None
        color_area = 0.0
        color_contact = 0
        color_center_dist = 999.0
        color_mask_out = None
        _COLOR_MIN_AREA = 180  # ignore colour blobs smaller than this
        _MIN_ELONGATION = 1.4  # reject blobby stains (lines are elongated)

        for candidate_mask in candidates:
            mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
            steer_mask = mask.copy()
            steer_mask[:steering_bias_top, :] = 0
            contours, _ = cv2.findContours(steer_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area <= 0.0:
                    continue

                overlap_ratio = 0.0
                is_coloured = False
                if cmask_morph is not None and area >= _COLOR_MIN_AREA:
                    contour_px = np.zeros_like(mask)
                    cv2.drawContours(contour_px, [contour], -1, 255, thickness=-1)
                    overlap = cv2.bitwise_and(contour_px, cmask_morph)
                    overlap_ratio = float(cv2.countNonZero(overlap)) / max(1.0, area)
                    is_coloured = overlap_ratio > 0.20

                contact_cols = self._bottom_contact_cols(
                    mask,
                    contour,
                    self.detector_config.bottom_contact_band_ratio,
                )
                min_contact_cols = int(mask.shape[1] * self.detector_config.bottom_contact_min_cols_ratio)
                has_bottom_contact = contact_cols >= max(1, min_contact_cols)

                if self.detector_config.require_bottom_contact and not has_bottom_contact:
                    # Allow large contours even without bottom contact —
                    # they are clearly a line that just doesn't extend to
                    # the very bottom of the ROI.
                    if area < 800:
                        continue

                # --- Elongation filter: reject blobby stains ---
                bx_e, by_e, bw_e, bh_e = cv2.boundingRect(contour)
                elongation = max(bw_e, bh_e) / max(1, min(bw_e, bh_e))
                if elongation < _MIN_ELONGATION and area < 2000:
                    # Short & fat blob — likely a stain, not a line.
                    # Allow very large contours through (they span junctions).
                    continue

                if not is_coloured:
                    support_rows = self._support_rows(
                        mask,
                        contour,
                        self.detector_config.black_support_min_cols,
                    )
                    if support_rows < self.detector_config.min_black_support_rows:
                        # Tile gaps and stains often occupy only a handful of rows.
                        continue

                    if bw_e > mask.shape[1] * 0.45 and bh_e < max(10, int(mask.shape[0] * 0.10)):
                        # Very wide but shallow black contour: likely a tile gap.
                        continue

                    contrast_delta = self._contour_contrast(
                        gray,
                        contour,
                        self.detector_config.black_contrast_ring_px,
                    )
                    strong_bottom_contact = contact_cols >= max(
                        14, int(mask.shape[1] * 0.035)
                    )
                    tall_supported_contour = support_rows >= max(
                        self.detector_config.min_black_support_rows + 6,
                        int(mask.shape[0] * 0.22),
                    )
                    if (
                        contrast_delta < self.detector_config.min_black_contrast_delta
                        and not (strong_bottom_contact and tall_supported_contour)
                    ):
                        # Grey stains and shadows are dark in absolute terms, but
                        # not dark enough relative to their immediate surroundings.
                        # A real line can still pass here if it has strong bottom
                        # contact and enough vertical support through the ROI.
                        continue

                # --- Black shadow filter: ignore narrow black-only contours ---
                # Use rotated min-area-rect to get the TRUE minimum width,
                # regardless of contour orientation.  Reject any non-coloured
                # contour whose narrow dimension is < ~1 cm.
                min_width_px = self.detector_config.min_black_width_px
                if min_width_px > 0:
                    rot_rect = cv2.minAreaRect(contour)
                    true_min_dim = min(rot_rect[1])  # (w, h) of rotated rect
                    if true_min_dim < min_width_px:
                        if not is_coloured:
                            # Black-only and too narrow → tile gap/shadow, skip
                            continue

                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = float(M["m10"] / M["m00"])
                else:
                    cx = float(bx_e + bw_e / 2.0)
                center_dist = abs(cx - mask.shape[1] / 2.0)

                # Continuity bonus: prefer contours near last known centroid.
                # This prevents distant noise from yanking steering.
                continuity_dist = 999.0
                if self.last_centroid_x is not None:
                    continuity_dist = abs(cx - self.last_centroid_x)
                # Score: bottom-contact first, then closeness to last pos
                score = contact_cols * 1000.0 - continuity_dist * 0.5 - center_dist * 0.3

                # --- Check if this contour is on the coloured mask ---
                if is_coloured:
                    # This contour is substantially coloured → track it
                    if (color_contour is None
                            or contact_cols > color_contact
                            or (contact_cols == color_contact and area > color_area)):
                        color_contour = contour
                        color_area = area
                        color_contact = contact_cols
                        color_center_dist = center_dist
                        color_mask_out = mask

                # --- Normal best-contour selection (score-based) ---
                if best_contour is None or score > best_score:
                    best_contact_cols = contact_cols
                    best_area = area
                    best_contour = contour
                    best_mask = mask
                    best_center_dist = center_dist
                    best_score = score

        # If a substantial coloured contour was found, prefer it over the
        # default (usually black) contour so the robot steers onto the
        # coloured branch.
        if color_contour is not None:
            best_contour = color_contour
            best_area = color_area
            best_contact_cols = color_contact
            best_center_dist = color_center_dist
            best_mask = color_mask_out

        if best_contour is None:
            self.lost_frames += 1
            if (
                self.detector_config.alternate_search_on_loss
                and self.lost_frames >= self.detector_config.lost_search_flip_frames
            ):
                self.search_direction *= -1
                self.lost_frames = 0
            return LineDetectionResult(
                found=False,
                error_px=self.last_error,
                centroid_x=None,
                centroid_y=None,
                contour_area=0.0,
                roi_top=roi_top,
                roi_bottom=roi_bottom,
                is_t_junction=False,
                mask=best_mask,
            )

        contour = best_contour
        contour_area = best_area
        if contour_area < self.detector_config.min_contour_area:
            self.lost_frames += 1
            return LineDetectionResult(
                found=False,
                error_px=self.last_error,
                centroid_x=None,
                centroid_y=None,
                contour_area=contour_area,
                roi_top=roi_top,
                roi_bottom=roi_bottom,
                is_t_junction=False,
                mask=best_mask,
            )

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            self.lost_frames += 1
            return LineDetectionResult(
                found=False,
                error_px=self.last_error,
                centroid_x=None,
                centroid_y=None,
                contour_area=contour_area,
                roi_top=roi_top,
                roi_bottom=roi_bottom,
                is_t_junction=False,
                mask=best_mask,
            )

        # Use the lower portion of the contour to compute the steering centroid.
        # centroid_bottom_band_ratio controls how much lookahead is included:
        # 0.60 → bottom 40% only (no lookahead), 0.35 → bottom 65% (anticipates curves).
        roi_h = roi_bottom - roi_top
        bottom_band_top = int(roi_h * self.detector_config.centroid_bottom_band_ratio)
        contour_mask_full = np.zeros((roi_h, frame_w), dtype=np.uint8)
        cv2.drawContours(contour_mask_full, [contour], -1, 255, thickness=-1)
        bottom_band = contour_mask_full[bottom_band_top:, :]
        bottom_moments = cv2.moments(bottom_band)
        if bottom_moments["m00"] > 0:
            centroid_x = int(bottom_moments["m10"] / bottom_moments["m00"])
            centroid_y = int(bottom_moments["m01"] / bottom_moments["m00"]) + bottom_band_top + roi_top
        else:
            # Fallback to full-contour centroid
            centroid_x = int(moments["m10"] / moments["m00"])
            centroid_y = int(moments["m01"] / moments["m00"]) + roi_top
        error_px = float(centroid_x - (frame_w // 2) - self.detector_config.center_offset_px)
        if abs(error_px) < self.steering_config.deadband_px:
            error_px = 0.0

        self.last_error = error_px
        if error_px < 0.0:
            self.search_direction = -1
        elif error_px > 0.0:
            self.search_direction = 1
        self.last_centroid_x = centroid_x
        self.lost_frames = 0
        contour_global = contour.copy()
        contour_global[:, 0, 1] += roi_top
        active_mask = best_mask if best_mask is not None else mask
        is_t_junction = self._is_t_junction(active_mask, contour)

        # Wide-contour heuristic: if the line contour is much wider than a
        # single line, it likely spans a branching point (Y- or T-junction)
        # even if the strict T-shape check didn't fire.
        if not is_t_junction:
            _bx, _by, _bw, _bh = cv2.boundingRect(contour)
            if _bw > frame_w * 0.30 and contour_area > 600:
                is_t_junction = True

        # Only trigger junction if the contour reaches the bottom half of the
        # ROI — prevents early triggering when the junction is still far away.
        if is_t_junction:
            _jx, _jy, _jw, _jh = cv2.boundingRect(contour)
            roi_h = roi_bottom - roi_top
            contour_bottom_in_roi = _jy + _jh
            if contour_bottom_in_roi < roi_h * 0.50:
                is_t_junction = False

        return LineDetectionResult(
            found=True,
            error_px=error_px,
            centroid_x=centroid_x,
            centroid_y=centroid_y,
            contour_area=contour_area,
            roi_top=roi_top,
            roi_bottom=roi_bottom,
            is_t_junction=is_t_junction,
            contour=contour_global,
            mask=active_mask,
        )

    def compute_steering(self, detection: LineDetectionResult, preferred_turn: str | None = None) -> MotorCommand:
        def _apply_turn_inversion(turn_left: bool) -> bool:
            if self.steering_config.invert_turn_direction:
                return not turn_left
            return turn_left

        if not detection.found:
            latched_preferred_turn = (
                self._junction_latched
                and preferred_turn in {"LEFT", "RIGHT"}
            )
            if not latched_preferred_turn:
                self._junction_latched = False

            # Briefly keep moving forward when the line disappears to avoid
            # instant pivots on short gaps or corner occlusions.
            if self.lost_frames <= max(0, self.detector_config.lost_turn_delay_frames):
                coast_speed = self.steering_config.base_speed * clamp(
                    self.detector_config.lost_forward_speed_scale,
                    0.0,
                    1.0,
                )
                return MotorCommand(
                    left_speed=coast_speed,
                    right_speed=coast_speed,
                    correction=0.0,
                    recovery_mode=True,
                )

            turn_speed = self.steering_config.lost_turn_speed
            if latched_preferred_turn:
                preferred = preferred_turn
            else:
                preferred = (self.detector_config.recovery_preferred_turn or "").strip().upper()
            if preferred in {"NONE", "NULL", "AUTO"}:
                preferred = ""
            if preferred in {"LEFT", "RIGHT"}:
                turn_left = preferred == "LEFT"
            elif self.last_error < 0.0:
                turn_left = True
            elif self.last_error > 0.0:
                turn_left = False
            elif self.last_centroid_x is not None and self.last_frame_width > 0:
                turn_left = self.last_centroid_x < (self.last_frame_width // 2)
            else:
                turn_left = self.search_direction < 0
            turn_left = _apply_turn_inversion(turn_left)

            if (
                not latched_preferred_turn
                and
                self.detector_config.alternate_search_on_loss
                and self.lost_frames >= self.detector_config.lost_search_flip_frames
            ):
                turn_left = not turn_left

            if turn_left:
                return MotorCommand(
                    left_speed=-turn_speed,
                    right_speed=turn_speed,
                    correction=0.0,
                    recovery_mode=True,
                )
            return MotorCommand(
                left_speed=turn_speed,
                right_speed=-turn_speed,
                correction=0.0,
                recovery_mode=True,
            )

        if detection.is_t_junction and preferred_turn in {"LEFT", "RIGHT"}:
            self._junction_latched = True
            outer = max(
                self.steering_config.base_speed + 18.0,
                self.steering_config.tank_turn_speed * 0.84,
            )
            inner = -max(8.0, outer * 0.12)
            turn_left = _apply_turn_inversion(preferred_turn == "LEFT")
            if turn_left:
                return MotorCommand(
                    left_speed=inner,
                    right_speed=outer,
                    correction=0.0,
                    recovery_mode=True,
                )
            return MotorCommand(
                left_speed=outer,
                right_speed=inner,
                correction=0.0,
                recovery_mode=True,
            )

        if self._junction_latched and not detection.is_t_junction:
            self._junction_latched = False

        alpha = self.steering_config.error_filter_alpha
        self.last_filtered_error = self.filtered_error
        self.filtered_error = (alpha * self.filtered_error) + ((1.0 - alpha) * detection.error_px)

        # Sharp turns are handled with a confirmed, filtered error to reduce
        # wrong-side snap turns from one-frame contour jitter.
        if abs(self.filtered_error) >= self.steering_config.tank_turn_error_px:
            self._high_error_frames += 1
        else:
            self._high_error_frames = 0

        if self._high_error_frames >= max(1, self.detector_config.high_error_confirm_frames):
            turn = self.steering_config.tank_turn_speed
            if abs(self.filtered_error) >= self.steering_config.tank_turn_error_px * 1.35:
                inner = -max(10.0, turn * 0.18)
            else:
                inner = max(0.0, turn * 0.05)
            turn_left = self.filtered_error < 0.0
            turn_left = _apply_turn_inversion(turn_left)
            if turn_left:
                return MotorCommand(
                    left_speed=inner,
                    right_speed=turn,
                    correction=0.0,
                    recovery_mode=True,
                )
            return MotorCommand(
                left_speed=turn,
                right_speed=inner,
                correction=0.0,
                recovery_mode=True,
            )

        # D term: dampens overshoot on tight corners
        d_term = self.steering_config.d_gain * (self.filtered_error - self.last_filtered_error)
        target_correction = clamp(
            self.steering_config.steer_gain * self.filtered_error + d_term,
            -self.steering_config.max_adjust,
            self.steering_config.max_adjust,
        )
        correction_delta = target_correction - self.last_correction
        correction_delta = clamp(
            correction_delta,
            -self.steering_config.max_correction_step,
            self.steering_config.max_correction_step,
        )
        correction = self.last_correction + correction_delta
        self.last_correction = correction

        if self.steering_config.invert_turn_direction:
            correction = -correction

        # Adaptive speed: fast on straights, slow through tight turns
        base = self.steering_config.base_speed
        if self.steering_config.speed_adapt:
            err_mag = abs(self.filtered_error)
            lo = self.steering_config.speed_straight_threshold
            hi = self.steering_config.speed_curve_threshold
            if err_mag <= lo:
                base *= self.steering_config.speed_straight_scale
            elif err_mag >= hi:
                base *= self.steering_config.speed_curve_scale
            else:
                t = (err_mag - lo) / max(1.0, hi - lo)
                scale = (self.steering_config.speed_straight_scale
                         + t * (self.steering_config.speed_curve_scale
                                - self.steering_config.speed_straight_scale))
                base *= scale

        return MotorCommand(
            left_speed=base + correction,
            right_speed=base - correction,
            correction=correction,
            recovery_mode=False,
        )


def draw_line_debug(frame_bgr: np.ndarray, detection: LineDetectionResult) -> None:
    cv2.rectangle(
        frame_bgr,
        (0, detection.roi_top),
        (frame_bgr.shape[1] - 1, detection.roi_bottom),
        (255, 255, 0),
        1,
    )
    cv2.line(
        frame_bgr,
        (frame_bgr.shape[1] // 2, detection.roi_top),
        (frame_bgr.shape[1] // 2, detection.roi_bottom),
        (255, 0, 255),
        1,
    )
    if detection.contour is not None:
        # Draw a tight rotated rectangle around the detected line
        rect = cv2.minAreaRect(detection.contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(frame_bgr, [box], 0, (0, 255, 0), 2)
    if detection.centroid_x is not None and detection.centroid_y is not None:
        cv2.circle(
            frame_bgr,
            (detection.centroid_x, detection.centroid_y),
            6,
            (0, 0, 255),
            -1,
        )
