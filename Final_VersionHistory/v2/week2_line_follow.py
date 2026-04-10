from __future__ import annotations

from dataclasses import dataclass
import time

import cv2
import numpy as np


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class Week2LineFollowConfig:
    line_color: str = "BLACK"
    line_priority: tuple[str, ...] = ("BLACK", "RED", "YELLOW")
    roi_top_ratio: float = 140.0 / 240.0
    roi_bottom_ratio: float = 1.0
    roi_left_ratio: float = 0.0
    roi_right_ratio: float = 1.0
    use_otsu: bool = True
    threshold_value: int = 60
    min_line_pixels: int = 150
    scan_rows: int = 16
    min_valid_rows: int = 6
    min_row_pixels: int = 4
    center_offset_px: float = 0.0
    blur_kernel: tuple[int, int] = (5, 5)
    morph_kernel_size: int = 3
    base_speed: float = 45.0
    min_speed: float = 35.0
    max_drive_speed: float = 75.0
    lost_forward_speed: float = 31.5
    stall_pwm: float = 30.0
    kp: float = 0.40
    ki: float = 0.003
    kd: float = 0.08
    i_limit: float = 80.0
    derivative_alpha: float = 0.75
    deadband_px: float = 4.0
    max_correction: float = 40.0
    max_correction_slew_per_sec: float = 200.0
    pivot_threshold_px: float = 35.0
    pivot_speed: float = 65.0
    pivot_exit_px: float = 15.0
    pivot_timeout_s: float = 0.6
    pivot_brake_ms: float = 40.0
    lost_timeout_s: float = 0.0
    search_speed: float = 50.0
    swing_pivot: bool = True
    pivot_reverse_scale: float = 1.0
    bottom_contact_band_ratio: float = 0.18
    bottom_contact_min_cols_ratio: float = 0.018
    min_black_width_px: int = 24
    min_black_support_rows: int = 20
    black_support_min_cols: int = 6
    min_black_contrast_delta: float = 24.0
    black_contrast_ring_px: int = 7
    min_elongation: float = 1.4
    priority_erode_iterations: int = 2
    priority_min_contour_area: float = 120.0
    priority_scan_start_ratio: float = 0.35
    priority_side_margin_px: int = 12
    priority_side_penalty: float = 900.0
    priority_continuity_weight: float = 6.0
    priority_color_min_contour_area: float = 220.0
    priority_color_bottom_contact_ratio: float = 0.010
    black_v_multiplier: float = 0.62
    black_bgr_upper: int = 130
    black_neutral_delta: int = 45
    blue_bgr_min: int = 40
    blue_dom_min_diff: int = 10
    blue_lab_blue_delta: int = 12
    blue_min_saturation: int = 28
    blue_white_v_min: int = 120
    blue_white_s_max: int = 42
    blue_white_channel_span_max: int = 24
    contour_continuity_weight: float = 8.0
    target_smoothing_alpha: float = 0.36
    target_jump_px: float = 44.0
    target_jump_alpha: float = 0.14
    confirm_frames: int = 2
    reliable_rows_bonus: int = 2
    reliable_area_scale: float = 2.0
    reliable_jump_px: float = 24.0
    confirm_confidence: float = 0.62


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
    valid_rows: int = 0
    confidence: float = 0.0
    confirmed: bool = False


@dataclass
class MotorCommand:
    left_speed: float
    right_speed: float
    correction: float
    recovery_mode: bool = False


class Week2LineFollower:
    """Week2-style line follower: row scan centerline + PID + pivot/search recovery."""

    def __init__(self, config: Week2LineFollowConfig | None = None) -> None:
        self.config = config or Week2LineFollowConfig()
        self._integral = 0.0
        self._prev_error = 0.0
        self._d_filtered = 0.0
        self._last_time = time.monotonic()
        self._last_seen = self._last_time
        self._last_error_sign = 1.0
        self._last_correction = 0.0
        self._pivot_direction = 0.0
        self._pivot_started_at = 0.0
        self._brake_until = 0.0
        self._roi_width = 320.0
        self._last_target_x: float | None = None
        self._smoothed_target_x: float | None = None
        self._confirmed_frames = 0

    def reset_steering(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0
        self._d_filtered = 0.0
        self._last_correction = 0.0
        self._pivot_direction = 0.0
        self._pivot_started_at = 0.0
        self._brake_until = 0.0
        self._last_time = time.monotonic()
        self._last_target_x = None
        self._smoothed_target_x = None
        self._confirmed_frames = 0

    def _reset_pid(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0
        self._d_filtered = 0.0
        self._last_correction = 0.0

    def _line_confidence(
        self,
        center_x: float,
        contour_area: float,
        valid_rows: int,
    ) -> float:
        cfg = self.config
        rows_target = max(1, cfg.min_valid_rows + max(0, cfg.reliable_rows_bonus))
        rows_score = clamp(valid_rows / rows_target, 0.0, 1.0)

        area_target = max(1.0, cfg.min_line_pixels * cfg.reliable_area_scale)
        area_score = clamp(contour_area / area_target, 0.0, 1.0)

        if self._last_target_x is None:
            continuity_score = 0.55
        else:
            continuity_score = clamp(
                1.0 - (
                    abs(center_x - self._last_target_x)
                    / max(1.0, cfg.reliable_jump_px)
                ),
                0.0,
                1.0,
            )

        return float(
            (rows_score * 0.42)
            + (area_score * 0.38)
            + (continuity_score * 0.20)
        )

    def _confirm_line(
        self,
        center_x: float,
        contour_area: float,
        valid_rows: int,
    ) -> tuple[bool, float]:
        cfg = self.config
        confidence = self._line_confidence(center_x, contour_area, valid_rows)
        continuity_good = (
            self._last_target_x is not None
            and abs(center_x - self._last_target_x) <= cfg.reliable_jump_px
        )
        reliable_now = confidence >= cfg.confirm_confidence

        if reliable_now or continuity_good:
            if continuity_good or confidence >= 0.82:
                self._confirmed_frames = max(1, self._confirmed_frames + 1)
            else:
                self._confirmed_frames += 1
        else:
            self._confirmed_frames = 0

        confirmed = continuity_good or self._confirmed_frames >= max(1, cfg.confirm_frames)
        return confirmed, confidence

    def _filtered_roi(self, roi_bgr: np.ndarray) -> np.ndarray:
        kx, ky = self.config.blur_kernel
        if kx < 3 or ky < 3:
            return roi_bgr
        return cv2.GaussianBlur(roi_bgr, self.config.blur_kernel, 0)

    def _stabilize_target_x(
        self,
        center_x: float,
        contour_area: float,
        valid_rows: int,
    ) -> float:
        cfg = self.config
        if self._smoothed_target_x is None:
            self._smoothed_target_x = center_x
            return center_x

        alpha = float(clamp(cfg.target_smoothing_alpha, 0.05, 1.0))
        jump_alpha = float(clamp(cfg.target_jump_alpha, 0.04, 0.8))
        jump_px = float(clamp(cfg.target_jump_px, 8.0, 160.0))

        jump = abs(center_x - self._smoothed_target_x)
        weak_rows = valid_rows <= max(cfg.min_valid_rows, 3)
        weak_area = contour_area < (cfg.min_line_pixels * 3.0)

        if jump > jump_px:
            if weak_rows or weak_area:
                alpha = min(alpha, jump_alpha)
            else:
                alpha = max(alpha, 0.55)

        smoothed = (
            ((1.0 - alpha) * self._smoothed_target_x)
            + (alpha * center_x)
        )
        self._smoothed_target_x = smoothed
        return smoothed

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
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size),
        )
        dilated = cv2.dilate(contour_mask, kernel, iterations=1)
        ring_mask = cv2.bitwise_and(dilated, cv2.bitwise_not(contour_mask))
        ring_vals = gray[ring_mask > 0]
        if ring_vals.size < 12:
            return 0.0

        return float(np.mean(ring_vals) - np.mean(inside_vals))

    def _select_best_contour(
        self,
        mask: np.ndarray,
        gray: np.ndarray | None,
    ) -> tuple[np.ndarray | None, float]:
        cfg = self.config
        color = cfg.line_color.strip().upper()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0.0

        best_contour = None
        best_score = -1.0
        best_area = 0.0
        min_contact_cols = int(mask.shape[1] * cfg.bottom_contact_min_cols_ratio)

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area <= 0.0:
                continue

            contact_cols = self._bottom_contact_cols(
                mask, contour, cfg.bottom_contact_band_ratio,
            )
            has_bottom_contact = contact_cols >= max(1, min_contact_cols)
            if not has_bottom_contact and area < 800.0:
                continue

            bx, by, bw, bh = cv2.boundingRect(contour)
            elongation = max(bw, bh) / max(1.0, min(bw, bh))

            if color == "BLACK":
                if elongation < cfg.min_elongation and area < 2000.0:
                    continue

                support_rows = self._support_rows(
                    mask, contour, cfg.black_support_min_cols,
                )
                if support_rows < cfg.min_black_support_rows:
                    continue

                if bw > mask.shape[1] * 0.45 and bh < max(10, int(mask.shape[0] * 0.10)):
                    continue

                contrast_delta = self._contour_contrast(
                    gray, contour, cfg.black_contrast_ring_px,
                )
                strong_bottom_contact = contact_cols >= max(
                    14, int(mask.shape[1] * 0.035),
                )
                tall_supported_contour = support_rows >= max(
                    cfg.min_black_support_rows + 6,
                    int(mask.shape[0] * 0.22),
                )
                if (
                    contrast_delta < cfg.min_black_contrast_delta
                    and not (strong_bottom_contact and tall_supported_contour)
                ):
                    continue

                rot_rect = cv2.minAreaRect(contour)
                true_min_dim = min(rot_rect[1])
                if true_min_dim < cfg.min_black_width_px:
                    continue
            else:
                if elongation < 1.2 and area < 900.0:
                    continue

            moments = cv2.moments(contour)
            if moments["m00"] > 0:
                cx = float(moments["m10"] / moments["m00"])
            else:
                cx = float(bx + (bw / 2.0))

            continuity_penalty = 0.0
            if self._last_target_x is not None:
                continuity_penalty = (
                    abs(cx - self._last_target_x)
                    * self.config.contour_continuity_weight
                )

            score = (contact_cols * 1000.0) + area - continuity_penalty
            if best_contour is None or score > best_score:
                best_contour = contour
                best_score = score
                best_area = area

        return best_contour, best_area

    @staticmethod
    def _largest_contour(mask: np.ndarray) -> tuple[np.ndarray | None, float]:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0.0
        contour = max(contours, key=cv2.contourArea)
        return contour, float(cv2.contourArea(contour))

    def _best_priority_contour(self, mask: np.ndarray) -> tuple[np.ndarray | None, float]:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0.0

        best_contour = None
        best_area = 0.0
        best_score = -1.0
        roi_cx = mask.shape[1] / 2.0
        side_margin = max(0, int(self.config.priority_side_margin_px))
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.config.priority_min_contour_area:
                continue
            contact_cols = self._bottom_contact_cols(
                mask, contour, self.config.bottom_contact_band_ratio,
            )
            moments = cv2.moments(contour)
            x, _, w, _ = cv2.boundingRect(contour)
            if moments["m00"] > 0:
                cx = float(moments["m10"] / moments["m00"])
            else:
                cx = float(x + (w / 2.0))
            center_penalty = abs(cx - roi_cx) * 4.0
            continuity_penalty = 0.0
            if self._last_target_x is not None:
                continuity_penalty = (
                    abs(cx - self._last_target_x)
                    * self.config.priority_continuity_weight
                )

            touches_left = x <= side_margin
            touches_right = (x + w) >= (mask.shape[1] - side_margin)
            side_penalty = 0.0
            if touches_left or touches_right:
                side_penalty = self.config.priority_side_penalty
                if self._last_target_x is not None:
                    # Keep a real curve candidate alive if it continues from the
                    # previous frame instead of being a sudden border jump.
                    side_penalty *= 0.55

            score = (
                (contact_cols * 1200.0)
                + area
                - center_penalty
                - continuity_penalty
                - side_penalty
            )
            if best_contour is None or score > best_score:
                best_contour = contour
                best_area = area
                best_score = score
        return best_contour, best_area

    @staticmethod
    def _scan_mask_centers(
        mask: np.ndarray,
        scan_rows: int,
        min_row_pixels: int,
    ) -> tuple[list[float], list[int]]:
        ys = np.linspace(mask.shape[0] - 1, 0, scan_rows, dtype=int)
        centers: list[float] = []
        valid_ys: list[int] = []
        for row_y in ys:
            xs = np.where(mask[row_y] > 0)[0]
            if xs.size >= min_row_pixels:
                centers.append(float(np.mean(xs)))
                valid_ys.append(int(row_y))
        return centers, valid_ys

    def _roi_bounds(self, frame_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
        frame_h, frame_w = frame_shape[:2]
        left = int(frame_w * self.config.roi_left_ratio)
        right = int(frame_w * self.config.roi_right_ratio)
        top = int(frame_h * self.config.roi_top_ratio)
        bottom = int(frame_h * self.config.roi_bottom_ratio)
        left = int(clamp(left, 0, frame_w - 1))
        right = int(clamp(right, left + 1, frame_w))
        top = int(clamp(top, 0, frame_h - 1))
        bottom = int(clamp(bottom, top + 1, frame_h))
        return top, bottom, left, right

    def _build_mask(self, roi_bgr: np.ndarray) -> np.ndarray:
        cfg = self.config
        color = cfg.line_color.strip().upper()
        roi_bgr = self._filtered_roi(roi_bgr)

        if color == "BLACK":
            mask = self._build_black_mask(roi_bgr)
        elif color == "BLUE":
            hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            ranges = (
                ((88, 40, 30), (138, 255, 255)),
                ((76, 28, 25), (96, 255, 255)),
                ((130, 36, 24), (148, 255, 255)),
            )
            for lower, upper in ranges:
                mask = cv2.bitwise_or(
                    mask,
                    cv2.inRange(
                        hsv,
                        np.array(lower, dtype=np.uint8),
                        np.array(upper, dtype=np.uint8),
                    ),
                )

            b = roi_bgr[:, :, 0].astype(np.int16)
            g = roi_bgr[:, :, 1].astype(np.int16)
            r = roi_bgr[:, :, 2].astype(np.int16)
            v = hsv[:, :, 2].astype(np.int16)
            s = hsv[:, :, 1].astype(np.int16)
            rgb_max = np.maximum(np.maximum(b, g), r)
            rgb_min = np.minimum(np.minimum(b, g), r)
            rgb_span = rgb_max - rgb_min
            blue_bgr_min = int(clamp(cfg.blue_bgr_min, 16, 90))
            blue_dom_min = int(clamp(cfg.blue_dom_min_diff, 2, 30))
            blue_min_s = int(clamp(cfg.blue_min_saturation, 8, 120))
            white_v_min = int(clamp(cfg.blue_white_v_min, 70, 220))
            white_s_max = int(clamp(cfg.blue_white_s_max, 8, 90))
            white_span_max = int(clamp(cfg.blue_white_channel_span_max, 6, 50))
            whiteish = (
                (v >= white_v_min)
                & (s <= white_s_max)
                & (rgb_span <= white_span_max)
            )
            blue_dom = (
                (
                    (b >= blue_bgr_min)
                    & (b >= g + blue_dom_min)
                    & (b >= r + blue_dom_min + 5)
                    & (s >= blue_min_s)
                )
                | (
                    (b >= max(20, blue_bgr_min - 8))
                    & (b * 10 >= g * 11)
                    & (b * 10 >= r * 13)
                    & (b - r >= max(6, blue_dom_min))
                    & (s >= max(blue_min_s - 6, 18))
                )
            )
            mask = cv2.bitwise_or(
                mask,
                np.where(blue_dom, 255, 0).astype(np.uint8),
            )

            lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
            lab_b = lab[:, :, 2].astype(np.int16)
            median_lab_b = int(np.median(lab_b))
            blue_lab_upper = int(
                clamp(
                    min(median_lab_b - cfg.blue_lab_blue_delta, 124),
                    96,
                    124,
                ),
            )
            lab_blue = (
                (lab_b <= blue_lab_upper)
                & (b >= max(18, blue_bgr_min - 8))
                & (b >= g + max(2, blue_dom_min - 3))
                & (b >= r + max(4, blue_dom_min - 1))
                & (s >= max(blue_min_s - 4, 18))
                & (rgb_span >= max(10, white_span_max - 4))
            )
            mask = cv2.bitwise_or(
                mask,
                np.where(lab_blue, 255, 0).astype(np.uint8),
            )
            mask = cv2.bitwise_and(
                mask,
                cv2.bitwise_not(np.where(whiteish, 255, 0).astype(np.uint8)),
            )
        else:
            raise ValueError(f"Unsupported line color: {cfg.line_color}")

        kernel = np.ones(
            (cfg.morph_kernel_size, cfg.morph_kernel_size), dtype=np.uint8,
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask

    def _build_black_mask(
        self,
        roi_bgr: np.ndarray,
        hsv: np.ndarray | None = None,
    ) -> np.ndarray:
        cfg = self.config
        if hsv is None:
            hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

        median_v = float(np.median(hsv[:, :, 2]))
        black_v_upper = int(clamp(median_v * cfg.black_v_multiplier, 55, 135))
        black_s_upper = 110 if median_v > 140 else 90
        hsv_mask = cv2.inRange(
            hsv,
            np.array((0, 0, 0), dtype=np.uint8),
            np.array((180, black_s_upper, black_v_upper), dtype=np.uint8),
        )

        bgr_upper = int(clamp(cfg.black_bgr_upper, 60, 170))
        bgr_dark_mask = cv2.inRange(
            roi_bgr,
            np.array((0, 0, 0), dtype=np.uint8),
            np.array((bgr_upper, bgr_upper, bgr_upper), dtype=np.uint8),
        )

        b = roi_bgr[:, :, 0].astype(np.int16)
        g = roi_bgr[:, :, 1].astype(np.int16)
        r = roi_bgr[:, :, 2].astype(np.int16)
        rgb_max = np.maximum(np.maximum(b, g), r)
        rgb_min = np.minimum(np.minimum(b, g), r)
        neutral_dark = (
            (rgb_max - rgb_min) <= int(clamp(cfg.black_neutral_delta, 10, 90))
        )
        neutral_dark_mask = np.where(neutral_dark, 255, 0).astype(np.uint8)

        mask = cv2.bitwise_or(hsv_mask, cv2.bitwise_and(bgr_dark_mask, neutral_dark_mask))
        return mask

    def _detect_priority_line(
        self,
        roi_bgr: np.ndarray,
        roi_top: int,
        roi_left: int,
    ) -> LineDetectionResult:
        cfg = self.config
        roi_filtered = self._filtered_roi(roi_bgr)
        hsv = cv2.cvtColor(roi_filtered, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(roi_filtered, cv2.COLOR_BGR2GRAY)
        kernel = np.ones(
            (cfg.morph_kernel_size, cfg.morph_kernel_size), dtype=np.uint8,
        )

        red_mask_1 = cv2.inRange(hsv, (136, 80, 80), (180, 255, 255))
        red_mask_2 = cv2.inRange(hsv, (0, 80, 80), (15, 255, 255))
        red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)
        yellow_mask = cv2.inRange(hsv, (15, 100, 100), (35, 255, 255))
        black_mask = self._build_black_mask(roi_filtered, hsv=hsv)

        masks = {
            "RED": red_mask,
            "YELLOW": yellow_mask,
            "BLACK": black_mask,
        }
        for key in masks:
            masks[key] = cv2.erode(
                masks[key],
                kernel,
                iterations=max(0, int(cfg.priority_erode_iterations)),
            )
            masks[key] = cv2.morphologyEx(
                masks[key], cv2.MORPH_OPEN, kernel, iterations=1,
            )
            masks[key] = cv2.morphologyEx(
                masks[key], cv2.MORPH_CLOSE, kernel, iterations=2,
            )

        chosen_mask = None
        contour = None
        contour_area = 0.0
        for token in cfg.line_priority:
            key = token.strip().upper()
            mask = masks.get(key)
            if mask is None:
                continue
            if key == "BLACK":
                candidate, area = self._select_best_contour(mask, gray)
            else:
                candidate, area = self._best_priority_contour(mask)
            if candidate is not None and area > 0.0:
                if key in {"RED", "YELLOW"}:
                    min_contact_cols = int(
                        mask.shape[1] * cfg.priority_color_bottom_contact_ratio
                    )
                    contact_cols = self._bottom_contact_cols(
                        mask, candidate, cfg.bottom_contact_band_ratio,
                    )
                    if (
                        area < cfg.priority_color_min_contour_area
                        or contact_cols < max(3, min_contact_cols)
                    ):
                        continue
                chosen_mask = mask
                contour = candidate
                contour_area = area
                break

        if chosen_mask is None or contour is None:
            self._confirmed_frames = 0
            return LineDetectionResult(
                found=False,
                error_px=0.0,
                centroid_x=None,
                centroid_y=None,
                contour_area=0.0,
                roi_top=roi_top,
                roi_bottom=roi_top + roi_bgr.shape[0],
                mask=cv2.bitwise_or(cv2.bitwise_or(red_mask, yellow_mask), black_mask),
                valid_rows=0,
                confidence=0.0,
                confirmed=False,
            )

        contour_global = contour.copy()
        contour_global[:, 0, 0] += roi_left
        contour_global[:, 0, 1] += roi_top

        selected_mask = np.zeros_like(chosen_mask)
        cv2.drawContours(selected_mask, [contour], -1, 255, thickness=-1)
        scan_start = int(
            selected_mask.shape[0]
            * clamp(cfg.priority_scan_start_ratio, 0.0, 0.90)
        )
        scan_mask = selected_mask[scan_start:, :]
        centers, valid_ys = self._scan_mask_centers(
            scan_mask, cfg.scan_rows, cfg.min_row_pixels,
        )
        valid_ys = [y + scan_start for y in valid_ys]
        valid_rows = len(centers)

        found = (
            int(cv2.countNonZero(selected_mask)) >= cfg.min_line_pixels
            and valid_rows >= max(1, cfg.min_valid_rows)
        )
        moments = cv2.moments(contour)
        if moments["m00"] > 0:
            contour_cx = int(moments["m10"] / moments["m00"]) + roi_left
            contour_cy = int(moments["m01"] / moments["m00"]) + roi_top
        else:
            x, y, w, h = cv2.boundingRect(contour)
            contour_cx = int(x + (w / 2.0)) + roi_left
            contour_cy = int(y + (h / 2.0)) + roi_top

        if valid_rows > 0:
            raw_center_x = float(np.median(centers))
            center_y = int(np.max(valid_ys)) + roi_top
        else:
            raw_center_x = float(contour_cx - roi_left)
            center_y = contour_cy

        confirmed, confidence = self._confirm_line(
            raw_center_x, contour_area, valid_rows,
        )
        if confirmed:
            center_x = self._stabilize_target_x(
                raw_center_x, contour_area, valid_rows,
            )
        else:
            center_x = raw_center_x

        target_cx = (roi_bgr.shape[1] / 2.0) + cfg.center_offset_px
        error_px = center_x - target_cx
        if abs(error_px) < cfg.deadband_px:
            error_px = 0.0

        found = found and confirmed
        if found:
            self._last_target_x = center_x

        return LineDetectionResult(
            found=found,
            error_px=error_px if found else 0.0,
            centroid_x=int(center_x) + roi_left,
            centroid_y=center_y,
            contour_area=contour_area,
            roi_top=roi_top,
            roi_bottom=roi_top + roi_bgr.shape[0],
            contour=contour_global,
            mask=selected_mask,
            valid_rows=valid_rows,
            confidence=confidence,
            confirmed=confirmed,
        )

    def detect_line(self, frame_bgr: np.ndarray) -> LineDetectionResult:
        roi_top, roi_bottom, roi_left, roi_right = self._roi_bounds(frame_bgr.shape)
        roi = frame_bgr[roi_top:roi_bottom, roi_left:roi_right]
        if self.config.line_color.strip().upper() == "PRIORITY":
            self._roi_width = float(max(1, roi_right - roi_left))
            return self._detect_priority_line(roi, roi_top, roi_left)
        roi_filtered = self._filtered_roi(roi)
        gray = cv2.cvtColor(roi_filtered, cv2.COLOR_BGR2GRAY)
        mask = self._build_mask(roi_filtered)
        self._roi_width = float(max(1, roi_right - roi_left))

        contour, contour_area = self._select_best_contour(mask, gray)
        selected_mask = np.zeros_like(mask)
        if contour is not None:
            cv2.drawContours(selected_mask, [contour], -1, 255, thickness=-1)
        total_pixels = int(cv2.countNonZero(selected_mask))

        if total_pixels < self.config.min_line_pixels:
            self._confirmed_frames = 0
            contour_global = None
            if contour is not None:
                contour_global = contour.copy()
                contour_global[:, 0, 0] += roi_left
                contour_global[:, 0, 1] += roi_top
            return LineDetectionResult(
                found=False,
                error_px=0.0,
                centroid_x=None,
                centroid_y=None,
                contour_area=contour_area,
                roi_top=roi_top,
                roi_bottom=roi_bottom,
                contour=contour_global,
                mask=mask,
                valid_rows=0,
                confidence=0.0,
                confirmed=False,
            )

        ys = np.linspace(selected_mask.shape[0] - 1, 0, self.config.scan_rows, dtype=int)
        centers: list[float] = []
        valid_ys: list[int] = []
        for y in ys:
            xs = np.where(selected_mask[y] > 0)[0]
            if xs.size >= self.config.min_row_pixels:
                centers.append(float(np.mean(xs)))
                valid_ys.append(int(y))

        contour_global = None
        contour_cx = None
        contour_cy = None
        if contour is not None:
            contour_global = contour.copy()
            contour_global[:, 0, 0] += roi_left
            contour_global[:, 0, 1] += roi_top
            moments = cv2.moments(contour)
            if moments["m00"] > 0:
                contour_cx = int(moments["m10"] / moments["m00"]) + roi_left
                contour_cy = int(moments["m01"] / moments["m00"]) + roi_top

        if len(centers) < self.config.min_valid_rows:
            self._confirmed_frames = 0
            return LineDetectionResult(
                found=False,
                error_px=0.0,
                centroid_x=contour_cx,
                centroid_y=contour_cy,
                contour_area=contour_area,
                roi_top=roi_top,
                roi_bottom=roi_bottom,
                contour=contour_global,
                mask=selected_mask,
                valid_rows=len(centers),
                confidence=0.0,
                confirmed=False,
            )

        raw_center_x = float(np.median(centers))
        confirmed, confidence = self._confirm_line(
            raw_center_x,
            max(float(total_pixels), contour_area),
            len(centers),
        )
        if confirmed:
            center_x = self._stabilize_target_x(
                raw_center_x, contour_area, len(centers),
            )
        else:
            center_x = raw_center_x
        centroid_x = int(center_x) + roi_left
        centroid_y = contour_cy if contour_cy is not None else int(np.max(valid_ys)) + roi_top
        target_cx = (self._roi_width / 2.0) + self.config.center_offset_px
        error_px = center_x - target_cx
        if abs(error_px) < self.config.deadband_px:
            error_px = 0.0

        if confirmed:
            self._last_target_x = center_x

        return LineDetectionResult(
            found=confirmed,
            error_px=error_px if confirmed else 0.0,
            centroid_x=centroid_x,
            centroid_y=centroid_y,
            contour_area=max(float(total_pixels), contour_area),
            roi_top=roi_top,
            roi_bottom=roi_bottom,
            contour=contour_global,
            mask=selected_mask,
            valid_rows=len(centers),
            confidence=confidence,
            confirmed=confirmed,
        )

    def _week2_error_scale(self) -> float:
        return 320.0 / max(1.0, self._roi_width)

    def _pivot_command(self) -> MotorCommand:
        speed = self.config.pivot_speed
        reverse_speed = -max(
            self.config.stall_pwm + 8.0,
            speed * self.config.pivot_reverse_scale,
        )
        if self._pivot_direction > 0.0:
            if self.config.swing_pivot:
                return MotorCommand(
                    left_speed=speed,
                    right_speed=0.0,
                    correction=0.0,
                    recovery_mode=True,
                )
            return MotorCommand(
                left_speed=speed,
                right_speed=reverse_speed,
                correction=0.0,
                recovery_mode=True,
            )
        if self.config.swing_pivot:
            return MotorCommand(
                left_speed=0.0,
                right_speed=speed,
                correction=0.0,
                recovery_mode=True,
                )
        return MotorCommand(
            left_speed=reverse_speed,
            right_speed=speed,
            correction=0.0,
            recovery_mode=True,
        )

    def compute_steering(
        self,
        detection: LineDetectionResult,
        preferred_turn: str | None = None,
    ) -> MotorCommand:
        now = time.monotonic()
        dt = max(now - self._last_time, 1e-3)
        self._last_time = now

        scale = self._week2_error_scale()
        preferred = (preferred_turn or "").strip().upper()

        if self.config.line_color.strip().upper() == "PRIORITY":
            if now < self._brake_until:
                return MotorCommand(
                    left_speed=0.0,
                    right_speed=0.0,
                    correction=0.0,
                    recovery_mode=True,
                )

            if self._pivot_direction != 0.0:
                aligned = (
                    detection.found
                    and abs(detection.error_px * scale) < self.config.pivot_exit_px
                )
                timed_out = (
                    (now - self._pivot_started_at) >= self.config.pivot_timeout_s
                )
                if aligned or timed_out:
                    self._pivot_direction = 0.0
                    self._pivot_started_at = 0.0
                    self._reset_pid()
                    self._brake_until = now + (
                        self.config.pivot_brake_ms / 1000.0
                    )
                    return MotorCommand(
                        left_speed=0.0,
                        right_speed=0.0,
                        correction=0.0,
                        recovery_mode=True,
                    )
                return self._pivot_command()

            if detection.found:
                self._last_seen = now
                control_error = detection.error_px * scale
                if abs(control_error) > 1.0:
                    self._last_error_sign = 1.0 if control_error > 0 else -1.0

                should_pivot = abs(control_error) >= self.config.pivot_threshold_px
                if preferred in {"LEFT", "RIGHT"}:
                    preferred_sign = -1.0 if preferred == "LEFT" else 1.0
                    early_pivot_threshold = max(
                        8.0, self.config.pivot_threshold_px * 0.55,
                    )
                    if (
                        abs(control_error) >= early_pivot_threshold
                        and (control_error * preferred_sign) >= -6.0
                    ):
                        should_pivot = True
                else:
                    preferred_sign = 0.0

                if should_pivot:
                    if preferred_sign != 0.0:
                        self._pivot_direction = preferred_sign
                    else:
                        self._pivot_direction = self._last_error_sign
                    self._pivot_started_at = now
                    self._reset_pid()
                    return self._pivot_command()

                self._integral += control_error * dt
                self._integral = clamp(
                    self._integral,
                    -self.config.i_limit,
                    self.config.i_limit,
                )
                d_raw = (control_error - self._prev_error) / dt if dt > 0.0 else 0.0
                self._d_filtered = (
                    self.config.derivative_alpha * self._d_filtered
                    + (1.0 - self.config.derivative_alpha) * d_raw
                )
                self._prev_error = control_error

                correction = (
                    self.config.kp * control_error
                    + self.config.ki * self._integral
                    + self.config.kd * self._d_filtered
                )
                correction = clamp(
                    correction,
                    -self.config.max_correction,
                    self.config.max_correction,
                )
                max_step = self.config.max_correction_slew_per_sec * dt
                correction = clamp(
                    correction,
                    self._last_correction - max_step,
                    self._last_correction + max_step,
                )
                self._last_correction = correction

                left_speed = clamp(
                    self.config.base_speed + correction,
                    self.config.min_speed,
                    self.config.max_drive_speed,
                )
                right_speed = clamp(
                    self.config.base_speed - correction,
                    self.config.min_speed,
                    self.config.max_drive_speed,
                )
                return MotorCommand(
                    left_speed=left_speed,
                    right_speed=right_speed,
                    correction=correction,
                    recovery_mode=False,
                )

            self._integral = 0.0
            self._prev_error = 0.0
            recent_lost = (
                self.config.lost_timeout_s > 0.0
                and (now - self._last_seen) < self.config.lost_timeout_s
            )
            if recent_lost:
                carry = clamp(
                    self._last_correction * 0.28,
                    -self.config.max_correction * 0.22,
                    self.config.max_correction * 0.22,
                )
                forward = max(self.config.lost_forward_speed, self.config.stall_pwm)
                return MotorCommand(
                    left_speed=clamp(
                        forward + carry,
                        0.0,
                        self.config.max_drive_speed,
                    ),
                    right_speed=clamp(
                        forward - carry,
                        0.0,
                        self.config.max_drive_speed,
                    ),
                    correction=carry,
                    recovery_mode=True,
                )

            self._last_correction = 0.0
            if preferred == "LEFT":
                search_sign = -1.0
            elif preferred == "RIGHT":
                search_sign = 1.0
            else:
                search_sign = self._last_error_sign
            spin = self.config.search_speed * search_sign
            return MotorCommand(
                left_speed=spin,
                right_speed=-spin,
                correction=0.0,
                recovery_mode=True,
            )

        if now < self._brake_until:
            return MotorCommand(
                left_speed=0.0,
                right_speed=0.0,
                correction=0.0,
                recovery_mode=True,
            )

        if self._pivot_direction != 0.0:
            aligned = (
                detection.found
                and abs(detection.error_px * scale) < self.config.pivot_exit_px
            )
            timed_out = (now - self._pivot_started_at) >= self.config.pivot_timeout_s
            if aligned or timed_out:
                self._pivot_direction = 0.0
                self._pivot_started_at = 0.0
                self._reset_pid()
                self._brake_until = now + (self.config.pivot_brake_ms / 1000.0)
                return MotorCommand(
                    left_speed=0.0,
                    right_speed=0.0,
                    correction=0.0,
                    recovery_mode=True,
                )
            return self._pivot_command()

        if detection.found:
            self._last_seen = now
            control_error = detection.error_px * scale
            if abs(control_error) > 1.0:
                self._last_error_sign = 1.0 if control_error > 0 else -1.0

            if abs(control_error) >= self.config.pivot_threshold_px:
                if preferred == "LEFT":
                    self._pivot_direction = -1.0
                elif preferred == "RIGHT":
                    self._pivot_direction = 1.0
                else:
                    self._pivot_direction = self._last_error_sign
                self._pivot_started_at = now
                self._reset_pid()
                return self._pivot_command()

            self._integral += control_error * dt
            self._integral = clamp(
                self._integral,
                -self.config.i_limit,
                self.config.i_limit,
            )
            d_raw = (control_error - self._prev_error) / dt
            self._d_filtered = (
                self.config.derivative_alpha * self._d_filtered
                + (1.0 - self.config.derivative_alpha) * d_raw
            )
            self._prev_error = control_error

            correction = (
                self.config.kp * control_error
                + self.config.ki * self._integral
                + self.config.kd * self._d_filtered
            )
            correction = clamp(
                correction,
                -self.config.max_correction,
                self.config.max_correction,
            )

            max_step = self.config.max_correction_slew_per_sec * dt
            correction = clamp(
                correction,
                self._last_correction - max_step,
                self._last_correction + max_step,
            )
            self._last_correction = correction

            turn_ratio = min(
                1.0,
                abs(correction) / max(1.0, self.config.max_correction),
            )
            forward = (
                self.config.base_speed
                - (self.config.base_speed - self.config.min_speed) * turn_ratio
            )
            forward = max(
                forward,
                self.config.stall_pwm + abs(correction) + 2.0,
            )
            forward = min(forward, self.config.base_speed)
            return MotorCommand(
                left_speed=forward + correction,
                right_speed=forward - correction,
                correction=correction,
                recovery_mode=False,
            )

        self._reset_pid()
        if preferred == "LEFT":
            search_sign = -1.0
        elif preferred == "RIGHT":
            search_sign = 1.0
        else:
            search_sign = self._last_error_sign

        if (now - self._last_seen) >= self.config.lost_timeout_s:
            spin = self.config.search_speed * search_sign
            return MotorCommand(
                left_speed=spin,
                right_speed=-spin,
                correction=0.0,
                recovery_mode=True,
            )
        return MotorCommand(
            left_speed=self.config.lost_forward_speed,
            right_speed=self.config.lost_forward_speed,
            correction=0.0,
            recovery_mode=True,
        )


def draw_line_debug(frame_bgr: np.ndarray, detection: LineDetectionResult) -> None:
    cv2.rectangle(
        frame_bgr,
        (0, detection.roi_top),
        (frame_bgr.shape[1] - 1, detection.roi_bottom),
        (255, 200, 0),
        1,
    )
    cv2.line(
        frame_bgr,
        (frame_bgr.shape[1] // 2, detection.roi_top),
        (frame_bgr.shape[1] // 2, detection.roi_bottom),
        (0, 0, 255),
        1,
    )
    if detection.contour is not None:
        cv2.drawContours(frame_bgr, [detection.contour], -1, (0, 255, 0), 2)
    if detection.centroid_x is not None and detection.centroid_y is not None:
        cv2.circle(
            frame_bgr,
            (detection.centroid_x, detection.centroid_y),
            6,
            (0, 255, 255),
            -1,
        )
