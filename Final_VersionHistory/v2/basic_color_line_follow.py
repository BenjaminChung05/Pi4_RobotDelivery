from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class BasicColorLineFollowConfig:
    line_color: str = "BLACK"
    roi_top_ratio: float = 0.48
    roi_bottom_ratio: float = 1.0
    blur_kernel: tuple[int, int] = (5, 5)
    morph_kernel_size: int = 5
    min_contour_area: float = 160.0
    min_bottom_contact_cols: int = 10
    center_offset_px: float = 0.0
    base_speed: float = 36.0
    min_speed: float = 28.0
    max_speed: float = 48.0
    steer_gain: float = 0.24
    d_gain: float = 0.08
    error_filter_alpha: float = 0.42
    deadband_px: float = 6.0
    max_correction: float = 24.0
    max_correction_step: float = 10.0
    tank_turn_error_px: float = 66.0
    tank_turn_speed: float = 62.0
    lost_forward_speed: float = 30.0
    lost_forward_frames: int = 2
    search_speed: float = 46.0


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


class BasicColorLineFollower:
    def __init__(self, config: BasicColorLineFollowConfig | None = None) -> None:
        self.config = config or BasicColorLineFollowConfig()
        self.last_error = 0.0
        self.filtered_error = 0.0
        self.last_filtered_error = 0.0
        self.last_correction = 0.0
        self.last_centroid_x: int | None = None
        self.last_frame_width = 0
        self.lost_frames = 0
        self.search_direction = -1

    def reset_steering(self) -> None:
        self.last_error = 0.0
        self.filtered_error = 0.0
        self.last_filtered_error = 0.0
        self.last_correction = 0.0
        self.last_centroid_x = None
        self.lost_frames = 0

    def _roi_bounds(self, frame_shape: tuple[int, int, int]) -> tuple[int, int]:
        frame_h = frame_shape[0]
        roi_top = int(frame_h * self.config.roi_top_ratio)
        roi_bottom = int(frame_h * self.config.roi_bottom_ratio)
        roi_top = int(clamp(roi_top, 0, frame_h - 1))
        roi_bottom = int(clamp(roi_bottom, roi_top + 1, frame_h))
        return roi_top, roi_bottom

    @staticmethod
    def _bottom_contact_cols(mask: np.ndarray, contour: np.ndarray) -> int:
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
        band_start = int(mask.shape[0] * 0.78)
        bottom_band = contour_mask[band_start:, :]
        if bottom_band.size == 0:
            return 0
        return int(np.count_nonzero(np.any(bottom_band > 0, axis=0)))

    @staticmethod
    def _is_t_junction(mask: np.ndarray, contour: np.ndarray) -> bool:
        if mask is None or contour is None:
            return False
        x, y, w, h = cv2.boundingRect(contour)
        if w < 32 or h < 20:
            return False
        roi = mask[y:y + h, x:x + w]
        if roi.size == 0:
            return False
        top_band = roi[: max(1, int(h * 0.30)), :]
        bottom_band = roi[int(h * 0.68):, :]
        if top_band.size == 0 or bottom_band.size == 0:
            return False
        top_cols = np.count_nonzero(np.any(top_band > 0, axis=0))
        center_bottom = np.count_nonzero(
            np.any(
                bottom_band[:, int(w * 0.35): int(w * 0.65)],
                axis=0,
            ),
        )
        return top_cols >= int(w * 0.45) and center_bottom >= max(3, int(w * 0.08))

    def _build_black_mask(self, roi_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        median_v = float(np.median(hsv[:, :, 2]))
        black_v_upper = int(clamp(median_v * 0.58, 55, 125))
        black_s_upper = 90 if median_v > 145 else 70
        hsv_mask = cv2.inRange(
            hsv,
            np.array((0, 0, 0), dtype=np.uint8),
            np.array((180, black_s_upper, black_v_upper), dtype=np.uint8),
        )

        b = roi_bgr[:, :, 0].astype(np.int16)
        g = roi_bgr[:, :, 1].astype(np.int16)
        r = roi_bgr[:, :, 2].astype(np.int16)
        rgb_max = np.maximum(np.maximum(b, g), r)
        rgb_min = np.minimum(np.minimum(b, g), r)
        neutral_dark = (
            (rgb_max <= black_v_upper + 20)
            & ((rgb_max - rgb_min) <= 42)
        )
        neutral_dark_mask = np.where(neutral_dark, 255, 0).astype(np.uint8)
        return cv2.bitwise_or(hsv_mask, neutral_dark_mask)

    def _build_blue_mask(self, roi_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        blue_ranges = (
            ((78, 28, 70), (105, 255, 255)),   # light blue / cyan-blue
            ((92, 32, 30), (132, 255, 170)),   # navy / dark blue
            ((105, 12, 18), (138, 130, 120)),  # low-sat dark blue
        )
        for lower, upper in blue_ranges:
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
        rgb_max = np.maximum(np.maximum(b, g), r)
        rgb_min = np.minimum(np.minimum(b, g), r)
        rgb_span = rgb_max - rgb_min

        blue_dom = (
            ((b >= 50) & (b >= g + 8) & (b >= r + 14))
            | ((b >= 32) & (b * 10 >= g * 11) & (b * 10 >= r * 14) & (b - r >= 10))
        )
        whiteish = (
            (hsv[:, :, 2] >= 130)
            & (hsv[:, :, 1] <= 30)
            & (rgb_span <= 24)
        )

        mask = cv2.bitwise_or(mask, np.where(blue_dom, 255, 0).astype(np.uint8))
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(np.where(whiteish, 255, 0).astype(np.uint8)))
        return mask

    def _build_mask(self, roi_bgr: np.ndarray) -> np.ndarray:
        color = self.config.line_color.strip().upper()
        if color == "BLACK":
            mask = self._build_black_mask(roi_bgr)
        elif color == "BLUE":
            mask = self._build_blue_mask(roi_bgr)
        else:
            raise ValueError(f"Unsupported line color: {self.config.line_color}")

        kernel = np.ones(
            (self.config.morph_kernel_size, self.config.morph_kernel_size),
            dtype=np.uint8,
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask

    def _select_best_contour(
        self,
        mask: np.ndarray,
        gray: np.ndarray,
    ) -> tuple[np.ndarray | None, float]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0.0

        best_contour = None
        best_area = 0.0
        best_score = -1.0
        roi_center_x = mask.shape[1] / 2.0
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.config.min_contour_area:
                continue

            contact_cols = self._bottom_contact_cols(mask, contour)
            if (
                contact_cols < self.config.min_bottom_contact_cols
                and area < self.config.min_contour_area * 3.0
            ):
                continue

            x, y, w, h = cv2.boundingRect(contour)
            elongation = max(w, h) / max(1.0, min(w, h))
            if elongation < 1.15 and area < 1200.0:
                continue

            contour_mask = np.zeros_like(mask)
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
            mean_gray = float(cv2.mean(gray, mask=contour_mask)[0])
            moments = cv2.moments(contour)
            if moments["m00"] > 0:
                centroid_x = float(moments["m10"] / moments["m00"])
            else:
                centroid_x = float(x + (w / 2.0))
            center_penalty = abs(centroid_x - roi_center_x)
            bottom_bonus = float(y + h)

            score = (contact_cols * 1200.0) + area + (bottom_bonus * 8.0) - (center_penalty * 0.8)
            if self.config.line_color.strip().upper() == "BLACK":
                score += max(0.0, 255.0 - mean_gray) * 10.0

            if best_contour is None or score > best_score:
                best_contour = contour
                best_area = area
                best_score = score

        return best_contour, best_area

    def detect_line(self, frame_bgr: np.ndarray) -> LineDetectionResult:
        frame_h, frame_w = frame_bgr.shape[:2]
        self.last_frame_width = frame_w
        roi_top, roi_bottom = self._roi_bounds(frame_bgr.shape)
        roi = frame_bgr[roi_top:roi_bottom, :]
        roi = cv2.GaussianBlur(roi, self.config.blur_kernel, 0)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mask = self._build_mask(roi)
        contour, contour_area = self._select_best_contour(mask, gray)

        if contour is None:
            self.lost_frames += 1
            return LineDetectionResult(
                found=False,
                error_px=self.last_error,
                centroid_x=None,
                centroid_y=None,
                contour_area=0.0,
                roi_top=roi_top,
                roi_bottom=roi_bottom,
                mask=mask,
            )

        contour_global = contour.copy()
        contour_global[:, 0, 1] += roi_top

        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
        bottom_start = int(contour_mask.shape[0] * 0.55)
        bottom_band = contour_mask[bottom_start:, :]
        band_moments = cv2.moments(bottom_band)
        full_moments = cv2.moments(contour)

        if band_moments["m00"] > 0:
            centroid_x = int(band_moments["m10"] / band_moments["m00"])
            centroid_y = int(band_moments["m01"] / band_moments["m00"]) + bottom_start + roi_top
        elif full_moments["m00"] > 0:
            centroid_x = int(full_moments["m10"] / full_moments["m00"])
            centroid_y = int(full_moments["m01"] / full_moments["m00"]) + roi_top
        else:
            x, y, w, h = cv2.boundingRect(contour)
            centroid_x = int(x + (w / 2.0))
            centroid_y = int(y + (h / 2.0)) + roi_top

        target_x = (frame_w / 2.0) + self.config.center_offset_px
        error_px = float(centroid_x - target_x)
        if abs(error_px) < self.config.deadband_px:
            error_px = 0.0

        self.last_error = error_px
        if error_px < 0.0:
            self.search_direction = -1
        elif error_px > 0.0:
            self.search_direction = 1
        self.last_centroid_x = centroid_x
        self.lost_frames = 0

        return LineDetectionResult(
            found=True,
            error_px=error_px,
            centroid_x=centroid_x,
            centroid_y=centroid_y,
            contour_area=contour_area,
            roi_top=roi_top,
            roi_bottom=roi_bottom,
            is_t_junction=self._is_t_junction(mask, contour),
            contour=contour_global,
            mask=mask,
        )

    def compute_steering(
        self,
        detection: LineDetectionResult,
        preferred_turn: str | None = None,
    ) -> MotorCommand:
        preferred = (preferred_turn or "").strip().upper()

        if not detection.found:
            if self.lost_frames <= max(0, self.config.lost_forward_frames):
                return MotorCommand(
                    left_speed=self.config.lost_forward_speed,
                    right_speed=self.config.lost_forward_speed,
                    correction=0.0,
                    recovery_mode=True,
                )

            if preferred == "LEFT":
                direction = -1.0
            elif preferred == "RIGHT":
                direction = 1.0
            else:
                direction = float(self.search_direction)

            return MotorCommand(
                left_speed=self.config.search_speed * direction,
                right_speed=-self.config.search_speed * direction,
                correction=0.0,
                recovery_mode=True,
            )

        if detection.is_t_junction and preferred in {"LEFT", "RIGHT"}:
            outer = self.config.tank_turn_speed
            inner = -max(8.0, outer * 0.15)
            if preferred == "LEFT":
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

        self.last_filtered_error = self.filtered_error
        alpha = self.config.error_filter_alpha
        self.filtered_error = (
            (alpha * self.filtered_error)
            + ((1.0 - alpha) * detection.error_px)
        )

        if abs(self.filtered_error) >= self.config.tank_turn_error_px:
            turn = self.config.tank_turn_speed
            inner = 0.0
            if self.filtered_error < 0.0:
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

        d_term = self.config.d_gain * (self.filtered_error - self.last_filtered_error)
        target_correction = clamp(
            (self.config.steer_gain * self.filtered_error) + d_term,
            -self.config.max_correction,
            self.config.max_correction,
        )
        correction_step = clamp(
            target_correction - self.last_correction,
            -self.config.max_correction_step,
            self.config.max_correction_step,
        )
        correction = self.last_correction + correction_step
        self.last_correction = correction

        turn_scale = min(
            1.0,
            abs(self.filtered_error) / max(1.0, self.config.tank_turn_error_px),
        )
        base_speed = self.config.base_speed - (
            (self.config.base_speed - self.config.min_speed) * turn_scale
        )
        left_speed = clamp(base_speed + correction, -100.0, self.config.max_speed)
        right_speed = clamp(base_speed - correction, -100.0, self.config.max_speed)
        return MotorCommand(
            left_speed=left_speed,
            right_speed=right_speed,
            correction=correction,
            recovery_mode=False,
        )


def draw_line_debug(frame_bgr: np.ndarray, detection: LineDetectionResult) -> None:
    cv2.rectangle(
        frame_bgr,
        (0, detection.roi_top),
        (frame_bgr.shape[1] - 1, detection.roi_bottom),
        (255, 220, 0),
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
