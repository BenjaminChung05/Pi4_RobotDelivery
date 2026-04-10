from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from utils import clamp


@dataclass
class LineDetectorConfig:
    roi_top_ratio: float = 0.46
    roi_bottom_ratio: float = 1.0
    threshold_value: int = 60
    use_adaptive_threshold: bool = True
    adaptive_block_size: int = 31
    adaptive_c: int = 7
    blur_kernel: tuple[int, int] = (5, 5)
    min_contour_area: float = 500.0
    morph_kernel_size: int = 3
    line_colors: tuple[str, ...] = ("BLACK",)
    lost_search_flip_frames: int = 4
    alternate_search_on_loss: bool = False
    lost_turn_delay_frames: int = 5
    lost_forward_speed_scale: float = 0.80
    recovery_preferred_turn: str | None = "LEFT"
    require_bottom_contact: bool = True
    bottom_contact_band_ratio: float = 0.22
    bottom_contact_min_cols_ratio: float = 0.06
    high_error_confirm_frames: int = 2


@dataclass
class SteeringConfig:
    base_speed: float = 45.0
    steer_gain: float = 0.10
    max_adjust: float = 26.0
    deadband_px: float = 12.0
    lost_turn_speed: float = 62.0
    tank_turn_error_px: float = 60.0
    tank_turn_speed: float = 70.0
    error_filter_alpha: float = 0.82
    max_correction_step: float = 2.5
    invert_turn_direction: bool = False


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
        self.last_correction = 0.0
        self.last_centroid_x: int | None = None
        self.last_frame_width: int = 0
        self.lost_frames = 0
        self.search_direction = -1
        self._junction_latched = False
        self._high_error_frames = 0

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

    def _build_color_mask(self, roi_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        cfg = self.detector_config

        # Built-in HSV ranges so we can extend from black-only to multi-color lines.
        color_ranges: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
            "BLACK": [((0, 0, 0), (180, 70, 60))],
            "BLUE": [((90, 70, 40), (130, 255, 255))],
            "YELLOW": [((18, 70, 70), (40, 255, 255))],
            "GREEN": [((35, 50, 40), (90, 255, 255))],
            "WHITE": [((0, 0, 170), (180, 70, 255))],
            "RED": [
                ((0, 80, 50), (10, 255, 255)),
                ((170, 80, 50), (180, 255, 255)),
            ],
        }

        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for token in cfg.line_colors:
            key = token.strip().upper()
            ranges = color_ranges.get(key)
            if not ranges:
                continue
            for lower, upper in ranges:
                lower_np = np.array(lower, dtype=np.uint8)
                upper_np = np.array(upper, dtype=np.uint8)
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower_np, upper_np))

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

        mask_color = self._build_color_mask(roi)
        steering_bias_top = int(roi.shape[0] * 0.38)
        candidates: list[np.ndarray] = []
        if "BLACK" in {token.strip().upper() for token in self.detector_config.line_colors}:
            candidates.append(mask_color)
            candidates.append(cv2.bitwise_or(mask_basic, mask_adaptive))
        else:
            candidates.append(mask_color)
            candidates.append(mask_basic)
            candidates.append(mask_adaptive)

        kernel_size = self.detector_config.morph_kernel_size
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        best_contour = None
        best_area = 0.0
        best_contact_cols = 0
        best_mask = None
        for candidate_mask in candidates:
            mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
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

                contact_cols = self._bottom_contact_cols(
                    mask,
                    contour,
                    self.detector_config.bottom_contact_band_ratio,
                )
                min_contact_cols = int(mask.shape[1] * self.detector_config.bottom_contact_min_cols_ratio)
                has_bottom_contact = contact_cols >= max(1, min_contact_cols)

                if self.detector_config.require_bottom_contact and not has_bottom_contact:
                    continue

                # Prefer contours that are visible near the robot wheels first,
                # then break ties by larger area.
                if (
                    contact_cols > best_contact_cols
                    or (contact_cols == best_contact_cols and area > best_area)
                ):
                    best_contact_cols = contact_cols
                    best_area = area
                    best_contour = contour
                    best_mask = mask

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

        centroid_x = int(moments["m10"] / moments["m00"])
        centroid_y = int(moments["m01"] / moments["m00"]) + roi_top
        error_px = float(centroid_x - (frame_w // 2))
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
            outer = max(self.steering_config.base_speed + 10.0, self.steering_config.tank_turn_speed * 0.70)
            inner = max(18.0, outer * 0.30)
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
        self.filtered_error = (alpha * self.filtered_error) + ((1.0 - alpha) * detection.error_px)

        # Sharp turns are handled with a confirmed, filtered error to reduce
        # wrong-side snap turns from one-frame contour jitter.
        if abs(self.filtered_error) >= self.steering_config.tank_turn_error_px:
            self._high_error_frames += 1
        else:
            self._high_error_frames = 0

        if self._high_error_frames >= max(1, self.detector_config.high_error_confirm_frames):
            turn = self.steering_config.tank_turn_speed
            inner = max(0.0, turn * 0.15)
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

        target_correction = clamp(
            self.steering_config.steer_gain * self.filtered_error,
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

        base = self.steering_config.base_speed
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
        cv2.drawContours(frame_bgr, [detection.contour], -1, (0, 255, 0), 2)
    if detection.centroid_x is not None and detection.centroid_y is not None:
        cv2.circle(
            frame_bgr,
            (detection.centroid_x, detection.centroid_y),
            6,
            (0, 0, 255),
            -1,
        )
