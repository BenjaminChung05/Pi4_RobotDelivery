from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2

try:
    from picamera2 import Picamera2
except ImportError:
    Picamera2 = None

from line_detector import (
    LineDetectionResult,
    LineDetectorConfig,
    LineFollower,
    MotorCommand,
    SteeringConfig,
    draw_line_debug,
)
from motor_ctrl import MotorConfig, MotorController
from object_detect import DetectedObject, ObjectDetectorConfig, OpenCVDnnObjectDetector, draw_object_debug
from shared_state import SharedState
from state_machine import RobotState, RobotStateMachine, StateMachineConfig
from symbol_detect import (
    FastFilterConfig,
    SymbolCandidate,
    SymbolDetectorConfig,
    SymbolResult,
    TFLiteSymbolDetector,
    draw_symbol_debug,
)
from utils import FpsCounter, clamp


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class CameraConfig:
    source: int = 0
    width: int = 320
    height: int = 240
    warmup_frames: int = 6
    use_picamera2_fallback: bool = True
    swap_rb: bool = field(default_factory=lambda: _env_flag("ROBOT_CAMERA_SWAP_RB", False))
    awb_enable: bool = field(default_factory=lambda: _env_flag("ROBOT_CAMERA_AWB_ENABLE", True))
    awb_red_gain: float = field(default_factory=lambda: _env_float("ROBOT_CAMERA_AWB_RED", 0.0))
    awb_blue_gain: float = field(default_factory=lambda: _env_float("ROBOT_CAMERA_AWB_BLUE", 0.0))


class PiCameraCapture:
    """Small adapter that mimics cv2.VideoCapture for Picamera2."""

    def __init__(self, camera_config: CameraConfig) -> None:
        self._camera = Picamera2()
        config = self._camera.create_video_configuration(
            main={"size": (camera_config.width, camera_config.height), "format": "BGR888"}
        )
        self._camera.configure(config)
        self._camera.start()

        controls: dict[str, object] = {"AwbEnable": bool(camera_config.awb_enable)}
        if camera_config.awb_red_gain > 0.0 and camera_config.awb_blue_gain > 0.0:
            controls["AwbEnable"] = False
            controls["ColourGains"] = (float(camera_config.awb_red_gain), float(camera_config.awb_blue_gain))
        try:
            self._camera.set_controls(controls)
        except Exception:
            # Keep camera running even when a control is unsupported on the device.
            pass

        self.backend_name = "PICAMERA2"

    def read(self):
        frame_bgr = self._camera.capture_array()
        if frame_bgr is None or frame_bgr.size == 0:
            return False, None
        return True, frame_bgr

    def release(self) -> None:
        self._camera.stop()


def _line_config_from_env() -> LineDetectorConfig:
    config = LineDetectorConfig()
    raw = os.getenv("ROBOT_LINE_COLORS", "").strip()
    if not raw:
        return config

    colors = tuple(token.strip().upper() for token in raw.split(",") if token.strip())
    if colors:
        config.line_colors = colors
    return config


@dataclass
class MainConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    line: LineDetectorConfig = field(default_factory=_line_config_from_env)
    steering: SteeringConfig = field(default_factory=SteeringConfig)
    motor: MotorConfig = field(default_factory=MotorConfig)
    state_machine: StateMachineConfig = field(default_factory=StateMachineConfig)
    symbol: SymbolDetectorConfig = field(
        default_factory=lambda: SymbolDetectorConfig(fast_filter=FastFilterConfig())
    )
    symbol_interval_frames: int = 1
    recover_speed_scale: float = 0.75
    action_turn_speed: float = 65.0
    recycle_turn_speed: float = 56.0
    recycle_duration_seconds: float = 1.25
    symbol_turn_guard_error_px: float = 20.0
    preferred_turn_timeout_seconds: float = 10.0
    preferred_turn_max_junction_uses: int = 2
    templates_dir: str = "templates"
    required_action_symbols: tuple[str, ...] = ("STOP", "LEFT", "RIGHT", "RECYCLE")
    symbol_confirm_frames: int = 1
    stop_confirm_frames: int = 5
    stop_min_confidence: float = 0.72
    object: ObjectDetectorConfig = field(default_factory=ObjectDetectorConfig)
    object_interval_frames: int = 2
    preview_scale: float = 2.4
    show_perf_overlay: bool = False
    show_ui_panels: bool = False
    show_debug: bool = field(
        default_factory=lambda: bool(os.getenv("DISPLAY")) and os.getenv("ROBOT_SHOW_DEBUG", "1") != "0"
    )
    log_period_seconds: float = 1.0


@dataclass
class PerfMetrics:
    line_ms: float = 0.0
    symbol_ms: float = 0.0
    object_ms: float = 0.0
    control_ms: float = 0.0
    loop_ms: float = 0.0


def _open_camera(config: CameraConfig):
    backend_candidates = [
        cv2.CAP_V4L2,
    ]

    last_error = f"Could not open camera source {config.source}."
    for backend in backend_candidates:
        camera = cv2.VideoCapture(config.source, backend)
        if not camera.isOpened():
            camera.release()
            continue

        # Request compact pixel formats first to reduce camera memory pressure.
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        frame_ok = False
        for _ in range(max(1, config.warmup_frames)):
            ok, frame = camera.read()
            if ok and frame is not None and frame.size > 0:
                frame_ok = True
                break

        if frame_ok:
            return camera, "CAP_V4L2"

        camera.release()
        backend_name = "CAP_V4L2"
        last_error = (
            f"Camera opened with {backend_name} but did not return valid frames. "
            "Check if another process is using /dev/video0."
        )

    if config.use_picamera2_fallback and Picamera2 is not None:
        try:
            camera = PiCameraCapture(config)
            for _ in range(max(1, config.warmup_frames)):
                ok, frame = camera.read()
                if ok and frame is not None and frame.size > 0:
                    return camera, "PICAMERA2"
            camera.release()
            last_error = "Picamera2 fallback started but did not return valid frames."
        except Exception as ex:
            last_error = f"Picamera2 fallback failed: {ex}"
    elif config.use_picamera2_fallback and Picamera2 is None:
        last_error = (
            f"{last_error} Picamera2 is not installed, so fallback is unavailable. "
            "Install with: sudo apt install -y python3-picamera2"
        )

    raise RuntimeError(last_error)


def _load_template_labels(templates_dir: str) -> set[str]:
    templates_path = Path(templates_dir)
    if not templates_path.exists() or not templates_path.is_dir():
        return set()

    aliases = {
        "STOP_SIGN": "STOP",
        "OCTAGON": "STOP",
        "BUTTON": "STOP",
        "HAZARD": "STOP",
        "FINGERPRINT": "QR",
        "QR": "QR",
        "ARROW_LEFT": "LEFT",
        "LEFT_ARROW": "LEFT",
        "LEFTTURN": "LEFT",
        "ARROW_RIGHT": "RIGHT",
        "RIGHT_ARROW": "RIGHT",
        "RIGHTTURN": "RIGHT",
    }

    labels: set[str] = set()
    for file_path in templates_path.iterdir():
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        if file_path.name.startswith("._"):
            continue
        token = file_path.stem.strip().upper().replace("-", "_").replace(" ", "_")
        canonical = aliases.get(token, token)
        if canonical in {"STOP", "LEFT", "RIGHT", "RECYCLE"}:
            labels.add(canonical)
    return labels


def _dispatch_week3_symbol(
    symbol_result: SymbolResult,
    state_machine: RobotStateMachine,
    now: float,
    logger: logging.Logger,
    recycle_duration_seconds: float,
) -> bool:
    label = symbol_result.action_label or symbol_result.label
    if not label:
        return False

    if label in {"QR", "FINGERPRINT"}:
        logger.info("Week 3 output symbol: %s", label)
        state_machine.mark_output_symbol(label, now)
        return True

    if label == "RECYCLE":
        state_machine.handle_symbol(label, now, duration_seconds=recycle_duration_seconds)
        logger.info("Week 3 action symbol: %s (360 turn)", label)
        return True

    if label == "STOP":
        state_machine.handle_symbol(label, now, duration_seconds=3.0)
        logger.info("Week 3 action symbol: %s (hold stop)", label)
        return True

    if label in {"LEFT", "RIGHT"}:
        state_machine.handle_symbol(label, now)
        logger.info("Week 3 action symbol: %s", label)
        return True

    logger.info("Ignoring unsupported symbol label: %s", label)
    return False


def _make_action_command(action_symbol: str | None, turn_speed: float) -> MotorCommand:
    inner = max(26.0, turn_speed - 18.0)
    if action_symbol == "LEFT":
        return MotorCommand(left_speed=inner, right_speed=turn_speed, correction=0.0, recovery_mode=False)
    if action_symbol == "RIGHT":
        return MotorCommand(left_speed=turn_speed, right_speed=inner, correction=0.0, recovery_mode=False)
    if action_symbol == "RECYCLE":
        return MotorCommand(left_speed=turn_speed, right_speed=-turn_speed, correction=0.0, recovery_mode=False)
    if action_symbol == "STOP":
        return MotorCommand(left_speed=0.0, right_speed=0.0, correction=0.0, recovery_mode=False)
    return MotorCommand(left_speed=0.0, right_speed=0.0, correction=0.0, recovery_mode=False)


def _apply_state_to_command(
    state: RobotState,
    state_machine: RobotStateMachine,
    line_command: MotorCommand,
    config: MainConfig,
) -> MotorCommand:
    if state == RobotState.EXECUTE_ACTION:
        action_symbol = state_machine.current_symbol
        turn_speed = config.recycle_turn_speed if action_symbol == "RECYCLE" else config.action_turn_speed
        return _make_action_command(action_symbol, turn_speed)

    if state == RobotState.RECOVER:
        return MotorCommand(
            left_speed=line_command.left_speed * config.recover_speed_scale,
            right_speed=line_command.right_speed * config.recover_speed_scale,
            correction=line_command.correction,
            recovery_mode=line_command.recovery_mode,
        )

    return line_command


def _draw_overlay(
    frame_bgr,
    line_result: LineDetectionResult,
    candidate: SymbolCandidate,
    symbol_result: SymbolResult,
    object_detections: list[DetectedObject],
    state_machine: RobotStateMachine,
    command: MotorCommand,
    fps: float,
    perf: PerfMetrics,
    fast_filter_status: str,
    ai_status: str,
    cooldown_remaining_s: float,
    show_perf_overlay: bool,
    detect_reason: str,
    show_ui_panels: bool,
) -> None:
    draw_line_debug(frame_bgr, line_result)
    draw_symbol_debug(frame_bgr, candidate, symbol_result)
    draw_object_debug(frame_bgr, object_detections)

    compact_status = (
        f"S:{symbol_result.label or '-'} {symbol_result.confidence:.2f} "
        f"| Last:{state_machine.last_detected_symbol or '-'} "
        f"| Why:{(symbol_result.reason or detect_reason or '-')}"
    )
    cv2.putText(
        frame_bgr,
        compact_status,
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )

    if not show_ui_panels:
        return

    display_symbol = symbol_result.label
    display_reason = symbol_result.reason or detect_reason or "ok"
    display_objects = ", ".join(
        f"{det.class_name}:{det.confidence:.2f}" for det in object_detections[:3]
    ) or "-"

    compact_line = (
        f"{state_machine.state.value} | S:{display_symbol or '-'} "
        f"| E:{line_result.error_px:.0f} | FPS:{fps:.1f}"
    )
    bar_h = 30
    cv2.rectangle(
        frame_bgr,
        (0, frame_bgr.shape[0] - bar_h),
        (frame_bgr.shape[1], frame_bgr.shape[0]),
        (30, 30, 30),
        -1,
    )
    cv2.putText(
        frame_bgr,
        compact_line,
        (8, frame_bgr.shape[0] - 9),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1,
    )

    if show_perf_overlay:
        perf_text = (
            f"line:{perf.line_ms:.1f}ms sym:{perf.symbol_ms:.1f}ms "
            f"ctrl:{perf.control_ms:.1f}ms loop:{perf.loop_ms:.1f}ms"
        )
        cv2.putText(
            frame_bgr,
            perf_text,
            (8, frame_bgr.shape[0] - bar_h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (220, 220, 220),
            1,
        )

    status_lines = [
        f"FastFilter: {fast_filter_status}",
        f"AI: {ai_status}",
        f"Cooldown: {cooldown_remaining_s:.2f}s",
        f"Label: {display_symbol or '-'} ({symbol_result.confidence:.2f})",
        f"Objects: {display_objects}",
        f"Why: {display_reason}",
    ]
    panel_width = 240
    panel_height = 150
    margin = 10
    x1 = frame_bgr.shape[1] - panel_width - margin
    y1 = margin
    x2 = frame_bgr.shape[1] - margin
    y2 = y1 + panel_height
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (40, 40, 40), -1)
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 255, 255), 1)
    for index, text in enumerate(status_lines):
        cv2.putText(
            frame_bgr,
            text,
            (x1 + 7, y1 + 18 + (18 * index)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )


def run_robot(config: MainConfig | None = None) -> None:
    config = config or MainConfig()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("week3")

    camera, camera_backend = _open_camera(config.camera)
    motor = MotorController(config.motor)
    motor.setup()

    line_follower = LineFollower(config.line, config.steering)
    symbol_detector = TFLiteSymbolDetector(config.symbol)
    object_detector = OpenCVDnnObjectDetector(config.object)
    state_machine = RobotStateMachine(config.state_machine)
    fps_counter = FpsCounter(config.log_period_seconds)
    shared_state = SharedState()
    template_labels = _load_template_labels(config.templates_dir)
    allowed_action_symbols = {
        symbol for symbol in config.required_action_symbols if symbol in template_labels
    }
    allowed_action_symbols.update({"LEFT", "RIGHT"})

    candidate = SymbolCandidate(found=False)
    symbol_result = SymbolResult(enabled=symbol_detector.enabled, reason=symbol_detector.reason)
    object_detections: list[DetectedObject] = []
    perf = PerfMetrics()
    frame_index = 0
    pending_symbol_label: str | None = None
    pending_symbol_hits = 0
    preferred_turn: str | None = None
    preferred_turn_set_at: float = 0.0
    preferred_turn_junction_uses: int = 0
    prev_is_t_junction: bool = False

    if symbol_detector.enabled:
        logger.info("TFLite symbol model loaded from %s", config.symbol.model_path)
    else:
        logger.warning("Symbol detector disabled: %s", symbol_detector.reason)

        if object_detector.enabled:
            logger.info("Object detector loaded from %s", config.object.weights_path)
        else:
            logger.warning("Object detector disabled: %s", object_detector.reason)
    logger.info(
        "Camera backend selected: %s (%dx%d)",
        camera_backend,
        config.camera.width,
        config.camera.height,
    )

    if allowed_action_symbols:
        logger.info(
            "Action symbols enabled from templates: %s",
            ", ".join(sorted(allowed_action_symbols)),
        )
    else:
        logger.warning(
            "No action templates found for %s in %s. Actions are disabled until template files exist.",
            ", ".join(config.required_action_symbols),
            config.templates_dir,
        )

    if not motor.available:
        logger.warning("Motor controller is in dry-run mode because RPi.GPIO is unavailable.")

    logger.info("Starting sequential robot loop. Press Ctrl+C to exit.")

    window_name = "Week 3 Robot"
    if config.show_debug:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        preview_width = max(config.camera.width, int(config.camera.width * config.preview_scale))
        preview_height = max(config.camera.height, int(config.camera.height * config.preview_scale))
        cv2.resizeWindow(window_name, preview_width, preview_height)

    try:
        while True:
            loop_time = time.perf_counter()
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera returned an empty frame.")

            if config.camera.swap_rb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame = cv2.resize(frame, (config.camera.width, config.camera.height))
            frame_index += 1
            candidate = SymbolCandidate(found=False)
            symbol_result = SymbolResult(enabled=symbol_detector.enabled, accepted=False, reason="interval")

            line_start = time.perf_counter()
            line_result = line_follower.detect_line(frame)
            line_command = line_follower.compute_steering(line_result, preferred_turn=preferred_turn)
            perf.line_ms = (time.perf_counter() - line_start) * 1000.0

            # Expire stale preferred turn memory so old arrows do not bias future routing forever.
            if preferred_turn is not None:
                if (loop_time - preferred_turn_set_at) > config.preferred_turn_timeout_seconds:
                    logger.info("Preferred turn expired by timeout: %s", preferred_turn)
                    preferred_turn = None
                    preferred_turn_set_at = 0.0
                    preferred_turn_junction_uses = 0

            # Count distinct T-junction entries while a preferred turn exists.
            if line_result.is_t_junction and not prev_is_t_junction and preferred_turn is not None:
                preferred_turn_junction_uses += 1
                if preferred_turn_junction_uses >= max(1, config.preferred_turn_max_junction_uses):
                    logger.info("Preferred turn expired by junction-use limit: %s", preferred_turn)
                    preferred_turn = None
                    preferred_turn_set_at = 0.0
                    preferred_turn_junction_uses = 0
            prev_is_t_junction = line_result.is_t_junction

            state_machine.update(loop_time)
            cooldown_remaining_s = max(0.0, state_machine.cooldown_until - loop_time)
            ai_status = "ON" if symbol_detector.enabled else "OFF"
            fast_filter_status = "SKIP"
            detect_reason = "-"
            symbol_turn_guard_active = line_command.recovery_mode or abs(line_result.error_px) > config.symbol_turn_guard_error_px

            object_start = time.perf_counter()
            if object_detector.enabled and frame_index % max(1, config.object_interval_frames) == 0:
                object_detections = object_detector.detect(frame)
            elif not object_detector.enabled:
                object_detections = []
            perf.object_ms = (time.perf_counter() - object_start) * 1000.0

            symbol_start = time.perf_counter()
            if symbol_turn_guard_active:
                symbol_result = SymbolResult(
                    enabled=symbol_detector.enabled,
                    accepted=False,
                    reason="Line turn guard active.",
                )
                detect_reason = symbol_result.reason
            elif frame_index % config.symbol_interval_frames == 0:
                candidate = symbol_detector.fast_filter(frame)
                if candidate.found:
                    fast_filter_status = "PASS"
                    try:
                        symbol_result = symbol_detector.classify(frame, candidate)
                    except cv2.error as ex:
                        symbol_result = SymbolResult(
                            enabled=symbol_detector.enabled,
                            accepted=False,
                            reason=f"OpenCV classify error: {ex}",
                        )
                    except Exception as ex:
                        symbol_result = SymbolResult(
                            enabled=symbol_detector.enabled,
                            accepted=False,
                            reason=f"Classify error: {ex}",
                        )
                    detect_reason = symbol_result.reason or "ok"
                    if symbol_result.label is not None:
                        if symbol_result.label == pending_symbol_label:
                            pending_symbol_hits += 1
                        else:
                            pending_symbol_label = symbol_result.label
                            pending_symbol_hits = 1

                        action_label = symbol_result.action_label or symbol_result.label
                        required_hits = config.stop_confirm_frames if action_label == "STOP" else config.symbol_confirm_frames
                        stop_conf_ok = action_label != "STOP" or symbol_result.confidence >= config.stop_min_confidence
                        should_dispatch = (
                            symbol_result.accepted
                            and
                            (action_label in allowed_action_symbols or symbol_result.label in {"QR", "FINGERPRINT"})
                            and state_machine.can_accept_symbol(loop_time)
                            and pending_symbol_hits >= required_hits
                            and stop_conf_ok
                        )
                        if should_dispatch:
                            if _dispatch_week3_symbol(
                                symbol_result,
                                state_machine,
                                loop_time,
                                logger,
                                config.recycle_duration_seconds,
                            ):
                                if action_label in {"LEFT", "RIGHT"}:
                                    preferred_turn = action_label
                                    preferred_turn_set_at = loop_time
                                    preferred_turn_junction_uses = 0
                                pending_symbol_label = None
                                pending_symbol_hits = 0
                    else:
                        pending_symbol_label = None
                        pending_symbol_hits = 0
                else:
                    fast_filter_status = "REJECT"
                    symbol_result = symbol_detector.probe_symbol(frame)
                    if symbol_result.accepted and symbol_result.label is not None:
                        fast_filter_status = "SYM_PROBE"
                        detect_reason = symbol_result.reason or "symbol probe"
                        if symbol_result.label == pending_symbol_label:
                            pending_symbol_hits += 1
                        else:
                            pending_symbol_label = symbol_result.label
                            pending_symbol_hits = 1

                        action_label = symbol_result.action_label or symbol_result.label
                        required_hits = config.stop_confirm_frames if action_label == "STOP" else config.symbol_confirm_frames
                        stop_conf_ok = action_label != "STOP" or symbol_result.confidence >= config.stop_min_confidence
                        should_dispatch = (
                            symbol_result.accepted
                            and
                            (action_label in allowed_action_symbols or symbol_result.label in {"QR", "FINGERPRINT"})
                            and state_machine.can_accept_symbol(loop_time)
                            and pending_symbol_hits >= required_hits
                            and stop_conf_ok
                        )
                        if should_dispatch:
                            if _dispatch_week3_symbol(
                                symbol_result,
                                state_machine,
                                loop_time,
                                logger,
                                config.recycle_duration_seconds,
                            ):
                                if action_label in {"LEFT", "RIGHT"}:
                                    preferred_turn = action_label
                                    preferred_turn_set_at = loop_time
                                    preferred_turn_junction_uses = 0
                                pending_symbol_label = None
                                pending_symbol_hits = 0
                    else:
                        symbol_result = SymbolResult(
                            enabled=symbol_detector.enabled,
                            accepted=False,
                            reason="Fast filter rejected frame.",
                        )
                        detect_reason = "filter reject"
                        pending_symbol_label = None
                        pending_symbol_hits = 0
            perf.symbol_ms = (time.perf_counter() - symbol_start) * 1000.0

            control_start = time.perf_counter()
            command = _apply_state_to_command(
                state=state_machine.state,
                state_machine=state_machine,
                line_command=line_command,
                config=config,
            )

            left_pwm = clamp(command.left_speed, -100.0, 100.0)
            right_pwm = clamp(command.right_speed, -100.0, 100.0)
            motor.set_motor(left_pwm, right_pwm)
            perf.control_ms = (time.perf_counter() - control_start) * 1000.0
            perf.loop_ms = (time.perf_counter() - loop_time) * 1000.0

            fps = fps_counter.update()
            with shared_state.lock:
                shared_state.frame_index = frame_index
                shared_state.fps = fps
                shared_state.robot_state = state_machine.state.value
                shared_state.last_symbol = state_machine.last_detected_symbol
                shared_state.last_symbol_score = symbol_result.confidence
                shared_state.last_shape = None
                shared_state.last_shape_score = 0.0
                shared_state.line_error_px = line_result.error_px
                shared_state.line_found = line_result.found
                shared_state.left_pwm = left_pwm
                shared_state.right_pwm = right_pwm
                shared_state.line_ms = perf.line_ms
                shared_state.symbol_ms = perf.symbol_ms
                shared_state.control_ms = perf.control_ms
                shared_state.loop_ms = perf.loop_ms

            if frame_index % max(1, config.symbol_interval_frames * 4) == 0:
                logger.info(
                    "fps=%.1f loop_ms=%.2f line_ms=%.2f symbol_ms=%.2f object_ms=%.2f state=%s line_found=%s err=%.1f motor=(%.1f, %.1f) symbol=%s last=%s filter=%s reason=%s objects=%s",
                    fps,
                    perf.loop_ms,
                    perf.line_ms,
                    perf.symbol_ms,
                    perf.object_ms,
                    shared_state.robot_state,
                    shared_state.line_found,
                    shared_state.line_error_px,
                    shared_state.left_pwm,
                    shared_state.right_pwm,
                    symbol_result.label or "-",
                    state_machine.last_detected_symbol or "-",
                    fast_filter_status,
                    symbol_result.reason or detect_reason or "-",
                    ", ".join(det.class_name for det in object_detections[:3]) or "-",
                )

            if config.show_debug:
                display = frame.copy()
                _draw_overlay(
                    frame_bgr=display,
                    line_result=line_result,
                    candidate=candidate,
                    symbol_result=symbol_result,
                    object_detections=object_detections,
                    state_machine=state_machine,
                    command=command,
                    fps=fps,
                    perf=perf,
                    fast_filter_status=fast_filter_status,
                    ai_status=ai_status,
                    cooldown_remaining_s=cooldown_remaining_s,
                    show_perf_overlay=config.show_perf_overlay,
                    detect_reason=detect_reason,
                    show_ui_panels=config.show_ui_panels,
                )
                cv2.imshow(window_name, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        logger.info("Stopping robot loop.")
    finally:
        motor.cleanup()
        camera.release()
        if config.show_debug:
            cv2.destroyAllWindows()


def main() -> None:
    run_robot()


if __name__ == "__main__":
    main()
