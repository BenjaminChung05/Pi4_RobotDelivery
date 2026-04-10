"""Raspberry Pi main script — Week2-style line following with symbol actions."""

import signal
import time

import cv2

from week2_line_follow import (
    Week2LineFollowConfig,
    Week2LineFollower,
    draw_line_debug,
)

try:
    import RPi.GPIO as GPIO
    ON_PI = True
except ImportError:
    ON_PI = False

try:
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
except ImportError:
    HAS_PICAMERA2 = False
    print("[pi_main] Picamera2 not available — will use cv2.VideoCapture")

from pi_client import PiClient

# =====================================================================
# Configuration
# =====================================================================
DEBUG_PREVIEW = True            # Show a local cv2 window on the Pi
ROTATE_180 = False               # Match the Week2 camera orientation
# =====================================================================

# --- Camera ---
FRAME_W, FRAME_H = 640, 480

# --- Motor GPIO pins (BCM numbering) ---
ENA, IN1, IN2 = 12, 17, 18     # Left motor
ENB, IN3, IN4 = 13, 22, 23     # Right motor
PWM_FREQ = 500                  # Hz — lower freq = more torque for heavy robot

BASE_SPEED = 42                 # used only for symbol-action commands
FOLLOW_STALL_PWM = 30.0         # Week2 follower expects a real stall floor
BLUE_MIN_AREA = 90.0            # earliest blue branch size worth trusting
BLUE_MIN_CENTROID_Y_RATIO = 0.36
BLUE_PREFERRED_Y_RATIO = 0.22
BLUE_BLACK_AREA_RATIO = 0.20    # blue must be reasonably competitive with black
BLUE_PREFERRED_AREA_RATIO = 0.05
BLUE_CENTROID_MARGIN_PX = 72.0
BLUE_LATCH_SECONDS = 0.35

CAMERA_FRAME_RATE = 60.0
CAMERA_SETTLE_S = 0.35
CAMERA_FALLBACK_COLOUR_GAINS = (1.50, 1.20)

# --- Symbol detection (while moving) ---
TURN_GUARD_ERROR_PX = 40        # only detect symbols when mostly aligned to the line
SYMBOL_SEND_INTERVAL = 2        # send every N-th frame for detection
SYMBOL_REQUEST_GAP = 0.40       # cap blocking Mac round-trips so steering stays responsive
SYMBOL_VOTE_WINDOW = 6          # keep last N detection results
SYMBOL_VOTE_THRESHOLD = 4       # was 2 — need 4/6 votes (67%) to confirm non-arrow
ARROW_VOTE_THRESHOLD = 5        # was 3 — arrows need 5/6 votes (83%) to confirm
SYMBOL_COOLDOWN = 2.0           # seconds after acting before detecting again
PREFERRED_TURN_TIMEOUT = 12.0   # drop stale arrow memory if no junction appears
PREFERRED_TURN_BLUE_FRAMES = 4  # clear arrow memory once we've committed to the branch

# --- Networking ---
SERVER_IP = "10.134.9.249"     # ← MacBook IP
SERVER_PORT = 5001

# =====================================================================
# State machine
# =====================================================================
FOLLOWING            = "FOLLOWING"
EXECUTING_ACTION     = "EXECUTING_ACTION"


class Robot:
    def __init__(self):
        self.state = FOLLOWING

        # Primary line following now uses the Week2 technique directly:
        # For the main path, use the sample code's logic directly:
        # narrow ROI, RED/YELLOW/BLACK contour priority, bbox-center error, simple PID.
        self.line_follower = Week2LineFollower(
            Week2LineFollowConfig(
                line_color="PRIORITY",
                line_priority=("BLACK", "RED", "YELLOW"),
                roi_top_ratio=130.0 / 240.0,
                roi_bottom_ratio=180.0 / 240.0,
                min_line_pixels=80,
                scan_rows=16,
                min_valid_rows=3,
                base_speed=36.0,
                min_speed=0.0,
                max_drive_speed=44.0,
                lost_forward_speed=32.0,
                stall_pwm=30.0,
                kp=0.55,
                ki=0.0,
                kd=0.14,
                i_limit=0.0,
                derivative_alpha=0.82,
                deadband_px=8.0,
                max_correction=24.0,
                max_correction_slew_per_sec=140.0,
                pivot_threshold_px=34.0,
                pivot_speed=74.0,
                pivot_exit_px=10.0,
                pivot_timeout_s=0.45,
                pivot_brake_ms=40.0,
                lost_timeout_s=0.12,
                search_speed=48.0,
                swing_pivot=False,
                pivot_reverse_scale=0.95,
                blur_kernel=(7, 7),
                morph_kernel_size=5,
                priority_erode_iterations=2,
                priority_min_contour_area=160.0,
                priority_scan_start_ratio=0.42,
                priority_side_margin_px=14,
                priority_side_penalty=1200.0,
                priority_continuity_weight=7.5,
                contour_continuity_weight=12.0,
                target_smoothing_alpha=0.24,
                target_jump_px=28.0,
                target_jump_alpha=0.08,
                priority_color_min_contour_area=260.0,
                priority_color_bottom_contact_ratio=0.012,
                black_bgr_upper=135,
                black_v_multiplier=0.70,
            ),
        )
        # Use the same Week2 technique on the blue branch mask so branch pickup
        # behaves like the old follower, just with the simpler control law.
        self.blue_line_follower = Week2LineFollower(
            Week2LineFollowConfig(
                line_color="BLUE",
                roi_top_ratio=0.45,
                min_line_pixels=80,
                scan_rows=16,
                min_valid_rows=3,
                base_speed=36.0,
                min_speed=28.0,
                lost_forward_speed=30.0,
                stall_pwm=30.0,
                kp=0.26,
                ki=0.002,
                kd=0.04,
                i_limit=60.0,
                derivative_alpha=0.82,
                deadband_px=6.0,
                max_correction=30.0,
                max_correction_slew_per_sec=130.0,
                pivot_threshold_px=24.0,
                pivot_speed=68.0,
                pivot_exit_px=10.0,
                pivot_timeout_s=0.45,
                pivot_brake_ms=25.0,
                lost_timeout_s=0.10,
                search_speed=46.0,
                swing_pivot=False,
                pivot_reverse_scale=0.90,
                blur_kernel=(7, 7),
                morph_kernel_size=5,
                contour_continuity_weight=12.0,
                target_smoothing_alpha=0.24,
                target_jump_px=28.0,
                target_jump_alpha=0.08,
                blue_bgr_min=36,
                blue_dom_min_diff=8,
                blue_min_saturation=18,
            ),
        )

        self.client = PiClient(SERVER_IP, SERVER_PORT)
        self.cap = None
        self.running = True
        self._last_symbol_time = 0.0
        self._last_symbol_request_time = 0.0
        self._frame_index = 0
        self._pending_label = None
        self._pending_hits = 0
        self._vote_history = []          # sliding window of recent detections
        self._preferred_turn = None      # remembered direction for the next branch
        self._preferred_turn_set_at = 0.0
        self._preferred_turn_blue_frames = 0
        self._follow_blue_until = 0.0

        # Motor handles
        self._pwm_a = None
        self._pwm_b = None

    # -----------------------------------------------------------------
    # GPIO / Motor helpers
    # -----------------------------------------------------------------
    def setup_gpio(self):
        if not ON_PI:
            return
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        for pin in (ENA, IN1, IN2, ENB, IN3, IN4):
            GPIO.setup(pin, GPIO.OUT)
        self._pwm_a = GPIO.PWM(ENA, PWM_FREQ)
        self._pwm_b = GPIO.PWM(ENB, PWM_FREQ)
        self._pwm_a.start(0)
        self._pwm_b.start(0)

    def set_motors(self, left_speed: float, right_speed: float):
        """Set motor speeds in range [-100, 100].  Positive = forward."""
        if not ON_PI:
            return
        # Left motor (forward = IN1 LOW, IN2 HIGH)
        if left_speed >= 0:
            GPIO.output(IN1, GPIO.LOW)
            GPIO.output(IN2, GPIO.HIGH)
        else:
            GPIO.output(IN1, GPIO.HIGH)
            GPIO.output(IN2, GPIO.LOW)
        self._pwm_a.ChangeDutyCycle(min(abs(left_speed), 100))

        # Right motor (forward = IN3 LOW, IN4 HIGH)
        if right_speed >= 0:
            GPIO.output(IN3, GPIO.LOW)
            GPIO.output(IN4, GPIO.HIGH)
        else:
            GPIO.output(IN3, GPIO.HIGH)
            GPIO.output(IN4, GPIO.LOW)
        self._pwm_b.ChangeDutyCycle(min(abs(right_speed), 100))

    def stop_motors(self):
        self.set_motors(0, 0)

    @staticmethod
    def _apply_speed_floor(speed: float, floor_pwm: float) -> float:
        """Avoid weak forward commands that stall the motor.
        Slight negative (inner-wheel during turn) → stop at 0, not snap-reverse.
        Only values already well below -floor (deliberate reverse) are left alone.
        """
        if speed == 0.0:
            return 0.0
        if 0.0 < speed < floor_pwm:
            return floor_pwm          # too-slow forward → boost
        if -floor_pwm < speed < 0.0:
            return 0.0                # accidental reverse → stop (no jerk)
        return speed

    def _blue_matches_preferred_turn(self, blue_detection) -> bool:
        if not blue_detection.found or self._preferred_turn not in {"LEFT", "RIGHT"}:
            return False
        if self._preferred_turn == "LEFT":
            return blue_detection.error_px < -8.0
        return blue_detection.error_px > 8.0

    def _should_follow_blue(self, blue_detection, black_detection, frame_shape) -> bool:
        if (not blue_detection.found
                or blue_detection.centroid_y is None
                or blue_detection.contour_area < BLUE_MIN_AREA):
            return False

        frame_h = frame_shape[0]
        preferred_match = self._blue_matches_preferred_turn(blue_detection)
        y_gate_ratio = (
            BLUE_PREFERRED_Y_RATIO if preferred_match else BLUE_MIN_CENTROID_Y_RATIO
        )
        blue_close_enough = (
            blue_detection.centroid_y >= frame_h * y_gate_ratio
        )
        if not blue_close_enough and not preferred_match:
            return False

        if not black_detection.found:
            return True

        black_centroid_y = black_detection.centroid_y or 0
        area_ratio = (
            blue_detection.contour_area / max(1.0, black_detection.contour_area)
        )

        if preferred_match:
            return (
                blue_detection.centroid_y
                >= black_centroid_y - (BLUE_CENTROID_MARGIN_PX + 16.0)
                and area_ratio >= BLUE_PREFERRED_AREA_RATIO
            )

        return (
            blue_detection.centroid_y
            >= black_centroid_y - BLUE_CENTROID_MARGIN_PX
            and area_ratio >= BLUE_BLACK_AREA_RATIO
        )

    def _select_line_target(self, frame):
        now = time.monotonic()
        black_detection = self.line_follower.detect_line(frame)
        blue_detection = self.blue_line_follower.detect_line(frame)
        follow_blue = self._should_follow_blue(
            blue_detection, black_detection, frame.shape,
        )
        if follow_blue:
            self._follow_blue_until = now + BLUE_LATCH_SECONDS
        elif now < self._follow_blue_until and blue_detection.found:
            follow_blue = True
        else:
            self._follow_blue_until = 0.0

        if follow_blue:
            return self.blue_line_follower, blue_detection, True
        return self.line_follower, black_detection, False

    def _clear_preferred_turn(self, reason: str | None = None):
        if self._preferred_turn and reason:
            print(f"[Robot] Clearing preferred turn {self._preferred_turn}: {reason}")
        self._preferred_turn = None
        self._preferred_turn_set_at = 0.0
        self._preferred_turn_blue_frames = 0

    def _update_preferred_turn_progress(self, following_blue, detection):
        if self._preferred_turn not in {"LEFT", "RIGHT"}:
            self._preferred_turn_blue_frames = 0
            return

        if (
            following_blue
            and detection.found
            and self._blue_matches_preferred_turn(detection)
        ):
            self._preferred_turn_blue_frames += 1
            if self._preferred_turn_blue_frames >= PREFERRED_TURN_BLUE_FRAMES:
                self._clear_preferred_turn("preferred branch taken")
            return

        if detection.found:
            self._preferred_turn_blue_frames = 0

    # -----------------------------------------------------------------
    # Camera
    # -----------------------------------------------------------------
    def open_camera(self):
        self._picam2 = None
        self.cap = None

        if HAS_PICAMERA2:
            try:
                cam_info = Picamera2.global_camera_info()
                if not cam_info:
                    raise RuntimeError("Picamera2 reports no cameras")

                self._picam2 = Picamera2()
                config = self._picam2.create_video_configuration(
                    main={"size": (FRAME_W, FRAME_H), "format": "RGB888"},
                    queue=False,
                    controls={
                        "FrameRate": CAMERA_FRAME_RATE,
                        "AeEnable": True,
                        "AwbEnable": True,
                    },
                )
                self._picam2.configure(config)
                self._picam2.start()
                time.sleep(CAMERA_SETTLE_S)
                try:
                    metadata = self._picam2.capture_metadata()
                    colour_gains = metadata.get(
                        "ColourGains", CAMERA_FALLBACK_COLOUR_GAINS,
                    )
                    self._picam2.set_controls(
                        {
                            "AwbEnable": False,
                            "ColourGains": tuple(colour_gains),
                        },
                    )
                    print(
                        "[Robot] Camera AWB locked at gains "
                        f"{tuple(round(float(v), 2) for v in colour_gains)}"
                    )
                except Exception as lock_err:
                    self._picam2.set_controls(
                        {
                            "AwbEnable": False,
                            "ColourGains": CAMERA_FALLBACK_COLOUR_GAINS,
                        },
                    )
                    print(
                        "[Robot] Camera AWB lock fallback in use: "
                        f"{lock_err}"
                    )
                print("[Robot] Camera opened via Picamera2")
                return
            except Exception as e:
                self._picam2 = None
                print(f"[Robot] Picamera2 init failed: {e}")
                print("[Robot] Falling back to OpenCV /dev/video0")

        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        if not self.cap.isOpened():
            raise RuntimeError(
                "No camera available. Picamera2 found no CSI camera and /dev/video0 "
                "could not be opened. Check camera cable orientation, enable camera "
                "interface, and ensure no other process is using the camera."
            )
        print("[Robot] Camera opened via OpenCV /dev/video0")

    def read_frame(self):
        """Return a BGR frame, or None on failure."""
        if self._picam2:
            frame = self._picam2.capture_array()
            if frame is None:
                return None
            # Picamera2 RGB888 → OpenCV BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            ret, frame = self.cap.read()
            if not ret:
                return None

        if ROTATE_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        return frame

    # -----------------------------------------------------------------
    # Try to send frame to Mac for symbol detection (non-blocking-ish)
    # -----------------------------------------------------------------
    def try_detect_symbol(self, frame, error_px):
        """Send frame to Mac periodically. Uses a sliding-window majority
        vote instead of consecutive hits for faster, noise-tolerant confirmation.
        """
        if self._preferred_turn:
            # Ignore fresh arrow detections until the remembered turn is consumed.
            pass

        # Cooldown after last action
        if time.monotonic() - self._last_symbol_time < SYMBOL_COOLDOWN:
            return None

        # Turn guard — don't detect during turns/curves
        if abs(error_px) > TURN_GUARD_ERROR_PX:
            self._vote_history.clear()
            return None

        # Only send every N-th frame
        self._frame_index += 1
        if self._frame_index % SYMBOL_SEND_INTERVAL != 0:
            return None

        now = time.monotonic()
        if now - self._last_symbol_request_time < SYMBOL_REQUEST_GAP:
            return None
        self._last_symbol_request_time = now

        try:
            cmd = self.client.send_frame(frame)
        except ConnectionError:
            return None

        if self._preferred_turn and cmd in {"LEFT", "RIGHT"}:
            return None

        if cmd and cmd != "FORWARD":
            self._vote_history.append(cmd)
            # Trim to window size
            if len(self._vote_history) > SYMBOL_VOTE_WINDOW:
                self._vote_history.pop(0)

            # Count votes for the latest label
            count = self._vote_history.count(cmd)
            required_votes = (
                ARROW_VOTE_THRESHOLD if cmd in {"LEFT", "RIGHT"}
                else SYMBOL_VOTE_THRESHOLD
            )
            print(f"[Robot] Detection: {cmd} (votes: {count}/"
                  f"{required_votes} in last {len(self._vote_history)})")

            if count >= required_votes:
                confirmed = cmd
                self._vote_history.clear()
                return confirmed
        else:
            # FORWARD/empty — add to history as noise dilution
            self._vote_history.append("FORWARD")
            if len(self._vote_history) > SYMBOL_VOTE_WINDOW:
                self._vote_history.pop(0)

        return None

    # -----------------------------------------------------------------
    # Command execution
    # -----------------------------------------------------------------
    TURN_SPEED = 50              # duty cycle for deliberate spins / recovery turns
    TURN_TIMEOUT = 4.0           # max seconds to search for line

    def _turn_until_line(self, left_speed, right_speed):
        """Spin at the given speeds until the line is re-acquired or timeout."""
        # Phase 1: blind turn to clear the current line
        self.set_motors(left_speed, right_speed)
        time.sleep(0.5)

        # Wait for the line to DISAPPEAR (ensures we've left current line)
        lost_start = time.monotonic()
        while time.monotonic() - lost_start < 0.8:
            frame = self.read_frame()
            if frame is None:
                continue
            _, det, _ = self._select_line_target(frame)
            if not det.found:
                break

        # Keep turning until the new line is found
        start = time.monotonic()
        while time.monotonic() - start < self.TURN_TIMEOUT:
            frame = self.read_frame()
            if frame is None:
                continue
            _, det, _ = self._select_line_target(frame)
            if det.found:
                print("[Robot] Line re-acquired after turn")
                self.stop_motors()
                return
            self.set_motors(left_speed, right_speed)

        print("[Robot] Turn timeout — line not found")
        self.stop_motors()

    def execute_command(self, cmd: str):
        """Drive motors according to the received command."""
        print(f"[Robot] Executing: {cmd}")

        if cmd == "FORWARD":
            # No-op — just resume line following
            pass

        elif cmd == "LEFT":
            # Remember direction for the next branch / corner.
            self._preferred_turn = "LEFT"
            self._preferred_turn_set_at = time.monotonic()
            self._preferred_turn_blue_frames = 0
            print("[Robot] Arrow detected → bias LEFT on the next turn")

        elif cmd == "RIGHT":
            # Remember direction for the next branch / corner.
            self._preferred_turn = "RIGHT"
            self._preferred_turn_set_at = time.monotonic()
            self._preferred_turn_blue_frames = 0
            print("[Robot] Arrow detected → bias RIGHT on the next turn")

        elif cmd == "BACKWARD":
            # Reverse briefly then spin 180°
            self.set_motors(-BASE_SPEED, -BASE_SPEED)
            time.sleep(0.4)
            self._turn_until_line(-self.TURN_SPEED, self.TURN_SPEED)

        elif cmd == "STOP":
            self.stop_motors()

        elif cmd == "ROTATE_360":
            # Full 360 spin — find line twice (ignore the first one)
            print("[Robot] Spinning 360 — will skip first line, stop on second")
            self.set_motors(self.TURN_SPEED, -self.TURN_SPEED)
            time.sleep(0.5)  # blind start

            lines_found = 0
            was_on_line = True  # assume starting on a line
            start = time.monotonic()
            while time.monotonic() - start < 6.0:  # 6s safety timeout
                frame = self.read_frame()
                if frame is None:
                    continue
                _, det, _ = self._select_line_target(frame)
                if det.found and not was_on_line:
                    lines_found += 1
                    print(f"[Robot] 360 spin: line found #{lines_found}")
                    if lines_found >= 2:
                        break
                was_on_line = det.found
                self.set_motors(self.TURN_SPEED, -self.TURN_SPEED)

            self.stop_motors()
            print(f"[Robot] 360 spin complete (found {lines_found} lines)")

        elif cmd in ("PRINT_QR", "PRINT_FINGERPRINT"):
            print(f"[Robot] Action: {cmd}")
            self.stop_motors()
            time.sleep(1.0)

        else:
            print(f"[Robot] Unknown command: {cmd}")
            self.stop_motors()

    # -----------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------
    def run(self):
        self.setup_gpio()
        self.open_camera()
        print("[Robot] Running — press Ctrl+C to stop")

        try:
            while self.running:
                frame = self.read_frame()
                if frame is None:
                    continue

                # ---- STATE: FOLLOWING --------------------------------
                if self.state == FOLLOWING:
                    active_follower, detection, following_blue = (
                        self._select_line_target(frame)
                    )

                    motor_cmd = active_follower.compute_steering(
                        detection, preferred_turn=self._preferred_turn,
                    )

                    if self._preferred_turn:
                        turn_age = time.monotonic() - self._preferred_turn_set_at
                        if turn_age > PREFERRED_TURN_TIMEOUT:
                            self._clear_preferred_turn(
                                f"timed out after {turn_age:.1f}s"
                            )

                    self._update_preferred_turn_progress(
                        following_blue, detection,
                    )

                    # Trust the follower's speed planning; only boost commands
                    # that are too weak to move the actual motors on gentle
                    # tracking. During sharp turns or recovery, preserve the
                    # commanded wheel split instead of boosting the inner wheel.
                    if motor_cmd.recovery_mode or abs(motor_cmd.correction) >= 24.0:
                        left = motor_cmd.left_speed
                        right = motor_cmd.right_speed
                    else:
                        left = self._apply_speed_floor(
                            motor_cmd.left_speed, FOLLOW_STALL_PWM
                        )
                        right = self._apply_speed_floor(
                            motor_cmd.right_speed, FOLLOW_STALL_PWM
                        )
                    left = max(-100.0, min(100.0, left))
                    right = max(-100.0, min(100.0, right))
                    self.set_motors(left, right)

                    if DEBUG_PREVIEW:
                        debug = frame.copy()
                        draw_line_debug(debug, detection)
                        h = frame.shape[0]
                        label = f"L={left:.0f} R={right:.0f}"
                        if detection.found:
                            label += f" e={detection.error_px:.0f}"
                            if following_blue:
                                label += " BLUE"
                        else:
                            label += " LOST"
                        cv2.putText(debug, label, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (255, 255, 0), 2)
                        state_text = self.state
                        if self._preferred_turn:
                            state_text += f" next:{self._preferred_turn}"
                        if self._vote_history:
                            from collections import Counter
                            top = Counter(self._vote_history).most_common(1)
                            if top and top[0][0] != "FORWARD":
                                state_text += f" [{top[0][0]} x{top[0][1]}]"
                        cv2.putText(debug, state_text, (10, h - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 255, 0), 2)
                        cv2.imshow("Pi Debug", debug)
                        cv2.waitKey(1)

                    # Symbol detection while moving — skip during turns to keep loop fast
                    confirmed_cmd = None
                    if (detection.found
                            and not motor_cmd.recovery_mode
                            and abs(motor_cmd.correction) < 22):
                        confirmed_cmd = self.try_detect_symbol(
                            frame, detection.error_px
                        )
                    if confirmed_cmd:
                            print(f"[Robot] Confirmed symbol: {confirmed_cmd}")
                            self.state = EXECUTING_ACTION
                            self._last_symbol_time = time.monotonic()
                            self.stop_motors()
                            self.execute_command(confirmed_cmd)

                # ---- STATE: EXECUTING_ACTION -------------------------
                elif self.state == EXECUTING_ACTION:
                    self.line_follower.reset_steering()
                    self.blue_line_follower.reset_steering()
                    self._preferred_turn_blue_frames = 0
                    self.state = FOLLOWING
                    print("[Robot] Returning to FOLLOWING")

        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()

    # -----------------------------------------------------------------
    def cleanup(self):
        print("[Robot] Cleaning up …")
        self.running = False
        self.stop_motors()
        if self._picam2:
            self._picam2.stop()
        elif self.cap:
            self.cap.release()
        self.client.close()
        if DEBUG_PREVIEW:
            cv2.destroyAllWindows()
        if ON_PI:
            GPIO.cleanup()


# =====================================================================
if __name__ == "__main__":
    robot = Robot()
    signal.signal(signal.SIGINT, lambda *_: setattr(robot, "running", False))
    robot.run()
