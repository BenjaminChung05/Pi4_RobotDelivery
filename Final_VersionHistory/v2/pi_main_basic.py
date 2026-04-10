"""Simpler Raspberry Pi main script.

Keeps the same camera, motor, symbol-voting, and action flow as `pi_main.py`,
but uses a basic line detector:
- follow the darkest valid black line
- also detect light blue and navy blue lines
- switch to blue when the black line is gone or when an arrow-selected branch
  is visible on the requested side
"""

import signal
import time

import cv2

from basic_color_line_follow import (
    BasicColorLineFollowConfig,
    BasicColorLineFollower,
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
    print("[pi_main_basic] Picamera2 not available - will use cv2.VideoCapture")

from pi_client import PiClient

DEBUG_PREVIEW = True
ROTATE_180 = False

FRAME_W, FRAME_H = 640, 480

ENA, IN1, IN2 = 12, 17, 18
ENB, IN3, IN4 = 13, 22, 23
PWM_FREQ = 500

BASE_SPEED = 42
FOLLOW_STALL_PWM = 30.0

BLUE_MIN_AREA = 110.0
BLUE_BRANCH_MIN_AREA_RATIO = 0.10
BLUE_MIN_CENTROID_Y_RATIO = 0.26
BLUE_CENTROID_MARGIN_PX = 64.0
BLUE_LATCH_SECONDS = 0.35

CAMERA_FRAME_RATE = 60.0
CAMERA_SETTLE_S = 0.35
CAMERA_FALLBACK_COLOUR_GAINS = (1.50, 1.20)

TURN_GUARD_ERROR_PX = 40
SYMBOL_SEND_INTERVAL = 2
SYMBOL_REQUEST_GAP = 0.40
SYMBOL_VOTE_WINDOW = 6
SYMBOL_VOTE_THRESHOLD = 4
ARROW_VOTE_THRESHOLD = 5
SYMBOL_COOLDOWN = 2.0
PREFERRED_TURN_TIMEOUT = 12.0
PREFERRED_TURN_BLUE_FRAMES = 4

SERVER_IP = "10.134.9.249"
SERVER_PORT = 5001

FOLLOWING = "FOLLOWING"
EXECUTING_ACTION = "EXECUTING_ACTION"


class Robot:
    TURN_SPEED = 50
    TURN_TIMEOUT = 4.0

    def __init__(self):
        self.state = FOLLOWING
        self.black_line_follower = BasicColorLineFollower(
            BasicColorLineFollowConfig(
                line_color="BLACK",
                roi_top_ratio=0.50,
                roi_bottom_ratio=1.0,
                min_contour_area=180.0,
                min_bottom_contact_cols=12,
                base_speed=36.0,
                min_speed=28.0,
                max_speed=48.0,
                steer_gain=0.25,
                d_gain=0.09,
                max_correction=24.0,
                max_correction_step=10.0,
                tank_turn_error_px=68.0,
                tank_turn_speed=62.0,
                lost_forward_speed=30.0,
                search_speed=46.0,
            ),
        )
        self.blue_line_follower = BasicColorLineFollower(
            BasicColorLineFollowConfig(
                line_color="BLUE",
                roi_top_ratio=0.45,
                roi_bottom_ratio=1.0,
                min_contour_area=120.0,
                min_bottom_contact_cols=8,
                base_speed=34.0,
                min_speed=26.0,
                max_speed=46.0,
                steer_gain=0.22,
                d_gain=0.07,
                max_correction=22.0,
                max_correction_step=9.0,
                tank_turn_error_px=62.0,
                tank_turn_speed=58.0,
                lost_forward_speed=28.0,
                search_speed=44.0,
            ),
        )

        self.client = PiClient(SERVER_IP, SERVER_PORT)
        self.cap = None
        self._picam2 = None
        self.running = True

        self._last_symbol_time = 0.0
        self._last_symbol_request_time = 0.0
        self._frame_index = 0
        self._vote_history: list[str] = []
        self._preferred_turn = None
        self._preferred_turn_set_at = 0.0
        self._preferred_turn_blue_frames = 0
        self._follow_blue_until = 0.0

        self._pwm_a = None
        self._pwm_b = None

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
        if not ON_PI:
            return

        if left_speed >= 0:
            GPIO.output(IN1, GPIO.LOW)
            GPIO.output(IN2, GPIO.HIGH)
        else:
            GPIO.output(IN1, GPIO.HIGH)
            GPIO.output(IN2, GPIO.LOW)
        self._pwm_a.ChangeDutyCycle(min(abs(left_speed), 100))

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
        if speed == 0.0:
            return 0.0
        if 0.0 < speed < floor_pwm:
            return floor_pwm
        if -floor_pwm < speed < 0.0:
            return 0.0
        return speed

    def _blue_matches_preferred_turn(self, blue_detection) -> bool:
        if not blue_detection.found or self._preferred_turn not in {"LEFT", "RIGHT"}:
            return False
        if self._preferred_turn == "LEFT":
            return blue_detection.error_px < -8.0
        return blue_detection.error_px > 8.0

    def _should_follow_blue(self, blue_detection, black_detection, frame_shape) -> bool:
        if (
            not blue_detection.found
            or blue_detection.centroid_y is None
            or blue_detection.contour_area < BLUE_MIN_AREA
        ):
            return False

        if not black_detection.found:
            return True

        frame_h = frame_shape[0]
        preferred_match = self._blue_matches_preferred_turn(blue_detection)
        if preferred_match:
            return blue_detection.centroid_y >= frame_h * 0.18

        black_centroid_y = black_detection.centroid_y or 0
        area_ratio = (
            blue_detection.contour_area / max(1.0, black_detection.contour_area)
        )
        return (
            blue_detection.centroid_y >= frame_h * BLUE_MIN_CENTROID_Y_RATIO
            and blue_detection.centroid_y >= black_centroid_y - BLUE_CENTROID_MARGIN_PX
            and area_ratio >= BLUE_BRANCH_MIN_AREA_RATIO
        )

    def _select_line_target(self, frame):
        now = time.monotonic()
        black_detection = self.black_line_follower.detect_line(frame)
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
        return self.black_line_follower, black_detection, False

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
                self._clear_preferred_turn("preferred blue branch taken")
            return

        if detection.found:
            self._preferred_turn_blue_frames = 0

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
                except Exception:
                    self._picam2.set_controls(
                        {
                            "AwbEnable": False,
                            "ColourGains": CAMERA_FALLBACK_COLOUR_GAINS,
                        },
                    )
                print("[Robot] Camera opened via Picamera2")
                return
            except Exception as exc:
                self._picam2 = None
                print(f"[Robot] Picamera2 init failed: {exc}")
                print("[Robot] Falling back to OpenCV /dev/video0")

        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        if not self.cap.isOpened():
            raise RuntimeError(
                "No camera available. Picamera2 found no CSI camera and /dev/video0 "
                "could not be opened."
            )
        print("[Robot] Camera opened via OpenCV /dev/video0")

    def read_frame(self):
        if self._picam2:
            frame = self._picam2.capture_array()
            if frame is None:
                return None
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            ret, frame = self.cap.read()
            if not ret:
                return None

        if ROTATE_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        return frame

    def try_detect_symbol(self, frame, error_px):
        if time.monotonic() - self._last_symbol_time < SYMBOL_COOLDOWN:
            return None

        if abs(error_px) > TURN_GUARD_ERROR_PX:
            self._vote_history.clear()
            return None

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
            if len(self._vote_history) > SYMBOL_VOTE_WINDOW:
                self._vote_history.pop(0)

            count = self._vote_history.count(cmd)
            required_votes = (
                ARROW_VOTE_THRESHOLD if cmd in {"LEFT", "RIGHT"}
                else SYMBOL_VOTE_THRESHOLD
            )
            print(
                f"[Robot] Detection: {cmd} (votes: {count}/"
                f"{required_votes} in last {len(self._vote_history)})"
            )
            if count >= required_votes:
                self._vote_history.clear()
                return cmd
        else:
            self._vote_history.append("FORWARD")
            if len(self._vote_history) > SYMBOL_VOTE_WINDOW:
                self._vote_history.pop(0)

        return None

    def _turn_until_line(self, left_speed, right_speed):
        self.set_motors(left_speed, right_speed)
        time.sleep(0.5)

        lost_start = time.monotonic()
        while time.monotonic() - lost_start < 0.8:
            frame = self.read_frame()
            if frame is None:
                continue
            _, det, _ = self._select_line_target(frame)
            if not det.found:
                break

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

        print("[Robot] Turn timeout - line not found")
        self.stop_motors()

    def execute_command(self, cmd: str):
        print(f"[Robot] Executing: {cmd}")

        if cmd == "FORWARD":
            return
        if cmd == "LEFT":
            self._preferred_turn = "LEFT"
            self._preferred_turn_set_at = time.monotonic()
            self._preferred_turn_blue_frames = 0
            print("[Robot] Arrow detected -> bias LEFT on the next turn")
            return
        if cmd == "RIGHT":
            self._preferred_turn = "RIGHT"
            self._preferred_turn_set_at = time.monotonic()
            self._preferred_turn_blue_frames = 0
            print("[Robot] Arrow detected -> bias RIGHT on the next turn")
            return
        if cmd == "BACKWARD":
            self.set_motors(-BASE_SPEED, -BASE_SPEED)
            time.sleep(0.4)
            self._turn_until_line(-self.TURN_SPEED, self.TURN_SPEED)
            return
        if cmd == "STOP":
            self.stop_motors()
            return
        if cmd == "ROTATE_360":
            print("[Robot] Spinning 360 - will skip first line, stop on second")
            self.set_motors(self.TURN_SPEED, -self.TURN_SPEED)
            time.sleep(0.5)

            lines_found = 0
            was_on_line = True
            start = time.monotonic()
            while time.monotonic() - start < 6.0:
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
            return
        if cmd in {"PRINT_QR", "PRINT_FINGERPRINT"}:
            print(f"[Robot] Action: {cmd}")
            self.stop_motors()
            time.sleep(1.0)
            return

        print(f"[Robot] Unknown command: {cmd}")
        self.stop_motors()

    def run(self):
        self.setup_gpio()
        self.open_camera()
        print("[Robot] Running - press Ctrl+C to stop")

        try:
            while self.running:
                frame = self.read_frame()
                if frame is None:
                    continue

                if self.state == FOLLOWING:
                    active_follower, detection, following_blue = self._select_line_target(frame)
                    motor_cmd = active_follower.compute_steering(
                        detection,
                        preferred_turn=self._preferred_turn,
                    )

                    if self._preferred_turn:
                        turn_age = time.monotonic() - self._preferred_turn_set_at
                        if turn_age > PREFERRED_TURN_TIMEOUT:
                            self._clear_preferred_turn(
                                f"timed out after {turn_age:.1f}s"
                            )

                    self._update_preferred_turn_progress(following_blue, detection)

                    if motor_cmd.recovery_mode or abs(motor_cmd.correction) >= 22.0:
                        left = motor_cmd.left_speed
                        right = motor_cmd.right_speed
                    else:
                        left = self._apply_speed_floor(
                            motor_cmd.left_speed, FOLLOW_STALL_PWM,
                        )
                        right = self._apply_speed_floor(
                            motor_cmd.right_speed, FOLLOW_STALL_PWM,
                        )

                    left = max(-100.0, min(100.0, left))
                    right = max(-100.0, min(100.0, right))
                    self.set_motors(left, right)

                    if DEBUG_PREVIEW:
                        debug = frame.copy()
                        draw_line_debug(debug, detection)
                        label = f"L={left:.0f} R={right:.0f}"
                        if detection.found:
                            label += f" e={detection.error_px:.0f}"
                            if following_blue:
                                label += " BLUE"
                            else:
                                label += " BLACK"
                        else:
                            label += " LOST"
                        cv2.putText(
                            debug,
                            label,
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 0),
                            2,
                        )
                        state_text = self.state
                        if self._preferred_turn:
                            state_text += f" next:{self._preferred_turn}"
                        cv2.putText(
                            debug,
                            state_text,
                            (10, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                        )
                        cv2.imshow("Pi Debug", debug)
                        cv2.waitKey(1)

                    confirmed_cmd = None
                    if (
                        detection.found
                        and not motor_cmd.recovery_mode
                        and abs(motor_cmd.correction) < 20.0
                    ):
                        confirmed_cmd = self.try_detect_symbol(
                            frame, detection.error_px,
                        )

                    if confirmed_cmd:
                        print(f"[Robot] Confirmed symbol: {confirmed_cmd}")
                        self.state = EXECUTING_ACTION
                        self._last_symbol_time = time.monotonic()
                        self.stop_motors()
                        self.execute_command(confirmed_cmd)

                elif self.state == EXECUTING_ACTION:
                    self.black_line_follower.reset_steering()
                    self.blue_line_follower.reset_steering()
                    self._preferred_turn_blue_frames = 0
                    self.state = FOLLOWING
                    print("[Robot] Returning to FOLLOWING")

        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()

    def cleanup(self):
        print("[Robot] Cleaning up ...")
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


if __name__ == "__main__":
    robot = Robot()
    signal.signal(signal.SIGINT, lambda *_: setattr(robot, "running", False))
    robot.run()
