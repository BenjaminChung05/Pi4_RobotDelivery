import time
import cv2
import numpy as np
import RPi.GPIO as GPIO
from picamera2 import Picamera2

# =============================================================================
# 1) CONFIG
# =============================================================================
FRAME_W, FRAME_H = 320, 240
ROI_TOP, ROI_BOTTOM = 100, 240
ROI_LEFT, ROI_RIGHT = 0, 320

ROTATE_180 = True
USE_OTSU = True
THRESHOLD_VAL = 60  # used only if USE_OTSU = False

# Centerline extraction
SCAN_ROWS = 20
MIN_VALID_ROWS = 8
MIN_LINE_PIXELS = 180
CENTER_OFFSET_PX = 0.0

# Motor / speed
MAX_SPEED = 75.0
BASE_SPEED = 40.0
MIN_SPEED = 30.0
LOST_FORWARD_SPEED = 24.0
SEARCH_SPIN_SPEED = 45.0
LOST_TIMEOUT_S = 0.10
STALL_PWM = 26.0
SHARP_TURN_ERROR_PX = 55.0
SHARP_PIVOT_SPEED = 48.0

# PID tuned for smoothness (low zig-zag)
Kp = 0.36
Ki = 0.004
Kd = 0.060
I_LIMIT = 120.0
DERIVATIVE_ALPHA = 0.80
ERROR_DEADBAND_PX = 3.0
MAX_CORRECTION = 38.0
MAX_CORRECTION_SLEW_PER_SEC = 320.0

# Display
SHOW_PREVIEW_WINDOW = True
SHOW_MASK_WINDOW = False

# =============================================================================
# 2) MOTOR PINS
# =============================================================================
ENA, IN1, IN2 = 12, 17, 18
ENB, IN3, IN4 = 13, 22, 23

LEFT_MOTOR_INVERT = False
RIGHT_MOTOR_INVERT = False

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
pwm_left = None
pwm_right = None


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def init_motor_gpio() -> None:
    GPIO.cleanup()
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for pin in (ENA, IN1, IN2, ENB, IN3, IN4):
        GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)


def set_motor(left_speed: float, right_speed: float) -> None:
    global pwm_left, pwm_right

    left_speed = clamp(float(left_speed), -MAX_SPEED, MAX_SPEED)
    right_speed = clamp(float(right_speed), -MAX_SPEED, MAX_SPEED)

    if LEFT_MOTOR_INVERT:
        left_speed = -left_speed
    if RIGHT_MOTOR_INVERT:
        right_speed = -right_speed

    # Compensate for motor dead-zone to avoid stalling at low duty cycles.
    if 0.0 < abs(left_speed) < STALL_PWM:
        left_speed = STALL_PWM if left_speed > 0 else -STALL_PWM
    if 0.0 < abs(right_speed) < STALL_PWM:
        right_speed = STALL_PWM if right_speed > 0 else -STALL_PWM

    if left_speed >= 0:
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH)
    else:
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.LOW)

    if right_speed >= 0:
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.HIGH)
    else:
        GPIO.output(IN3, GPIO.HIGH)
        GPIO.output(IN4, GPIO.LOW)

    pwm_left.ChangeDutyCycle(abs(left_speed))
    pwm_right.ChangeDutyCycle(abs(right_speed))


def stop_motors() -> None:
    if pwm_left is not None:
        pwm_left.ChangeDutyCycle(0)
    if pwm_right is not None:
        pwm_right.ChangeDutyCycle(0)
    GPIO.output([IN1, IN2, IN3, IN4], GPIO.LOW)


# =============================================================================
# 3) PID
# =============================================================================
class PIDController:
    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.d_filtered = 0.0

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0
        self.d_filtered = 0.0

    def update(self, error: float, dt: float) -> float:
        dt = max(dt, 1e-3)

        self.integral += error * dt
        self.integral = clamp(self.integral, -I_LIMIT, I_LIMIT)

        d_raw = (error - self.prev_error) / dt
        self.d_filtered = (DERIVATIVE_ALPHA * self.d_filtered) + ((1.0 - DERIVATIVE_ALPHA) * d_raw)
        self.prev_error = error

        return (self.kp * error) + (self.ki * self.integral) + (self.kd * self.d_filtered)


# =============================================================================
# 4) VISION
# =============================================================================
def build_line_mask(roi_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    if USE_OTSU:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(gray, THRESHOLD_VAL, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def row_center_x(row: np.ndarray) -> float | None:
    xs = np.where(row > 0)[0]
    if xs.size < 4:
        return None
    return float(np.mean(xs))


def detect_centerline(mask: np.ndarray) -> tuple[float | None, int]:
    h, w = mask.shape
    total = int(np.count_nonzero(mask))
    if total < MIN_LINE_PIXELS:
        return None, 0

    ys = np.linspace(h - 1, 0, SCAN_ROWS, dtype=int)
    centers = []

    for y in ys:
        cx = row_center_x(mask[y, :])
        if cx is not None:
            centers.append(cx)

    if len(centers) < MIN_VALID_ROWS:
        return None, len(centers)

    # Median gives stable centerline and rejects row outliers.
    return float(np.median(centers)), len(centers)


# =============================================================================
# 5) MAIN LOOP
# =============================================================================
def main() -> None:
    global pwm_left, pwm_right

    init_motor_gpio()
    pwm_left = GPIO.PWM(ENA, 1000)
    pwm_right = GPIO.PWM(ENB, 1000)
    pwm_left.start(0)
    pwm_right.start(0)

    picam2 = Picamera2()
    picam2.configure(
        picam2.create_video_configuration(main={"size": (FRAME_W, FRAME_H), "format": "RGB888"})
    )
    picam2.start()

    roi_w = ROI_RIGHT - ROI_LEFT
    roi_cx = roi_w / 2.0
    target_cx = roi_cx + CENTER_OFFSET_PX

    pid = PIDController(Kp, Ki, Kd)
    last_time = time.time()
    last_seen = last_time
    last_error_sign = 1.0
    last_correction = 0.0

    time.sleep(1.0)
    print("Centerline PID follower started. Press Ctrl+C to stop.")

    try:
        while True:
            frame_rgb = picam2.capture_array()
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            if ROTATE_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            roi = frame[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT]
            mask = build_line_mask(roi)

            now = time.time()
            dt = max(now - last_time, 1e-3)
            last_time = now

            cx, valid_rows = detect_centerline(mask)

            if cx is not None:
                last_seen = now
                error = cx - target_cx

                if abs(error) < ERROR_DEADBAND_PX:
                    error = 0.0

                if abs(error) > 1.0:
                    last_error_sign = 1.0 if error > 0 else -1.0

                # Immediate sharp-turn action based on last seen line direction.
                # Left turn: left motor forward(+), right motor reverse(-).
                if abs(error) >= SHARP_TURN_ERROR_PX:
                    pid.reset()
                    last_correction = 0.0
                    if last_error_sign < 0:
                        set_motor(SHARP_PIVOT_SPEED, -SHARP_PIVOT_SPEED)
                    else:
                        set_motor(-SHARP_PIVOT_SPEED, SHARP_PIVOT_SPEED)
                    time.sleep(0.01)
                    continue

                correction = pid.update(error, dt)
                correction = clamp(correction, -MAX_CORRECTION, MAX_CORRECTION)

                # Limit correction rate to suppress rapid oscillation.
                max_step = MAX_CORRECTION_SLEW_PER_SEC * dt
                correction = clamp(correction, last_correction - max_step, last_correction + max_step)
                last_correction = correction

                # Reduce forward speed while turning harder.
                turn_ratio = min(1.0, abs(correction) / max(1.0, MAX_CORRECTION))
                forward = BASE_SPEED - (BASE_SPEED - MIN_SPEED) * turn_ratio

                left_speed = forward + correction
                right_speed = forward - correction
                set_motor(left_speed, right_speed)
            else:
                pid.reset()
                last_correction = 0.0
                if (now - last_seen) >= LOST_TIMEOUT_S:
                    spin = SEARCH_SPIN_SPEED * last_error_sign
                    set_motor(spin, -spin)
                else:
                    set_motor(LOST_FORWARD_SPEED, LOST_FORWARD_SPEED)

            if SHOW_PREVIEW_WINDOW:
                preview = frame.copy()
                cv2.rectangle(preview, (ROI_LEFT, ROI_TOP), (ROI_RIGHT, ROI_BOTTOM), (255, 220, 0), 1)

                if cx is not None:
                    draw_x = int(cx + ROI_LEFT)
                    draw_y = int((ROI_TOP + ROI_BOTTOM) / 2)
                    cv2.circle(preview, (draw_x, draw_y), 5, (0, 255, 0), -1)

                target_draw_x = int(target_cx + ROI_LEFT)
                cv2.line(preview, (target_draw_x, ROI_TOP), (target_draw_x, ROI_BOTTOM), (0, 0, 255), 1)

                text = f"cx={-1 if cx is None else int(cx)} rows={valid_rows}/{SCAN_ROWS}"
                cv2.putText(preview, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 255, 50), 1, cv2.LINE_AA)

                cv2.imshow("Line Follower", preview)
                if SHOW_MASK_WINDOW:
                    cv2.imshow("Mask", mask)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        stop_motors()
        try:
            picam2.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        GPIO.cleanup()
        print("Stopped.")


if __name__ == "__main__":
    main()
