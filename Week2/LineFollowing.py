import time
import cv2
import numpy as np
import RPi.GPIO as GPIO
from picamera2 import Picamera2

# =============================================================================
# CONFIG — tune these, nothing else needs touching
# =============================================================================

# Camera
FRAME_W, FRAME_H = 320, 240
ROTATE_180       = True

# Region of interest — rows the robot looks at
ROI_TOP    = 140        # raise if robot reacts too early to distant curves
ROI_BOTTOM = 240
ROI_LEFT   = 0
ROI_RIGHT  = 320

# Vision
USE_OTSU       = True
THRESHOLD_VAL  = 60     # only used when USE_OTSU = False
MIN_LINE_PX    = 150    # ignore blobs smaller than this
SCAN_ROWS      = 16
MIN_VALID_ROWS = 6

# Straight-line speeds (all as PWM duty cycle 0–100)
BASE_SPEED = 45.0       # normal forward speed
MIN_SPEED  = 35.0       # minimum forward speed during gentle curves

# Stall threshold — set to the PWM where your motors JUST start moving + 3
STALL_PWM  = 30.0

# PID gains
Kp = 0.40
Ki = 0.003
Kd = 0.08
I_LIMIT    = 80.0
D_ALPHA    = 0.75       # derivative filter (0=no filter, 1=frozen)
DEADBAND   = 4.0        # px error smaller than this → treated as zero
MAX_CORR   = 40.0       # maximum PID correction applied to each motor

# Sharp-turn pivot
# Pivot triggers when error exceeds this many pixels
PIVOT_THRESHOLD   = 35.0   # LOWERED — trigger pivot much earlier
PIVOT_SPEED       = 65.0   # raised for more torque
PIVOT_EXIT_PX     = 15.0
PIVOT_TIMEOUT     = 0.6    # slightly longer in case it needs more time
PIVOT_BRAKE_MS    = 40     # hard brake duration after pivot completes

# Lost-line recovery
LOST_TIMEOUT_S    = 0.0    # spin immediately when line is lost — no forward creep
SEARCH_SPEED      = 50.0

# Motor GPIO pins (BCM numbering)
ENA, IN1, IN2 = 12, 17, 18   # left motor
ENB, IN3, IN4 = 13, 22, 23   # right motor
PWM_FREQ      = 1000          # Hz — safe for all RPi GPIO backends

# Flip either motor if it runs backwards
LEFT_INVERT  = False
RIGHT_INVERT = False

# =============================================================================
# MOTOR DRIVER
# =============================================================================
_pwm_l = None
_pwm_r = None


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def motor_init():
    global _pwm_l, _pwm_r
    GPIO.cleanup()
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for pin in (ENA, IN1, IN2, ENB, IN3, IN4):
        GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
    _pwm_l = GPIO.PWM(ENA, PWM_FREQ)
    _pwm_r = GPIO.PWM(ENB, PWM_FREQ)
    _pwm_l.start(0)
    _pwm_r.start(0)


def motor_set(left: float, right: float):
    """Set motor speeds. Positive = forward, negative = reverse. Range ±100."""
    left  = _clamp(float(left),  -100.0, 100.0)
    right = _clamp(float(right), -100.0, 100.0)

    if LEFT_INVERT:  left  = -left
    if RIGHT_INVERT: right = -right

    # Dead-zone: bump any non-zero command that's below stall up to STALL_PWM.
    # Exact 0.0 bypasses this — it means truly stop.
    if 0.0 < abs(left)  < STALL_PWM: left  = STALL_PWM  if left  > 0 else -STALL_PWM
    if 0.0 < abs(right) < STALL_PWM: right = STALL_PWM  if right > 0 else -STALL_PWM

    # Direction pins
    GPIO.output(IN1, GPIO.LOW  if left  >= 0 else GPIO.HIGH)
    GPIO.output(IN2, GPIO.HIGH if left  >= 0 else GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW  if right >= 0 else GPIO.HIGH)
    GPIO.output(IN4, GPIO.HIGH if right >= 0 else GPIO.LOW)

    _pwm_l.ChangeDutyCycle(abs(left))
    _pwm_r.ChangeDutyCycle(abs(right))


def motor_stop():
    if _pwm_l: _pwm_l.ChangeDutyCycle(0)
    if _pwm_r: _pwm_r.ChangeDutyCycle(0)
    GPIO.output([IN1, IN2, IN3, IN4], GPIO.LOW)


# =============================================================================
# PID
# =============================================================================
class PID:
    def __init__(self):
        self.integral   = 0.0
        self.prev_error = 0.0
        self.d_filt     = 0.0

    def reset(self):
        self.integral   = 0.0
        self.prev_error = 0.0
        self.d_filt     = 0.0

    def update(self, error: float, dt: float) -> float:
        dt = max(dt, 1e-3)
        if abs(error) < DEADBAND:
            error = 0.0
        self.integral += error * dt
        self.integral  = _clamp(self.integral, -I_LIMIT, I_LIMIT)
        d_raw       = (error - self.prev_error) / dt
        self.d_filt = D_ALPHA * self.d_filt + (1.0 - D_ALPHA) * d_raw
        self.prev_error = error
        return _clamp(Kp * error + Ki * self.integral + Kd * self.d_filt,
                      -MAX_CORR, MAX_CORR)


# =============================================================================
# VISION
# =============================================================================
def get_line_center(frame_bgr) -> tuple[float | None, int]:
    roi  = frame_bgr[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    if USE_OTSU:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(gray, THRESHOLD_VAL, 255, cv2.THRESH_BINARY_INV)

    k    = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    if cv2.countNonZero(mask) < MIN_LINE_PX:
        return None, 0

    h  = mask.shape[0]
    ys = np.linspace(h - 1, 0, SCAN_ROWS, dtype=int)
    centers = []
    for y in ys:
        xs = np.where(mask[y] > 0)[0]
        if xs.size >= 4:
            centers.append(float(np.mean(xs)))

    if len(centers) < MIN_VALID_ROWS:
        return None, len(centers)

    return float(np.median(centers)), len(centers)


# =============================================================================
# PIVOT (90° turn)
# =============================================================================
def do_pivot(picam2, pid: PID, direction: float):
    """
    Swing-turn pivot: one motor drives, one stops.
    direction > 0  →  turn right  (left motor drives)
    direction < 0  →  turn left   (right motor drives)

    Swing turn chosen over in-place spin to avoid the reverse-motor
    current spike that sags battery voltage and stalls the forward motor.
    """
    pid.reset()

    roi_w     = ROI_RIGHT - ROI_LEFT
    target_cx = roi_w / 2.0

    if direction > 0:
        motor_set(PIVOT_SPEED, 0.0)   # turn right
    else:
        motor_set(0.0, PIVOT_SPEED)   # turn left

    t0 = time.time()
    while (time.time() - t0) < PIVOT_TIMEOUT:
        frame = cv2.cvtColor(picam2.capture_array(), cv2.COLOR_RGB2BGR)
        if ROTATE_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        cx, _ = get_line_center(frame)
        if cx is not None and abs(cx - target_cx) < PIVOT_EXIT_PX:
            break
        time.sleep(0.008)

    # Hard brake
    motor_stop()
    time.sleep(PIVOT_BRAKE_MS / 1000.0)
    pid.reset()


# =============================================================================
# MAIN
# =============================================================================
def main():
    motor_init()

    cam = Picamera2()
    cam.configure(cam.create_video_configuration(
        main={"size": (FRAME_W, FRAME_H), "format": "RGB888"}
    ))
    cam.start()
    time.sleep(1.0)

    roi_w     = ROI_RIGHT - ROI_LEFT
    target_cx = roi_w / 2.0

    pid           = PID()
    last_t        = time.time()
    last_seen     = last_t
    last_sign     = 1.0     # +1 = line was last seen to the right
    last_corr     = 0.0

    print("Running. Press Ctrl+C to stop.")

    try:
        while True:
            frame_rgb = cam.capture_array()
            frame     = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            if ROTATE_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            now  = time.time()
            dt   = max(now - last_t, 1e-3)
            last_t = now

            cx, n_rows = get_line_center(frame)

            # ── LINE VISIBLE ─────────────────────────────────────────────────
            if cx is not None:
                last_seen = now
                error     = cx - target_cx

                if abs(error) > 1.0:
                    last_sign = 1.0 if error > 0 else -1.0

                # Sharp turn
                if abs(error) >= PIVOT_THRESHOLD:
                    do_pivot(cam, pid, last_sign)
                    last_corr = 0.0
                    last_t    = time.time()
                    continue

                # Normal PID tracking
                corr     = pid.update(error, dt)
                # Slew limiter — no sudden correction jumps
                max_step = 200.0 * dt
                corr     = _clamp(corr, last_corr - max_step, last_corr + max_step)
                last_corr = corr

                # Slow down proportionally to correction magnitude
                turn_ratio = min(1.0, abs(corr) / max(1.0, MAX_CORR))
                fwd        = BASE_SPEED - (BASE_SPEED - MIN_SPEED) * turn_ratio
                # Ensure slower motor stays above stall
                fwd        = max(fwd, STALL_PWM + abs(corr) + 2.0)
                fwd        = min(fwd, BASE_SPEED)

                motor_set(fwd + corr, fwd - corr)

            # ── LINE LOST ────────────────────────────────────────────────────
            else:
                pid.reset()
                last_corr = 0.0
                if (now - last_seen) >= LOST_TIMEOUT_S:
                    # Spin toward last known line side
                    s = SEARCH_SPEED * last_sign
                    motor_set(s, -s)
                else:
                    motor_set(BASE_SPEED * 0.7, BASE_SPEED * 0.7)

            # ── PREVIEW ──────────────────────────────────────────────────────
            preview = frame.copy()
            cv2.rectangle(preview, (ROI_LEFT, ROI_TOP), (ROI_RIGHT, ROI_BOTTOM), (255, 200, 0), 1)
            mid_y = (ROI_TOP + ROI_BOTTOM) // 2
            cv2.line(preview, (int(target_cx + ROI_LEFT), ROI_TOP),
                               (int(target_cx + ROI_LEFT), ROI_BOTTOM), (0, 0, 255), 1)
            if cx is not None:
                cv2.circle(preview, (int(cx + ROI_LEFT), mid_y), 6, (0, 255, 0), -1)
            err_str = f"{int(cx - target_cx):+d}" if cx is not None else "lost"
            cv2.putText(preview, f"err={err_str} rows={n_rows}",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 255, 50), 1)
            cv2.imshow("Line Follower", preview)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        motor_stop()
        try: cam.stop()
        except: pass
        cv2.destroyAllWindows()
        GPIO.cleanup()
        print("Stopped.")


if __name__ == "__main__":
    main()