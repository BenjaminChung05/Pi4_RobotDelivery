import time
import os
import cv2
import numpy as np
import RPi.GPIO as GPIO
from picamera2 import Picamera2

from symbol_detector import TemplateLibrary, SymbolDetector

# =============================================================================
# WEEK 3 INTEGRATED ROBOT
# Better priority logic:
#   1. Follow line every frame
#   2. If at intersection -> arrow-only detection
#   3. Else -> symbol-only detection
# This prevents arrows from being misclassified as QR / recycle / fingerprint.
# =============================================================================


# =============================================================================
# CONFIG
# =============================================================================

# Camera
FRAME_W, FRAME_H = 320, 240
ROTATE_180 = False

# ---------------- ROIs ----------------
# Bottom ROI for line following
LINE_ROI_TOP = 135
LINE_ROI_BOTTOM = 240
LINE_ROI_LEFT = 0
LINE_ROI_RIGHT = 320

# Upper ROI for arrows/symbols
DETECT_ROI_TOP = 15
DETECT_ROI_BOTTOM = 145
DETECT_ROI_LEFT = 0
DETECT_ROI_RIGHT = 320

# Smaller centre ROI for more reliable arrow checks
ARROW_ROI_TOP = 25
ARROW_ROI_BOTTOM = 135
ARROW_ROI_LEFT = 40
ARROW_ROI_RIGHT = 280

# ---------------- Line Following ----------------
MIN_LINE_AREA = 160
BASE_SPEED = 41.0
MIN_SPEED = 31.0
STALL_PWM = 30.0

Kp = 0.32
Ki = 0.002
Kd = 0.06
I_LIMIT = 80.0
MAX_CORR = 34.0
DEADBAND = 3.0
D_ALPHA = 0.75

# Sharp turn recovery
PIVOT_THRESHOLD = 45.0
PIVOT_SPEED = 58.0
PIVOT_EXIT_PX = 14.0
PIVOT_TIMEOUT = 0.65

SEARCH_SPEED = 40.0
LOST_TIMEOUT_S = 0.10

# ---------------- Intersection / Arrow ----------------
INTERSECTION_MIN_WIDTH_RATIO = 0.50
INTERSECTION_MIN_AREA = 1700

ARROW_CONFIRM_FRAMES = 2
ARROW_CHECK_EVERY = 2
ARROW_COOLDOWN_S = 1.2
ARROW_MAX_COUNT = 2

# ---------------- Symbol ----------------
SYMBOL_CONFIRM_FRAMES = 2
SYMBOL_CHECK_EVERY = 8
SYMBOL_COOLDOWN_S = 1.2

# ---------------- Action Durations ----------------
TURN_LEFT_90_TIME = 0.58
TURN_RIGHT_90_TIME = 0.58
TURN_BACK_TIME = 1.05
ROTATE_360_TIME = 2.15

# ---------------- GPIO ----------------
ENA, IN1, IN2 = 12, 17, 18   # left motor
ENB, IN3, IN4 = 13, 22, 23   # right motor
PWM_FREQ = 1000

LEFT_INVERT = False
RIGHT_INVERT = False


# =============================================================================
# MOTOR
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
    left = _clamp(float(left), -100.0, 100.0)
    right = _clamp(float(right), -100.0, 100.0)

    if LEFT_INVERT:
        left = -left
    if RIGHT_INVERT:
        right = -right

    if 0.0 < abs(left) < STALL_PWM:
        left = STALL_PWM if left > 0 else -STALL_PWM
    if 0.0 < abs(right) < STALL_PWM:
        right = STALL_PWM if right > 0 else -STALL_PWM

    GPIO.output(IN1, GPIO.LOW if left >= 0 else GPIO.HIGH)
    GPIO.output(IN2, GPIO.HIGH if left >= 0 else GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW if right >= 0 else GPIO.HIGH)
    GPIO.output(IN4, GPIO.HIGH if right >= 0 else GPIO.LOW)

    _pwm_l.ChangeDutyCycle(abs(left))
    _pwm_r.ChangeDutyCycle(abs(right))


def motor_stop():
    if _pwm_l:
        _pwm_l.ChangeDutyCycle(0)
    if _pwm_r:
        _pwm_r.ChangeDutyCycle(0)
    GPIO.output([IN1, IN2, IN3, IN4], GPIO.LOW)


def brake(ms=80):
    motor_stop()
    time.sleep(ms / 1000.0)


def forward(speed):
    motor_set(speed, speed)


def rotate_left(speed):
    motor_set(-speed, speed)


def rotate_right(speed):
    motor_set(speed, -speed)


# =============================================================================
# PID
# =============================================================================
class PID:
    def __init__(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.d_filt = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.d_filt = 0.0

    def update(self, error: float, dt: float) -> float:
        dt = max(dt, 1e-3)

        if abs(error) < DEADBAND:
            error = 0.0

        self.integral += error * dt
        self.integral = _clamp(self.integral, -I_LIMIT, I_LIMIT)

        d_raw = (error - self.prev_error) / dt
        self.d_filt = D_ALPHA * self.d_filt + (1.0 - D_ALPHA) * d_raw
        self.prev_error = error

        out = Kp * error + Ki * self.integral + Kd * self.d_filt
        return _clamp(out, -MAX_CORR, MAX_CORR)


# =============================================================================
# CAMERA / IMAGE
# =============================================================================
def get_frame(cam):
    frame_rgb = cam.capture_array()
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    if ROTATE_180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    return frame


def preprocess_hsv(roi):
    blur = cv2.GaussianBlur(roi, (5, 5), 0)
    return cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)


def largest_contour(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


def get_colour_masks(hsv):
    # Black
    black_lower = np.array([0, 0, 0])
    black_upper = np.array([180, 255, 70])

    # "Yellow appears light blue"
    lightblue_lower = np.array([80, 45, 70])
    lightblue_upper = np.array([108, 255, 255])

    # "Red appears ultramarine blue"
    ultrablue_lower = np.array([109, 70, 40])
    ultrablue_upper = np.array([135, 255, 255])

    mask_black = cv2.inRange(hsv, black_lower, black_upper)
    mask_lightblue = cv2.inRange(hsv, lightblue_lower, lightblue_upper)
    mask_ultrablue = cv2.inRange(hsv, ultrablue_lower, ultrablue_upper)

    kernel = np.ones((3, 3), np.uint8)

    def clean(mask):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask

    return {
        "BLACK": clean(mask_black),
        "LIGHTBLUE": clean(mask_lightblue),
        "ULTRAMARINE": clean(mask_ultrablue),
    }


def get_best_line(line_roi):
    hsv = preprocess_hsv(line_roi)
    masks = get_colour_masks(hsv)

    best = None
    best_area = 0

    for colour, mask in masks.items():
        cnt = largest_contour(mask)
        if cnt is None:
            continue

        area = cv2.contourArea(cnt)
        if area < MIN_LINE_AREA:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        x, y, w, h = cv2.boundingRect(cnt)

        if area > best_area:
            best_area = area
            best = {
                "colour": colour,
                "cx": cx,
                "area": area,
                "cnt": cnt,
                "bbox": (x, y, w, h),
                "mask": mask,
            }

    return best


def intersection_detected(line_info, roi_width):
    if line_info is None:
        return False

    area = line_info["area"]
    x, y, w, h = line_info["bbox"]

    return area > INTERSECTION_MIN_AREA and (w / float(roi_width)) > INTERSECTION_MIN_WIDTH_RATIO


# =============================================================================
# LABEL CONFIRMATION
# =============================================================================
def confirm_label(history, candidate, needed):
    history.append(candidate)
    if len(history) > needed:
        history.pop(0)

    if len(history) < needed:
        return None

    if all(v == candidate and v is not None for v in history):
        return candidate
    return None


# =============================================================================
# DETECTION LOGIC
# =============================================================================
def detect_arrow_only(detector, frame):
    candidate = detector.fast_filter(frame)
    result = None

    if candidate.found:
        result = detector.classify(frame, candidate)

    if (result is None) or (not result.accepted) or (result.label not in {"LEFT", "RIGHT"}):
        result = detector.probe_symbol(frame)

    if result.accepted and result.label in {"LEFT", "RIGHT"}:
        return result.label, result.bbox

    return None, None


def detect_symbol_only(detector, frame):
    candidate = detector.fast_filter(frame)
    result = None

    if candidate.found:
        result = detector.classify(frame, candidate)

    if (result is None) or (not result.accepted) or (result.label not in {"STOP", "RECYCLE", "QR", "FINGERPRINT"}):
        result = detector.probe_symbol(frame)

    if result.accepted and result.label in {"STOP", "RECYCLE", "QR", "FINGERPRINT"}:
        return result.label, result.bbox

    return None, None

# =============================================================================
# ACTIONS
# =============================================================================
def do_turn(direction):
    brake(80)

    if direction == "LEFT":
        rotate_left(PIVOT_SPEED)
        time.sleep(TURN_LEFT_90_TIME)

    elif direction == "RIGHT":
        rotate_right(PIVOT_SPEED)
        time.sleep(TURN_RIGHT_90_TIME)

    elif direction == "DOWN":
        rotate_right(PIVOT_SPEED)
        time.sleep(TURN_BACK_TIME)

    elif direction == "UP":
        forward(BASE_SPEED)
        time.sleep(0.18)

    brake(100)


def do_symbol_action(symbol):
    if symbol == "STOP":
        print("[ACTION] STOP detected -> stop temporarily")
        brake(1500)

    elif symbol == "RECYCLE":
        print("[ACTION] RECYCLE detected -> rotate 360")
        rotate_right(PIVOT_SPEED)
        time.sleep(ROTATE_360_TIME)
        brake(120)

    elif symbol == "FINGERPRINT":
        print("[ACTION] Detected biometric: FINGERPRINT")
        brake(500)

    elif symbol == "QR":
        print("[ACTION] Detected biometric: QR")
        brake(500)


def do_search_spin(last_sign):
    s = SEARCH_SPEED * last_sign
    motor_set(s, -s)


def do_pivot_recover(cam, pid, direction_sign):
    pid.reset()

    roi_w = LINE_ROI_RIGHT - LINE_ROI_LEFT
    target_cx = roi_w / 2.0

    if direction_sign > 0:
        motor_set(PIVOT_SPEED, 0.0)
    else:
        motor_set(0.0, PIVOT_SPEED)

    t0 = time.time()
    while (time.time() - t0) < PIVOT_TIMEOUT:
        frame = get_frame(cam)
        line_roi = frame[LINE_ROI_TOP:LINE_ROI_BOTTOM, LINE_ROI_LEFT:LINE_ROI_RIGHT]
        info = get_best_line(line_roi)

        if info is not None:
            cx = info["cx"]
            if abs(cx - target_cx) < PIVOT_EXIT_PX:
                break

        time.sleep(0.01)

    brake(60)
    pid.reset()


# =============================================================================
# MAIN
# =============================================================================
def main():
    if not os.path.isdir("templates"):
        raise RuntimeError("Put your template images inside a folder named 'templates'.")

    motor_init()

    cam = Picamera2()
    cam.configure(cam.create_video_configuration(
        main={"size": (FRAME_W, FRAME_H), "format": "RGB888"}
    ))
    cam.start()
    time.sleep(1.0)

    lib = TemplateLibrary("templates")
    lib.load()
    detector = SymbolDetector(lib)

    pid = PID()

    frame_count = 0
    last_t = time.time()
    last_seen = last_t
    last_sign = 1.0
    last_corr = 0.0

    arrow_history = []
    symbol_history = []

    arrow_done_count = 0
    arrow_cooldown_until = 0.0
    symbol_cooldown_until = 0.0

    roi_w = LINE_ROI_RIGHT - LINE_ROI_LEFT
    target_cx = roi_w / 2.0

    print("Running integrated Week 3 code. Press Ctrl+C to stop.")

    try:
        while True:
            frame = get_frame(cam)
            now = time.time()
            dt = max(now - last_t, 1e-3)
            last_t = now

            display = frame.copy()

            # -------------------------------------------------
            # 1. LINE FOLLOWING
            # -------------------------------------------------
            line_roi = frame[LINE_ROI_TOP:LINE_ROI_BOTTOM, LINE_ROI_LEFT:LINE_ROI_RIGHT]
            line_info = get_best_line(line_roi)

            cx = None
            at_intersection = False
            error = None

            cv2.rectangle(display, (LINE_ROI_LEFT, LINE_ROI_TOP), (LINE_ROI_RIGHT, LINE_ROI_BOTTOM), (255, 200, 0), 1)
            cv2.rectangle(display, (ARROW_ROI_LEFT, ARROW_ROI_TOP), (ARROW_ROI_RIGHT, ARROW_ROI_BOTTOM), (255, 0, 255), 1)

            if line_info is not None:
                cx = line_info["cx"]
                error = cx - target_cx
                last_seen = now

                if abs(error) > 1.0:
                    last_sign = 1.0 if error > 0 else -1.0

                at_intersection = intersection_detected(line_info, roi_w)

                cv2.circle(display, (int(cx + LINE_ROI_LEFT), (LINE_ROI_TOP + LINE_ROI_BOTTOM) // 2), 5, (0, 255, 0), -1)
                cv2.line(display,
                         (int(target_cx + LINE_ROI_LEFT), LINE_ROI_TOP),
                         (int(target_cx + LINE_ROI_LEFT), LINE_ROI_BOTTOM),
                         (0, 0, 255), 1)

                cv2.putText(display, f"LINE: {line_info['colour']}", (8, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.putText(display, f"ERR: {int(error):+d}", (8, 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            else:
                cv2.putText(display, "LINE: LOST", (8, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

            # -------------------------------------------------
            # 2. ARROW MODE (ONLY AT INTERSECTION)
            # -------------------------------------------------
            did_arrow_action = False

            if (
                line_info is not None
                and at_intersection
                and arrow_done_count < ARROW_MAX_COUNT
                and now > arrow_cooldown_until
                and frame_count % ARROW_CHECK_EVERY == 0
            ):
                cv2.putText(display, "MODE: ARROW", (8, 64),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

                arrow_label, arrow_bbox = detect_arrow_only(detector, frame)
                confirmed_arrow = confirm_label(arrow_history, arrow_label, ARROW_CONFIRM_FRAMES) if arrow_label else None

                if arrow_label and arrow_bbox is not None:
                    x, y, w, h = arrow_bbox
                    cv2.rectangle(display, (x, y), (x + w, y + h), (255, 255, 0), 2)
                    cv2.putText(display, f"ARROW: {arrow_label}", (x, max(18, y - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

                if confirmed_arrow:
                    print(f"[ARROW {arrow_done_count + 1}] confirmed -> {confirmed_arrow}")
                    arrow_done_count += 1
                    do_turn(confirmed_arrow)
                    arrow_history.clear()
                    arrow_cooldown_until = time.time() + ARROW_COOLDOWN_S
                    did_arrow_action = True

            if did_arrow_action:
                frame_count += 1
                continue

            # -------------------------------------------------
            # 3. SYMBOL MODE (ONLY WHEN NOT AT INTERSECTION)
            # -------------------------------------------------
            did_symbol_action = False

            if (
                not at_intersection
                and now > symbol_cooldown_until
                and frame_count % SYMBOL_CHECK_EVERY == 0
            ):
                cv2.putText(display, "MODE: SYMBOL", (8, 64),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

                symbol_label, symbol_bbox = detect_symbol_only(detector, frame)
                confirmed_symbol = confirm_label(symbol_history, symbol_label, SYMBOL_CONFIRM_FRAMES) if symbol_label else None

                if symbol_label and symbol_bbox is not None:
                    x, y, w, h = symbol_bbox
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    cv2.putText(display, f"SYMBOL: {symbol_label}", (x, max(18, y - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

                if confirmed_symbol:
                    print(f"[SYMBOL] confirmed -> {confirmed_symbol}")
                    do_symbol_action(confirmed_symbol)
                    symbol_history.clear()
                    symbol_cooldown_until = time.time() + SYMBOL_COOLDOWN_S
                    did_symbol_action = True

            if did_symbol_action:
                frame_count += 1
                continue

            # -------------------------------------------------
            # 4. NORMAL MOVEMENT
            # -------------------------------------------------
            if line_info is not None:
                if at_intersection:
                    cv2.putText(display, "INTERSECTION", (8, 88),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

                if abs(error) >= PIVOT_THRESHOLD:
                    do_pivot_recover(cam, pid, last_sign)
                    last_corr = 0.0
                    last_t = time.time()
                    frame_count += 1
                    continue

                corr = pid.update(error, dt)

                max_step = 180.0 * dt
                corr = _clamp(corr, last_corr - max_step, last_corr + max_step)
                last_corr = corr

                turn_ratio = min(1.0, abs(corr) / max(1.0, MAX_CORR))
                fwd = BASE_SPEED - (BASE_SPEED - MIN_SPEED) * turn_ratio
                fwd = max(fwd, STALL_PWM + abs(corr) + 2.0)
                fwd = min(fwd, BASE_SPEED)

                motor_set(fwd + corr, fwd - corr)

            else:
                pid.reset()
                last_corr = 0.0

                if (now - last_seen) >= LOST_TIMEOUT_S:
                    do_search_spin(last_sign)
                else:
                    forward(BASE_SPEED * 0.7)

            cv2.putText(display, f"Arrows done: {arrow_done_count}/2", (8, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Week3 Integrated", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

    except KeyboardInterrupt:
        pass

    finally:
        motor_stop()
        try:
            cam.stop()
        except:
            pass
        cv2.destroyAllWindows()
        GPIO.cleanup()
        print("Stopped.")


if __name__ == "__main__":
    main()