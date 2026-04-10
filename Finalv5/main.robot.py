import threading
import time
import cv2
import numpy as np
import queue
import RPi.GPIO as GPIO
from picamera2 import Picamera2

# =============================================================================
# IMPORT SYMBOL DETECTOR (Ensure run_symbol.py is in the same folder)
# =============================================================================
from run_symbol import TemplateLibra, SymbolDetector, StableLabel, TEMPLATES_DIR

# =============================================================================
# 1) CONFIGURATION
# =============================================================================
FRAME_W, FRAME_H = 320, 240
ROI_TOP, ROI_BOTTOM = 100, 240
ROI_LEFT, ROI_RIGHT = 0, 320

ROTATE_180 = True
USE_OTSU = True
THRESHOLD_VAL = 60  

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

# PID tuned for smoothness
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

# =============================================================================
# 2) MOTOR PINS & HARDWARE CONTROL
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
    global pwm_left, pwm_right
    GPIO.cleanup()
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for pin in (ENA, IN1, IN2, ENB, IN3, IN4):
        GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
    pwm_left = GPIO.PWM(ENA, 1000)
    pwm_right = GPIO.PWM(ENB, 1000)
    pwm_left.start(0)
    pwm_right.start(0)

def set_motor(left_speed: float, right_speed: float) -> None:
    global pwm_left, pwm_right

    left_speed = clamp(float(left_speed), -MAX_SPEED, MAX_SPEED)
    right_speed = clamp(float(right_speed), -MAX_SPEED, MAX_SPEED)

    if LEFT_MOTOR_INVERT: left_speed = -left_speed
    if RIGHT_MOTOR_INVERT: right_speed = -right_speed

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

    if pwm_left is not None and pwm_right is not None:
        pwm_left.ChangeDutyCycle(abs(left_speed))
        pwm_right.ChangeDutyCycle(abs(right_speed))

def stop_motors() -> None:
    if pwm_left is not None: pwm_left.ChangeDutyCycle(0)
    if pwm_right is not None: pwm_right.ChangeDutyCycle(0)
    GPIO.output([IN1, IN2, IN3, IN4], GPIO.LOW)

# =============================================================================
# 3) PID CONTROLLER
# =============================================================================
class PIDController:
    def _init_(self, kp: float, ki: float, kd: float):
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
# 4) VISION: ALL-COLOR MASK & CONTOUR CENTROID
# =============================================================================
def build_line_mask(roi_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    if USE_OTSU:
        _, mask_black = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, mask_black = cv2.threshold(gray, THRESHOLD_VAL, 255, cv2.THRESH_BINARY_INV)

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

    mask_yellow = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([30, 255, 255]))
    mask_red1 = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    combined_mask = cv2.bitwise_or(mask_black, mask_yellow)
    combined_mask = cv2.bitwise_or(combined_mask, mask_red)

    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return combined_mask

def detect_centerline(mask: np.ndarray, last_known_cx: float) -> tuple[float | None, int]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None, 0

    valid_contours = [c for c in contours if cv2.contourArea(c) > MIN_LINE_PIXELS]
    if not valid_contours: return None, 0

    best_cx = None
    min_dist = float('inf')
    
    for c in valid_contours:
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = float(M["m10"] / M["m00"])
            dist = abs(cx - last_known_cx)
            if dist < min_dist:
                min_dist = dist
                best_cx = cx

    return best_cx, len(valid_contours)

# =============================================================================
# 5) THREADING ARCHITECTURE & SHARED STATE
# =============================================================================
class SharedState:
    def _init_(self):
        self.lock = threading.Lock()
        self.steering = (0.0, 0.0)  
        self.symbol = None
        self.running = True
        self.frame_queue = queue.Queue(maxsize=2)

    def set_steering(self, left, right):
        with self.lock:
            self.steering = (left, right)

    def get_steering(self):
        with self.lock:
            return self.steering

state = SharedState()

def thread_camera_reader(picam2):
    while state.running:
        frame_rgb = picam2.capture_array()
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if ROTATE_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            
        if not state.frame_queue.full():
            state.frame_queue.put(frame)
        else:
            try:
                state.frame_queue.get_nowait()
                state.frame_queue.put(frame)
            except queue.Empty:
                pass
        time.sleep(0.01)

def thread_line_follow(target_cx, pid):
    last_time = time.time()
    last_seen = last_time
    last_known_cx = target_cx 
    last_error_sign = 1.0
    last_correction = 0.0
    
    while state.running:
        if state.frame_queue.empty():
            time.sleep(0.01)
            continue
            
        frame = state.frame_queue.queue[-1].copy() 
        roi = frame[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT]
        mask = build_line_mask(roi)
        
        now = time.time()
        dt = max(now - last_time, 1e-3)
        last_time = now

        cx, num_lines = detect_centerline(mask, last_known_cx)

        if cx is not None:
            last_known_cx = cx 
            last_seen = now
            error = cx - target_cx

            if abs(error) < ERROR_DEADBAND_PX:
                error = 0.0
            if abs(error) > 1.0:
                last_error_sign = 1.0 if error > 0 else -1.0

            if abs(error) >= SHARP_TURN_ERROR_PX:
                pid.reset()
                last_correction = 0.0
                if last_error_sign < 0:
                    state.set_steering(SHARP_PIVOT_SPEED, -SHARP_PIVOT_SPEED)
                else:
                    state.set_steering(-SHARP_PIVOT_SPEED, SHARP_PIVOT_SPEED)
                time.sleep(0.01)
                continue

            correction = pid.update(error, dt)
            correction = clamp(correction, -MAX_CORRECTION, MAX_CORRECTION)

            max_step = MAX_CORRECTION_SLEW_PER_SEC * dt
            correction = clamp(correction, last_correction - max_step, last_correction + max_step)
            last_correction = correction

            turn_ratio = min(1.0, abs(correction) / max(1.0, MAX_CORRECTION))
            forward = BASE_SPEED - (BASE_SPEED - MIN_SPEED) * turn_ratio

            left_speed = forward + correction
            right_speed = forward - correction
            state.set_steering(left_speed, right_speed)
            
        else:
            pid.reset()
            last_correction = 0.0
            if (now - last_seen) >= LOST_TIMEOUT_S:
                spin = SEARCH_SPIN_SPEED * last_error_sign
                state.set_steering(spin, -spin)
            else:
                state.set_steering(LOST_FORWARD_SPEED, LOST_FORWARD_SPEED)

def thread_symbol_detect(detector, stable):
    last_print_time = 0
    
    while state.running:
        if state.frame_queue.empty():
            time.sleep(0.05)
            continue
            
        frame = state.frame_queue.queue[-1].copy()
        
        results, _ = detector.detect(frame)
        raw_label = results[0]["label"] if results else None
        stable_label = stable.update(raw_label)
        
        with state.lock:
            state.symbol = stable_label

        if SHOW_PREVIEW_WINDOW:
            display_frame = frame.copy()
            if stable_label:
                # Displays the symbol text on the video feed window
                cv2.putText(display_frame, f"DETECTED: {stable_label}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Prints to the console window specifically for Fingerprint or QR
                now = time.time()
                if stable_label in ['QR_Code', 'Fingerprint'] and (now - last_print_time > 1.0):
                    print(f"\n>>> BIOMETRIC DISPLAY: {stable_label.upper()} DETECTED <<<")
                    last_print_time = now
                    
            cv2.imshow("Robot Vision System", display_frame)
            cv2.waitKey(1)
            
        time.sleep(0.15)

def thread_motor_ctrl():
    init_motor_gpio()
    
    while state.running:
        with state.lock:
            current_sym = state.symbol
        
        if current_sym in ['Hazard', 'Button']:
            print(f"[{current_sym}] detected! Pausing...")
            set_motor(0, 0)
            time.sleep(1.5) 
            with state.lock: state.symbol = None 
            
        elif current_sym == 'Recycle':
            print("[Recycle] detected! Spinning 360 degrees...")
            set_motor(50, -50) 
            time.sleep(1.2) 
            with state.lock: state.symbol = None
            
        elif current_sym in ['QR_Code', 'Fingerprint']:
            # Display logic is handled in Thread 2, we just clear the motor flag
            with state.lock: state.symbol = None
            
        elif current_sym == 'Arrow_Left':
            print("[Arrow_Left] Forcing sharp left onto new track!")
            set_motor(-40, 50)
            time.sleep(0.5) 
            with state.lock: state.symbol = None
            
        elif current_sym == 'Arrow_Right':
            print("[Arrow_Right] Forcing sharp right onto new track!")
            set_motor(50, -40)
            time.sleep(0.5) 
            with state.lock: state.symbol = None
            
        else:
            l, r = state.get_steering()
            set_motor(l, r)
            
        time.sleep(0.05) 

# =============================================================================
# 6) MASTER SYSTEM BOOT SEQUENCE
# =============================================================================
def main():
    print("Loading Templates and Initializing Cameras...")
    lib = TemplateLibrary(TEMPLATES_DIR).load()
    detector = SymbolDetector(lib, min_area=1800, match_thresh=0.42)
    stable = StableLabel(window=12, require=8)
    pid = PIDController(Kp, Ki, Kd)
    
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(main={"size": (FRAME_W, FRAME_H), "format": "RGB888"}))
    picam2.start()
    
    target_cx = (ROI_RIGHT - ROI_LEFT) / 2.0 

    threads = [
        threading.Thread(target=thread_camera_reader, args=(picam2,)), 
        threading.Thread(target=thread_line_follow, args=(target_cx, pid)),
        threading.Thread(target=thread_symbol_detect, args=(detector, stable)),
        threading.Thread(target=thread_motor_ctrl)
    ]

    print("System Ready. Waiting for track deployment. Press Ctrl+C to stop.")
    try:
        for t in threads: t.start()
        for t in threads: t.join()
    except KeyboardInterrupt:
        print("\nShutting down motors and closing threads...")
    finally:
        state.running = False
        stop_motors()
        if SHOW_PREVIEW_WINDOW:
            cv2.destroyAllWindows()
        GPIO.cleanup()
        print("Robot gracefully powered down.")

if _name_ == "_main_":
    main()
