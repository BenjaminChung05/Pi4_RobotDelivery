import RPi.GPIO as GPIO
import cv2
import numpy as np
import time
import os
from collections import Counter, deque
from picamera2 import Picamera2
from symbol_detector import TemplateLibrary, SymbolDetector

# --- PIN CONFIGURATION ---
ENA, IN1, IN2 = 12, 17, 18
ENB, IN3, IN4 = 13, 22, 23
show_windows = True 

# --- GPIO SETUP ---
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup([ENA, IN1, IN2, ENB, IN3, IN4], GPIO.OUT)

pwm_a = GPIO.PWM(ENA, 100)
pwm_b = GPIO.PWM(ENB, 100)
pwm_a.start(0)
pwm_b.start(0)

def set_motors(left_speed, right_speed):
    """Controls the motors with reversed logic."""
    left_speed = max(min(left_speed, 100), -100)
    right_speed = max(min(right_speed, 100), -100)

    if left_speed >= 0:
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH)
    else:
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.LOW)
        left_speed = -left_speed 

    if right_speed >= 0:
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.HIGH)
    else:
        GPIO.output(IN3, GPIO.HIGH)
        GPIO.output(IN4, GPIO.LOW)
        right_speed = -right_speed 

    pwm_a.ChangeDutyCycle(left_speed)
    pwm_b.ChangeDutyCycle(right_speed)

# --- SYMBOL DETECTOR SETUP ---
TEMPLATES_DIR = "templates"
DETECT_WINDOW = 8
DETECT_REQUIRE = 5


class StableLabel:
    def __init__(self, window=12, require=8):
        self.buf = deque(maxlen=window)
        self.window = window
        self.require = require

    def update(self, label):
        self.buf.append(label)
        if len(self.buf) < self.window:
            return None
        counts = Counter([x for x in self.buf if x is not None])
        if not counts:
            return None
        best, n = counts.most_common(1)[0]
        return best if n >= self.require else None


if not os.path.exists(TEMPLATES_DIR):
    os.makedirs(TEMPLATES_DIR)
    print(f"WARNING: '{TEMPLATES_DIR}' folder created. Please add your template images!")

lib = TemplateLibrary(TEMPLATES_DIR, swap_octagon_hazard=True).load()
detector = SymbolDetector(
    lib,
    min_area=1800,
    match_thresh=0.42,
    cross_shape_thresh=0.85,
    fp_min_good=10,
    fp_ratio=0.75,
)
stable = StableLabel(window=DETECT_WINDOW, require=DETECT_REQUIRE)

# --- CAMERA SETUP ---
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)}) 
picam2.configure(config)
picam2.start()

# Tuning Variables
base_speed = 38
base_speed_slow = 28
kp = 0.22
kd = 0.10
max_adjust = 30
deadband = 8
err_smooth = 0.65
last_cx = 320
prev_error = 0
smooth_error = 0
robot_state = "FOLLOWING" 

detected_shape_text = ""
shape_box_to_draw = None

print("Camera warming up...")
time.sleep(2)
print("Starting Line Follower... Press Ctrl+C to stop.")

try:
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.flip(frame, -1) 
        
        h_frame, w_frame = frame.shape[:2]

        if robot_state == "FOLLOWING":
            # --- 1. SYMBOL DETECTION (Stable over multiple frames) ---
            roi = (
                int(0.12 * w_frame),
                int(0.05 * h_frame),
                int(0.88 * w_frame),
                int(0.80 * h_frame),
            )
            results, _ = detector.detect(frame, roi=roi)

            raw_label = results[0]["label"] if results else None
            stable_label = stable.update(raw_label)

            if results:
                shape_box_to_draw = results[0]["bbox"]

            if stable_label is not None:
                detected_shape_text = stable_label
                robot_state = "STOPPED"
                set_motors(0, 0)  # Halt immediately
                print(f"*** STOPPED! Detected: {detected_shape_text} ***")

            # --- 2. LINE FOLLOWING (If no stable shape was found) ---
            if robot_state == "FOLLOWING":
                speed = base_speed_slow if results else base_speed
                crop = frame[h_frame//2:h_frame, :] 
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                blur_crop = cv2.GaussianBlur(gray_crop, (5,5), 0)
                
                # Keep this at 60 so it cleanly sees the black line
                _, thresh_crop = cv2.threshold(blur_crop, 60, 255, cv2.THRESH_BINARY_INV)

                contours_crop, _ = cv2.findContours(thresh_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                if len(contours_crop) > 0:
                    c = max(contours_crop, key=cv2.contourArea)
                    M = cv2.moments(c)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        last_cx = cx 
                        error = cx - (w_frame // 2)
                        if abs(error) < deadband:
                            error = 0

                        smooth_error = (err_smooth * smooth_error) + ((1.0 - err_smooth) * error)
                        d_error = smooth_error - prev_error
                        prev_error = smooth_error

                        adjustment = (kp * smooth_error) + (kd * d_error)
                        if adjustment > max_adjust:
                            adjustment = max_adjust
                        elif adjustment < -max_adjust:
                            adjustment = -max_adjust
                        set_motors(speed + adjustment, speed - adjustment)
                else:
                    # Sharp Turn Recovery
                    if last_cx < w_frame // 2:
                        set_motors(-75, 75) 
                    else:
                        set_motors(75, -75) 

        # --- 3. DISPLAY LOGIC ---
        if robot_state == "STOPPED" and shape_box_to_draw is not None:
            x, y, w_box, h_box = shape_box_to_draw
            cv2.rectangle(frame, (x, y), (x+w_box, y+h_box), (0, 0, 255), 4)
            cv2.putText(frame, detected_shape_text, (x, max(y - 10, 20)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        if show_windows:
            cv2.imshow("Robot View", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

except KeyboardInterrupt:
    print("\nProgram stopped by user.")

finally:
    set_motors(0, 0)
    pwm_a.stop()
    pwm_b.stop()
    GPIO.cleanup()
    picam2.stop()
    picam2.close()
    if show_windows:
        cv2.destroyAllWindows()
    print("Cleanup complete.")
