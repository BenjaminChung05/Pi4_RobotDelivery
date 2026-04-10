import RPi.GPIO as GPIO
import cv2
import numpy as np
import time
from picamera2 import Picamera2 # The modern camera library for Bookworm

# --- PIN CONFIGURATION ---
# Left Motor
ENA, IN1, IN2 = 12, 17, 18
# Right Motor
ENB, IN3, IN4 = 13, 22, 23

# --- GUI TOGGLE ---
# Set this to False if you run via SSH or get a "XDG_RUNTIME_DIR" error
# Set to True ONLY if you have a monitor plugged into the Pi
show_windows = False 

# --- GPIO SETUP ---
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup([ENA, IN1, IN2, ENB, IN3, IN4], GPIO.OUT)

pwm_a = GPIO.PWM(ENA, 100)
pwm_b = GPIO.PWM(ENB, 100)
pwm_a.start(0)
pwm_b.start(0)

def set_motors(left_speed, right_speed):
    """Controls the motors. Speeds from -100 to 100."""
    left_speed = max(min(left_speed, 100), -100)
    right_speed = max(min(right_speed, 100), -100)

    # --- REVERSED MOTOR LOGIC ---
    # We swapped HIGH and LOW here so positive numbers move forward!
    
    # Left Motor Direction
    if left_speed >= 0:
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH)
    else:
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.LOW)
        left_speed = -left_speed # Make positive for PWM

    # Right Motor Direction
    if right_speed >= 0:
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.HIGH)
    else:
        GPIO.output(IN3, GPIO.HIGH)
        GPIO.output(IN4, GPIO.LOW)
        right_speed = -right_speed # Make positive for PWM

    # Set Speeds
    pwm_a.ChangeDutyCycle(left_speed)
    pwm_b.ChangeDutyCycle(right_speed)

# --- CAMERA SETUP (Picamera2) ---
picam2 = Picamera2()
# Configure for small resolution to keep processing lightning fast
config = picam2.create_preview_configuration(main={"size": (320, 240)})
picam2.configure(config)
picam2.start()

# Tuning Variables
base_speed = 40
kp = 0.4        
last_cx = 160   

print("Camera warming up...")
time.sleep(2) # Give the sensor time to adjust exposure
print("Starting Line Follower... Press Ctrl+C to stop.")

try:
    while True:
        # Grab frame straight from Pi camera memory
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 1. FIX INVERTED CAMERA
        frame = cv2.flip(frame, -1)

        # 2. IMAGE PROCESSING
        h, w = frame.shape[:2]
        crop = frame[h//2:h, :] 
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        
        _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)

        # 3. FIND CONTOURS
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                last_cx = cx 
                
                error = cx - (w // 2)

                adjustment = error * kp
                left_speed = base_speed + adjustment
                right_speed = base_speed - adjustment

                # --- NEW: Print what the robot sees! ---
                print(f"Line seen at X: {cx:3d} | Error: {error:4d} | Motor Speeds -> L: {left_speed:4.0f}, R: {right_speed:4.0f}")

                set_motors(left_speed, right_speed)

        else:
            # 4. SHARP TURN RECOVERY
            if last_cx < w // 2:
                print("No line found! Spinning Left to recover...")
                # Increased from 45 to 75 to overcome friction
                set_motors(-75, 75) 
            else:
                print("No line found! Spinning Right to recover...")
                # Increased from 45 to 75 to overcome friction
                set_motors(75, -75)

        # We will leave the cv2.imshow out so it doesn't crash via sudo

except KeyboardInterrupt:
    print("\nProgram stopped by user.")

except KeyboardInterrupt:
    print("\nProgram stopped by user.")

finally:
    # Safely shut down everything
    set_motors(0, 0)
    pwm_a.stop()
    pwm_b.stop()
    GPIO.cleanup()
    picam2.stop()
    picam2.close()
    if show_windows:
        cv2.destroyAllWindows()
    print("Cleanup complete.")