import RPi.GPIO as GPIO
import time

# =============================================================================
# SECTION 1 — GPIO MODE
# Purpose: Configure GPIO library to use BCM numbering.
# =============================================================================
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# =============================================================================
# SECTION 2 — PIN DEFINITIONS (BCM)
# Purpose: Define Raspberry Pi pins connected to the L298N motor driver.
# =============================================================================
# Motor A (Left Side - 2 motors wired together)
ENA = 12
IN1 = 17
IN2 = 18

# Motor B (Right Side - 2 motors wired together)
ENB = 13
IN3 = 22
IN4 = 23

PWM_FREQ = 1000  # Hz

# =============================================================================
# SECTION 3 — DEFAULT SPEEDS + TRIM
# Purpose: Default PWM duty cycles (0–100) and left/right trim correction.
# =============================================================================
NORMAL_SPEED = 60   # Forward/Reverse PWM
TURN_SPEED = 75     # Turning PWM
TURN_DURATION_SCALE = 0.913 #<1 prevents overshoot, >1 introduces overshoot

LEFT_TRIM = 1.0
RIGHT_TRIM = 1.0

# Default durations (seconds) if time is omitted for w/s
FWD_TIME = 1.0
REV_TIME = 1.0

# =============================================================================
# SECTION 4 — MINIMUM SAFETY LIMITS
# =============================================================================
MIN_TURN_SEC = 0.05
MIN_MOVE_SEC = 0.05

# =============================================================================
# SECTION 5 — TURN RATE CALIBRATION TABLES (PWM -> deg/s)
# Purpose: Turning rate depends on PWM and can be different left vs right.
# =============================================================================
RIGHT_TURN_RATE_TABLE = {
    60: 83,    # updated (was ~84.75); fixes 180->~200 overshoot at PWM60
    75: 120.53,   # updated (was ~105.88); fixes 180->~210 overshoot at PWM75
}

LEFT_TURN_RATE_TABLE = {
    60: 78.2, # ≈ 84.75 deg/s
    75: 110.2,   # updated (was ~105.88); fixes 180->225 overshoot at PWM75
}

# =============================================================================
# SECTION 6 — LINEAR SPEED CALIBRATION TABLES (PWM -> cm/s)
# Purpose: Forward/Reverse speed depends on PWM (open-loop estimate).
#
# Your observation:
# - Tile is 60 cm
# - f 60 took ~1.5 s at PWM 60, but f 120 overshot to ~2.25 tiles
#   => adjust to ~45 cm/s at PWM 60 for better long-run accuracy.
# =============================================================================
FORWARD_SPEED_TABLE = {
    60: 41.38,   # tuned from overshoot observation (approx)
}

REVERSE_SPEED_TABLE = {
    60: 42.76,   # optional; measure separately if reverse differs
}

# =============================================================================
# SECTION 7 — OPTIONAL RAMPING (smooth manual changes only)
# Note: Timed/angle/distance actions run with ramp=False for accuracy.
# =============================================================================
RAMP_STEP = 5
RAMP_DELAY = 0.03
CURRENT_LEFT = 0.0
CURRENT_RIGHT = 0.0

# =============================================================================
# SECTION 8 — DISTANCE TRACKING (TOTAL PATH LENGTH)
# Purpose: Track total linear distance travelled (sum of forward + reverse).
# Turns do not add to this total.
# =============================================================================
TOTAL_DISTANCE_CM = 0.0

# =============================================================================
# SECTION 9 — INITIALIZE GPIO + PWM
# =============================================================================
motor_pins = [ENA, IN1, IN2, ENB, IN3, IN4]
for pin in motor_pins:
    GPIO.setup(pin, GPIO.OUT)

pwm_a = GPIO.PWM(ENA, PWM_FREQ)
pwm_b = GPIO.PWM(ENB, PWM_FREQ)
pwm_a.start(0)
pwm_b.start(0)

# =============================================================================
# SECTION 10 — LOW-LEVEL HELPERS
# =============================================================================
def clamp(val, lo, hi):
    return max(lo, min(hi, val))

def apply_pwm(left_dc, right_dc):
    """Apply PWM duty cycles to both channels (0–100)."""
    pwm_a.ChangeDutyCycle(left_dc)
    pwm_b.ChangeDutyCycle(right_dc)

def set_motor_speed(speed_left, speed_right, ramp=False):
    """
    Set PWM duty cycle for left/right motors (0–100) applying trim.
    ramp=True performs a gradual change to the target duty cycle.
    """
    global CURRENT_LEFT, CURRENT_RIGHT

    target_left = clamp(speed_left * LEFT_TRIM, 0, 100)
    target_right = clamp(speed_right * RIGHT_TRIM, 0, 100)

    if not ramp:
        apply_pwm(target_left, target_right)
        CURRENT_LEFT, CURRENT_RIGHT = target_left, target_right
        return

    steps = int(max(abs(target_left - CURRENT_LEFT), abs(target_right - CURRENT_RIGHT)) // RAMP_STEP) + 1
    start_left, start_right = CURRENT_LEFT, CURRENT_RIGHT

    for i in range(1, steps + 1):
        new_left = start_left + (target_left - start_left) * (i / steps)
        new_right = start_right + (target_right - start_right) * (i / steps)
        apply_pwm(new_left, new_right)
        time.sleep(RAMP_DELAY)

    CURRENT_LEFT, CURRENT_RIGHT = target_left, target_right

def stop(coast=True, ramp=False):
    """
    Stop the robot.
    coast=True  -> IN pins LOW/LOW  (free roll)
    coast=False -> IN pins HIGH/HIGH (brake)
    ramp controls whether PWM ramps down smoothly.
    """
    if coast:
        print("State: STOP (COAST)")
        GPIO.output(IN1, GPIO.LOW);  GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.LOW);  GPIO.output(IN4, GPIO.LOW)
    else:
        print("State: STOP (BRAKE)")
        GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN2, GPIO.HIGH)
        GPIO.output(IN3, GPIO.HIGH); GPIO.output(IN4, GPIO.HIGH)

    set_motor_speed(0, 0, ramp=ramp)

# =============================================================================
# SECTION 11 — MOVEMENT PRIMITIVES (NO SLEEP INSIDE)
# =============================================================================
def move_forward(speed=NORMAL_SPEED, ramp=False):
    print(f"State: FORWARD (Speed: {speed})")
    GPIO.output(IN1, GPIO.LOW);  GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW);  GPIO.output(IN4, GPIO.HIGH)
    set_motor_speed(speed, speed, ramp=ramp)

def move_backward(speed=NORMAL_SPEED, ramp=False):
    print(f"State: REVERSE (Speed: {speed})")
    GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH); GPIO.output(IN4, GPIO.LOW)
    set_motor_speed(speed, speed, ramp=ramp)

def turn_left(speed=TURN_SPEED, ramp=False):
    """Pivot left: left reverse, right forward"""
    print(f"State: TURN LEFT (Speed: {speed})")
    GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW);  GPIO.output(IN4, GPIO.HIGH)
    set_motor_speed(speed, speed, ramp=ramp)

def turn_right(speed=TURN_SPEED, ramp=False):
    """Pivot right: left forward, right reverse"""
    print(f"State: TURN RIGHT (Speed: {speed})")
    GPIO.output(IN1, GPIO.LOW);  GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH); GPIO.output(IN4, GPIO.LOW)
    set_motor_speed(speed, speed, ramp=ramp)

# =============================================================================
# SECTION 12 — TIMED EXECUTION (ACCURATE TIMING)
# =============================================================================
def run_for_duration(action_func, duration, speed=None, brake_after=False):
    """
    Run a movement action for duration (seconds), then stop.
    Uses ramp=False to avoid timing distortion.
    """
    if duration is None or duration <= 0:
        print("Duration must be > 0")
        return

    # Start immediately (no ramp)
    if speed is not None:
        action_func(speed, ramp=False)
    else:
        action_func(ramp=False)

    t0 = time.monotonic()
    time.sleep(duration)
    t1 = time.monotonic()

    stop(coast=not brake_after, ramp=False)
    print(f"Commanded: {duration:.2f}s | Actual: {t1 - t0:.2f}s")

# =============================================================================
# SECTION 13 — LOOKUP + INTERPOLATION UTILITIES
# =============================================================================
def lookup_with_interpolation(table, pwm):
    """
    Generic lookup for a PWM->value table with:
    - exact match if present
    - linear interpolation between nearest points
    - proportional scaling outside range (fallback)
    """
    pwm = int(clamp(pwm, 1, 100))
    keys = sorted(table.keys())
    if not keys:
        raise ValueError("Calibration table is empty.")

    if pwm in table:
        return table[pwm]

    if pwm < keys[0]:
        base_pwm = keys[0]
        return table[base_pwm] * (pwm / base_pwm)

    if pwm > keys[-1]:
        base_pwm = keys[-1]
        return table[base_pwm] * (pwm / base_pwm)

    lower = max(k for k in keys if k < pwm)
    upper = min(k for k in keys if k > pwm)

    v0 = table[lower]
    v1 = table[upper]
    return v0 + (v1 - v0) * ((pwm - lower) / (upper - lower))

# =============================================================================
# SECTION 14 — ANGLE CONTROL (PWM->deg/s table)
# =============================================================================
def angle_to_duration(angle_deg, pwm, turn_rate_table):
    angle_deg = abs(float(angle_deg))
    rate = lookup_with_interpolation(turn_rate_table, pwm)  # deg/s
    duration = (angle_deg / rate) * TURN_DURATION_SCALE
    return max(MIN_TURN_SEC, duration), rate

def turn_left_angle(angle_deg, speed=TURN_SPEED, brake_after=False):
    duration, rate = angle_to_duration(angle_deg, speed, LEFT_TURN_RATE_TABLE)
    print(f"TURN LEFT: {angle_deg}° @PWM {speed} (rate≈{rate:.1f} deg/s) -> {duration:.2f}s")
    run_for_duration(turn_left, duration, speed, brake_after=brake_after)

def turn_right_angle(angle_deg, speed=TURN_SPEED, brake_after=False):
    duration, rate = angle_to_duration(angle_deg, speed, RIGHT_TURN_RATE_TABLE)
    print(f"TURN RIGHT: {angle_deg}° @PWM {speed} (rate≈{rate:.1f} deg/s) -> {duration:.2f}s")
    run_for_duration(turn_right, duration, speed, brake_after=brake_after)

# =============================================================================
# SECTION 15 — DISTANCE CONTROL (PWM->cm/s table)
# =============================================================================
def distance_to_duration_cm(distance_cm, pwm, speed_table):
    distance_cm = abs(float(distance_cm))
    v = lookup_with_interpolation(speed_table, pwm)  # cm/s
    duration = distance_cm / v
    return max(MIN_MOVE_SEC, duration), v

def move_forward_distance(distance_cm, speed=NORMAL_SPEED, brake_after=False):
    duration, v = distance_to_duration_cm(distance_cm, speed, FORWARD_SPEED_TABLE)
    print(f"FORWARD: {distance_cm} cm @PWM {speed} (v≈{v:.1f} cm/s) -> {duration:.2f}s")
    run_for_duration(move_forward, duration, speed, brake_after=brake_after)

def move_backward_distance(distance_cm, speed=NORMAL_SPEED, brake_after=False):
    duration, v = distance_to_duration_cm(distance_cm, speed, REVERSE_SPEED_TABLE)
    print(f"REVERSE: {distance_cm} cm @PWM {speed} (v≈{v:.1f} cm/s) -> {duration:.2f}s")
    run_for_duration(move_backward, duration, speed, brake_after=brake_after)

# =============================================================================
# SECTION 16 — DISTANCE ESTIMATION FOR TIMED COMMANDS + TOTAL DISTANCE COUNTER
# =============================================================================
def estimate_distance_cm(duration, pwm, speed_table):
    """
    Estimate distance traveled during a timed move (open-loop).
    """
    v = lookup_with_interpolation(speed_table, pwm)  # cm/s
    return v * duration

def add_to_total_distance(distance_cm):
    """
    Add the absolute linear distance traveled to the running total.
    Turns do not call this.
    """
    global TOTAL_DISTANCE_CM
    TOTAL_DISTANCE_CM += abs(float(distance_cm))

# =============================================================================
# SECTION 17 — COMMAND PARSING
# =============================================================================
def parse_command(user_input):
    parts = user_input.strip().lower().split()
    if not parts:
        return None, []

    cmd = parts[0]
    args = []
    for p in parts[1:]:
        raw = p.strip().strip("[](){}").strip(",")
        try:
            args.append(float(raw))
        except ValueError:
            args.append(raw)
    return cmd, args

# =============================================================================
# SECTION 18 — MAIN LOOP
# =============================================================================
def main():
    global NORMAL_SPEED, TURN_SPEED, TOTAL_DISTANCE_CM

    print("--- Robot Control System Started ---")
    print("Timed motion (seconds):")
    print("  w [t]       -> forward for t seconds (prints estimated distance)")
    print("  s [t]       -> reverse for t seconds (prints estimated distance)")
    print("Angle turns (degrees):")
    print("  a [angle]   -> turn left by angle degrees")
    print("  d [angle]   -> turn right by angle degrees")
    print("Distance moves (cm):")
    print("  f [cm]      -> move forward by cm")
    print("  r [cm]      -> move reverse by cm")
    print("Speed control (PWM %):")
    print("  v [0-100]   -> set NORMAL_SPEED (forward/reverse PWM)")
    print("  t [0-100]   -> set TURN_SPEED (turning PWM)")
    print("Total distance:")
    print("  total       -> show total linear distance travelled so far (cm)")
    print("  reset       -> reset total distance counter to 0")
    print("Brake:")
    print("  b           -> toggle brake mode ON/OFF")
    print("Other:")
    print("  q           -> stop")
    print("  e           -> exit")
    print()

    brake_mode = False
    stop(coast=True, ramp=False)

    try:
        while True:
            user_input = input("Enter Command: ")
            cmd, args = parse_command(user_input)

            # ---- Timed forward (estimate distance + add to total) ----
            if cmd == 'w':
                duration = float(args[0]) if args else FWD_TIME
                run_for_duration(move_forward, duration, NORMAL_SPEED, brake_after=brake_mode)

                est = estimate_distance_cm(duration, NORMAL_SPEED, FORWARD_SPEED_TABLE)
                add_to_total_distance(est)
                print(f"Estimated forward distance: {est:.1f} cm (open-loop)")
                print(f"Total distance travelled: {TOTAL_DISTANCE_CM:.1f} cm")

            # ---- Timed reverse (estimate distance + add to total) ----
            elif cmd == 's':
                duration = float(args[0]) if args else REV_TIME
                run_for_duration(move_backward, duration, NORMAL_SPEED, brake_after=brake_mode)

                est = estimate_distance_cm(duration, NORMAL_SPEED, REVERSE_SPEED_TABLE)
                add_to_total_distance(est)
                print(f"Estimated reverse distance: {est:.1f} cm (open-loop)")
                print(f"Total distance travelled: {TOTAL_DISTANCE_CM:.1f} cm")

            # ---- Angle turns (do NOT add to total distance) ----
            elif cmd == 'a':
                if not args:
                    print("Example: a 90")
                else:
                    turn_left_angle(args[0], TURN_SPEED, brake_after=brake_mode)

            elif cmd == 'd':
                if not args:
                    print("Example: d 180")
                else:
                    turn_right_angle(args[0], TURN_SPEED, brake_after=brake_mode)

            # ---- Distance moves (add commanded distance to total) ----
            elif cmd == 'f':
                if not args:
                    print("Example: f 60")
                else:
                    dist = float(args[0])
                    move_forward_distance(dist, NORMAL_SPEED, brake_after=brake_mode)
                    add_to_total_distance(dist)
                    print(f"Total distance travelled: {TOTAL_DISTANCE_CM:.1f} cm")

            elif cmd == 'r':
                if not args:
                    print("Example: r 60")
                else:
                    dist = float(args[0])
                    move_backward_distance(dist, NORMAL_SPEED, brake_after=brake_mode)
                    add_to_total_distance(dist)
                    print(f"Total distance travelled: {TOTAL_DISTANCE_CM:.1f} cm")

            # ---- Speed controls ----
            elif cmd == 'v':
                if not args:
                    print("Example: v 60")
                else:
                    NORMAL_SPEED = int(clamp(args[0], 0, 100))
                    print(f"NORMAL_SPEED set to {NORMAL_SPEED}")

            elif cmd == 't':
                if not args:
                    print("Example: t 75")
                else:
                    TURN_SPEED = int(clamp(args[0], 0, 100))
                    print(f"TURN_SPEED set to {TURN_SPEED}")

            # ---- Total distance commands ----
            elif cmd == 'total':
                print(f"Total distance travelled: {TOTAL_DISTANCE_CM:.1f} cm")

            elif cmd == 'reset':
                TOTAL_DISTANCE_CM = 0.0
                print("Total distance counter reset to 0.0 cm")

            # ---- Brake toggle ----
            elif cmd == 'b':
                brake_mode = not brake_mode
                print(f"Brake mode is now: {'ON' if brake_mode else 'OFF'}")

            # ---- Stop / Exit ----
            elif cmd == 'q':
                stop(coast=not brake_mode, ramp=False)

            elif cmd == 'e':
                stop(coast=True, ramp=False)
                print("Exiting...")
                break

            else:
                print("Invalid command.")

    except KeyboardInterrupt:
        print("\nForce Stop Triggered.")
    finally:
        stop(coast=True, ramp=False)
        pwm_a.stop()
        pwm_b.stop()
        GPIO.cleanup()
        print("GPIO Cleaned up. System Shutdown.")

# =============================================================================
# SECTION 19 — ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()