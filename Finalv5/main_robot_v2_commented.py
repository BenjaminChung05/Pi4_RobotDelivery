# NOTE: OpenCV capture image as RGB and expect BGR image for display
"""
Multiprocessing line-following + image-recognition robot.

Process layout
──────────────
  main        – camera capture  ▸  motor control  ▸  display
  line_proc   – line detection  ▸  PID calculation
  img_proc    – ORB symbol / colour-shape recognition

Frame sharing uses multiprocessing.shared_memory (zero-copy),
so no pickling overhead between processes.
"""

from picamera2 import Picamera2                 # Pi Camera Module 2 driver (libcamera-based, replaces legacy PiCamera)
import cv2 as cv                                 # OpenCV – image processing (masks, contours, ORB, drawing)
import numpy as np                               # Fast array maths – all image pixels are numpy arrays
import time                                      # time.monotonic() for dt in PID & FPS (monotonic = never goes backwards)
import RPi.GPIO as GPIO                          # Direct pin control for L298N motor driver
from collections import deque                    # Fixed-size FIFO buffer for temporal debounce of detections
import multiprocessing as mp                     # True parallel processes (bypasses Python's GIL, uses all 4 Pi cores)
from multiprocessing import shared_memory        # Zero-copy frame sharing between processes – avoids pickling 500 KB/frame

# ══════════════════════════════════════════════════════════════
# PIN DECLARATIONS  (BCM)
# BCM = Broadcom pin numbering (GPIO17, not physical pin 11).
# L298N motor driver needs: 2 direction pins + 1 PWM pin per motor.
# IN1/IN2 HIGH-LOW = forward, LOW-HIGH = reverse, both LOW = brake.
# ══════════════════════════════════════════════════════════════
IN1, IN2 = 17, 18   # motor_a (Left) direction pins into L298N
IN3, IN4 = 22, 23   # motor_b (Right) direction pins into L298N
ENA, ENB = 12, 13   # PWM enable pins – duty cycle controls motor speed (0-100%)

# ══════════════════════════════════════════════════════════════
# FRAME CONFIG
# 480x360 chosen as sweet-spot: big enough for ORB keypoints,
# small enough to keep >=15 FPS on Pi 4. Higher res = slower.
# ══════════════════════════════════════════════════════════════
FRAME_W      = 480
FRAME_H      = 360
FRAME_SHAPE  = (FRAME_H, FRAME_W, 3)          # (rows, cols, channels) – numpy convention is H before W!
FRAME_NBYTES = FRAME_H * FRAME_W * 3          # Total bytes to allocate in shared memory (uint8 per channel)

# Display output frames (BGR because OpenCV's imshow expects BGR, not RGB)
# Workers write annotated frames here; main process reads them for cv.imshow.
LINE_DISP_SHAPE  = (180, FRAME_W, 3)          # Only bottom half of frame (line is on the ground in front of robot)
LINE_DISP_NBYTES = 180 * FRAME_W * 3
IMG_DISP_SHAPE   = (FRAME_H, FRAME_W, 3)      # Full frame because symbols could be anywhere
IMG_DISP_NBYTES  = FRAME_H * FRAME_W * 3

# ══════════════════════════════════════════════════════════════
# SPEED / PID CONSTANTS
# PWM duty cycles 0-100. Low values (20-27) chosen because:
#   - robot is lightweight -> high speeds overshoot PID corrections
#   - camera motion blur ruins ORB matching at high speed
# PUSH > NORMAL: used briefly when a symbol is detected to commit
# to the approach, preventing PID from stalling mid-turn.
# ══════════════════════════════════════════════════════════════
MOTOR_A_SPEED_NORMAL,  MOTOR_B_SPEED_NORMAL  = 20, 20   # Cruising speed – slow enough for clean ORB frames
MOTOR_A_SPEED_PUSH,    MOTOR_B_SPEED_PUSH    = 27, 27   # Boost when we've locked onto a symbol target

# PID constants tuned experimentally (Ziegler-Nichols-style trial & error):
#   KP high -> snappy but oscillates / weaves
#   KI tiny -> just enough to kill steady-state drift without wind-up
#   KD moderate -> damps oscillation from KP
KP, KI, KD         = 0.24, 0.005, 0.0315
X_CENTRE_REF        = 240                    # Target x-pixel (image is 480 wide -> centre = 240); error = 240 - cx
Y_CENTRE_REF        = 180                    # Only used as a visual reference dot; PID is x-only

# ══════════════════════════════════════════════════════════════════════
# IMAGE-RECOGNITION CONFIG
#
# Why ORB (Oriented FAST + Rotated BRIEF)?
#   - Free & patent-unencumbered (unlike SIFT/SURF).
#   - Binary descriptors -> matched with Hamming distance (very fast).
#   - Rotation + scale invariant -> robot doesn't need perfect approach angle.
#   - Runs in real-time on a Pi 4 (SIFT would not).
#
# SAMPLE_DICT format: { symbol_id : ([reference_image_files], match_threshold) }
# Threshold = minimum number of "good" ORB matches needed to confirm the symbol.
#   Lower threshold  -> more sensitive, more false positives.
#   Higher threshold -> stricter, may miss valid detections.
# Tuned per symbol because each has different feature density
# (e.g. fingerprint has many unique ridges -> threshold 20;
#  hazard triangle is simpler + often blurred -> threshold 6).
# ══════════════════════════════════════════════════════════════════════
SAMPLE_DICT = {
    0: (["button.png"],7),              # Round green button – few features, low threshold works
    1: (["fingerprint.pnsg"],20),       # BUG: typo ".pnsg" instead of ".png" – file won't load!
    2: (["qr.png"],17),                 # QR has many sharp corners – ORB loves it
    3: (["recycle.png"], 7),            # Triangle of arrows – moderate features
    4: (["hazard.png"],  6)             # Lowered because robot often sees it mid-motion (motion blur)
}

# Human-readable names used in print statements and the instruction lookup
SYMBOL_NAMES = {
    0: "Button",
    1: "FingerPrint",
    2: "QR",
    3: "Recycle",
    4: "Hazard",
}

# Colour-range filters for SYMBOL detection (background/card colour).
# HSV vs LAB chosen per colour based on which space separates it best:
#   - HSV: good when Hue is well-defined (pure green, pure yellow).
#   - LAB: better under uneven lighting because L (lightness) is
#     decoupled from A (green-red) and B (blue-yellow). Shadows
#     change L only, leaving chroma stable -> robust detection.
IMAGE_COLOUR_RANGES = {
    "Green":    {"space": "HSV", "lower": np.array([40,  60,  50]), "upper": np.array([85,  255, 255])},   # Button, Recycle background
    "Yellow":   {"space": "HSV", "lower": np.array([25, 150,  50]), "upper": np.array([35,  255, 255])},   # Hazard sign – narrow hue to avoid matching yellow LINE
    "Purple":   {"space": "LAB", "lower": np.array([0, 145,  60 ]), "upper": np.array([255, 195, 135])},   # FingerPrint / QR backing
    "Blue/Teal":{"space": "LAB", "lower": np.array([0 , 100,  60]), "upper": np.array([230, 165, 120])},   # Alternate FingerPrint / QR backing
    "Red":      {"space": "LAB", "lower": np.array([0 , 160, 130]), "upper": np.array([255, 255, 180])},   # Shape fallback (arrows)
    "Orange":   {"space": "LAB", "lower": np.array([0, 130, 165 ]), "upper": np.array([255, 180, 200])},   # Shape fallback
}

# Colour ranges for LINE following (the tape on the floor).
# Red needs TWO ranges because red hue wraps around 0°/180° in HSV –
# a single inRange() can't span that wrap, so we OR two masks together.
LINE_COLOUR_RANGES = {
    "Red":      {"lower_1": np.array([0, 100, 100]), "lower_2": np.array([160, 100, 100]), "upper_1": np.array([10, 255, 255]), "upper_2": np.array([180, 255, 255])},
    "Yellow":   {"lower": np.array([20, 80, 80]),   "upper": np.array([40, 255, 255])},     # Priority lane colour
    "Black":    {"lower": np.array([0, 0, 0]),      "upper": np.array([180, 255, 70])}      # Value<70 catches black regardless of hue
}

# Maps confirmed detection labels -> driving commands consumed by main loop.
# Note the Arrow labels are composed at detect time e.g. "Arrow (Left)".
LABEL_TO_INSTRUCTION = {
    "Arrow (Left)":    "TURN_LEFT",
    "Arrow (Right)":   "TURN_RIGHT",
    "Arrow (Up)":      "MOVE_FORWARD",
    "Button":    "STOP",
    "Hazard":  "STOP",
    "Recycle": "360-TURN",
}
86  # BUG: stray integer literal left in code – harmless (evaluates & is discarded) but should be removed
# ══════════════════════════════════════════════════════════════
# MOTOR HELPERS  (main process only – child processes NEVER touch GPIO,
# otherwise both would clash on the same hardware pins)
# ══════════════════════════════════════════════════════════════
def setup_gpio():
    """Initialise BCM mode, set all motor pins as outputs, start PWM at 50 Hz."""
    GPIO.setmode(GPIO.BCM)                        # Use GPIO numbers (not physical pin numbers)
    for pin in (IN1, IN2, IN3, IN4, ENA, ENB):
        GPIO.setup(pin, GPIO.OUT)                 # All motor pins are outputs
    # 50 Hz PWM chosen because L298N handles low-freq PWM well.
    # Higher freq (e.g. 1 kHz) can cause motor whine & driver overheating.
    pwm_motor_a = GPIO.PWM(ENA, 50);  pwm_motor_a.start(0)   # Start at 0% duty (motor stopped)
    pwm_motor_b = GPIO.PWM(ENB, 50);  pwm_motor_b.start(0)
    return pwm_motor_a, pwm_motor_b

def _drive_side(in_a, in_b, pwm, speed):
    """Drive one motor. Negative speed = reverse, positive = forward.
    The sign of `speed` encodes direction; abs(speed) is the PWM duty."""
    if speed < 0:
        GPIO.output(in_a, GPIO.HIGH)              # Reverse polarity
        GPIO.output(in_b, GPIO.LOW)
        speed = max(3, min(100, abs(speed)))      # Clamp: minimum 3% to overcome motor dead-band (static friction)
    else:
        GPIO.output(in_a, GPIO.LOW)               # Forward polarity
        GPIO.output(in_b, GPIO.HIGH)
        speed = max(3, min(100, speed))           # Clamp to PWM valid range (0-100)
    pwm.ChangeDutyCycle(speed)                    # Actually apply the new speed

def move_forward(pwm_motor_a, pwm_motor_b, speed_motor_a, speed_motor_b):
    """Name is slightly misleading: this is the universal drive function.
    Pass negative speed on one side to pivot, or both negative to reverse."""
    _drive_side(IN1, IN2, pwm_motor_a, speed_motor_a)
    _drive_side(IN3, IN4, pwm_motor_b, speed_motor_b)

def stop_motors(pwm_motor_a, pwm_motor_b):
    """Hard brake: pull all direction pins LOW + set PWM to 0 duty.
    Belt-and-braces – either alone would stop it, both = guaranteed stop."""
    for pin in (IN1, IN2, IN3, IN4):
        GPIO.output(pin, GPIO.LOW)
    pwm_motor_a.ChangeDutyCycle(0)
    pwm_motor_b.ChangeDutyCycle(0)

# ══════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════
def best_contour(mask):
    """Return the largest contour and its area from a binary mask.
    RETR_EXTERNAL  = only outer boundaries (ignores holes inside) – faster.
    CHAIN_APPROX_SIMPLE = store only corner points (compresses straight edges)."""
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0
    c = max(contours, key=cv.contourArea)         # Biggest blob wins – usually the real object, not noise
    return c, cv.contourArea(c)

# mp.Array of type 'c' is a shared char buffer. These helpers
# write/read null-terminated strings between processes without
# locking (we just overwrite the whole buffer atomically enough for our use).
def _write_str(arr: mp.Array, text: str, max_bytes: int):
    enc = text.encode()[:max_bytes - 1]           # Truncate to fit (leave 1 byte for null terminator)
    arr.raw = enc + b'\x00' * (max_bytes - len(enc))  # Pad with nulls so stale bytes don't leak through

def _read_str(arr: mp.Array) -> str:
    return arr.raw.rstrip(b'\x00').decode(errors='replace')  # Strip padding, tolerate broken bytes

# ══════════════════════════════════════════════════════════════
# LINE-FOLLOWING WORKER PROCESS
# ══════════════════════════════════════════════════════════════
def line_worker(
    shm_name,           # Name of camera-frame shared memory block (created by main)
    frame_lock,         # Lock protecting the camera frame during read
    shared_fid,         # Frame ID written by main (increments each new frame)
    my_fid,             # Last frame ID this worker processed (avoid duplicates)
    out_pid,            # OUTPUT: computed PID correction value
    out_cx,             # OUTPUT: line centroid X (for debug / display)
    out_cy,             # OUTPUT: line centroid Y
    out_has_line,       # OUTPUT: True if a line is currently visible
    out_lineArea,       # OUTPUT: area of followed contour (size hint)
    out_is_priority,    # OUTPUT: True if following Red/Yellow priority colour
    out_turn_cmd,       # OUTPUT: 1=exit-LEFT, 2=exit-RIGHT (junction exit signal)
    disp_shm_name,      # Shared memory name for the annotated display frame
    disp_lock,          # Lock for the display buffer
):
    # Attach to the same shared memory blocks created by main.
    # np.ndarray wraps the raw bytes so we can treat them as an image directly – zero copy.
    shm  = shared_memory.SharedMemory(name=shm_name)
    fbuf = np.ndarray(FRAME_SHAPE, dtype=np.uint8, buffer=shm.buf)

    disp_shm  = shared_memory.SharedMemory(name=disp_shm_name)
    disp_buf  = np.ndarray(LINE_DISP_SHAPE, dtype=np.uint8, buffer=disp_shm.buf)

    # PID state persists across iterations – integral accumulates, derivative needs previous error.
    pid_state = {'last_error': 0.0, 'integral': 0.0, 'last_time': time.monotonic()}
    pid_out = 0.0
    cx, cy = X_CENTRE_REF, 150          # Default centroid = image centre (avoids NaN before first detection)

    # FPS tracking with exponential moving average smoothing (0.9 old + 0.1 new).
    prev_fps_time = time.monotonic()
    smoothed_fps = 0.0

    # State for the "red-lane voting" algorithm (explained further below).
    lane_memory = None        # "Left" / "Right" once committed after enough votes
    red_left_votes = 0        # Times red was seen to the left of black
    red_right_votes = 0       # Times red was seen to the right of black
    was_on_red = False        # Edge-detect: were we just on red last iteration?
    was_on_yellow = False     # Edge-detect: prevents spamming yellow print logs

    while True:
        # ── FRAME-ID HANDSHAKE ─────────────────────────────────
        # Only process a frame once. If main hasn't written a new one,
        # sleep briefly and loop (busy-wait would burn CPU).
        fid = shared_fid.value
        if fid == my_fid.value:
            time.sleep(0.002)               # 2 ms sleep – 500 Hz polling is plenty for 30 FPS camera
            continue
        my_fid.value = fid                  # Mark this frame as "claimed" by us

        # ── FPS MEASUREMENT (diagnostic only) ──────────────────
        now = time.monotonic()
        dt_fps = now - prev_fps_time
        prev_fps_time = now
        if dt_fps > 0:
            current_fps = 1.0 / dt_fps
            # Exponential moving average = low-pass filter. Without smoothing,
            # display jitters wildly between frames.
            smoothed_fps = (0.9 * smoothed_fps) + (0.1 * current_fps)

        # ── COPY FRAME OUT OF SHARED MEMORY ────────────────────
        # Lock held only for the fast .copy(); all heavy processing
        # happens on our private copy so main can write the next frame.
        with frame_lock:
            frame = fbuf.copy()

        # Crop to the BOTTOM HALF only – the line is physically just in
        # front of the wheels; processing the whole image wastes CPU and
        # risks detecting lines in the distance we haven't reached yet.
        crop_rgb = frame[180:360, :]
        crop_bgr = cv.cvtColor(crop_rgb, cv.COLOR_RGB2BGR)   # BGR for display (imshow convention)

        # Pre-processing: small Gaussian blur removes high-frequency sensor
        # noise so colour thresholding produces cleaner masks.
        blur = cv.GaussianBlur(crop_rgb, (3, 3), 0)
        hsv  = cv.cvtColor(blur, cv.COLOR_RGB2HSV)           # HSV separates Hue from brightness – better for colour segmentation than RGB

        # Build one binary mask per candidate line colour.
        # inRange returns 255 where pixel is inside the range, 0 otherwise.
        mask_black = cv.inRange(hsv, LINE_COLOUR_RANGES["Black"]["lower"], LINE_COLOUR_RANGES["Black"]["upper"])
        mask_yellow = cv.inRange(hsv, LINE_COLOUR_RANGES["Yellow"]["lower"], LINE_COLOUR_RANGES["Yellow"]["upper"])
        # Red hue wraps around 0°/180° -> need TWO ranges OR-ed together.
        mask_red = cv.bitwise_or(cv.inRange(hsv, LINE_COLOUR_RANGES["Red"]["lower_1"], LINE_COLOUR_RANGES["Red"]["upper_1"]),
                                 cv.inRange(hsv, LINE_COLOUR_RANGES["Red"]["lower_2"], LINE_COLOUR_RANGES["Red"]["upper_2"]))

        # Get the biggest blob per colour – that's our candidate line.
        cnt_black,  area_black  = best_contour(mask_black)
        cnt_yellow, area_yellow = best_contour(mask_yellow)
        cnt_red,    area_red    = best_contour(mask_red)

        # Bounding box dimensions are used as a shape sanity check.
        # A real line is a long rectangle – stray blobs fail the w/h test.
        def get_bbox(cnt): return cv.boundingRect(cnt) if cnt is not None else (0, 0, 0, 0)
        xr, yr, wr, hr = get_bbox(cnt_red)
        xy, yy, wy, hy = get_bbox(cnt_yellow)
        xb, yb, wb, hb = get_bbox(cnt_black)

        # ── COLOUR PRIORITY LADDER ─────────────────────────────
        # Red beats Yellow beats Black. Red/Yellow are "priority" lanes
        # signalling special manoeuvres at junctions. Black is the default.
        if area_red > 4500 and hr > 20 and wr > 20:
            cnts = [cnt_red] if cnt_red is not None else []
            follow_colour = "Red"
            draw_colour   = (0, 0, 255)                     # BGR red
        elif area_yellow > 4500 and hy > 135 and wy > 20:
            # Yellow needs a TALL bbox (h>135) – a short yellow blob is likely
            # noise or a yellow symbol, not the long yellow lane tape.
            print(f"{hy}")
            cnts = [cnt_yellow] if cnt_yellow is not None else []
            follow_colour = "Yellow"
            draw_colour   = (0, 255, 255)                    # BGR yellow

            if not was_on_yellow:                            # Print once on entry (edge-triggered log)
                print(f"\n[line_worker] ---> DETECTING YELLOW LINE (Box: {wy}x{hy}) <--- \n")
                was_on_yellow = True

        elif area_black > 4500:
            cnts = [cnt_black] if cnt_black is not None else []
            follow_colour = "Black"
            draw_colour   = (0, 255, 0)                      # BGR green (visual cue we're on normal track)
        else:
            cnts = []                                        # No usable line in view this frame
            follow_colour = "None"
            draw_colour   = (0, 255, 0)

        if follow_colour != "Yellow":
            was_on_yellow = False                            # Reset edge latch when we leave yellow
            
        has_line = False
        current_area = 0.0

        if cnts:
            has_line = True

            largest_contour = max(cnts, key=cv.contourArea)
            area = cv.contourArea(largest_contour)
            current_area = area

            # Draw debug overlays – visible in the "Line Following" window.
            x, y, w, h = cv.boundingRect(largest_contour)
            cv.rectangle(crop_bgr, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv.putText(crop_bgr, f"w:{w} h:{h}", (x, max(10, y - 10)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            # ── CENTROID via IMAGE MOMENTS ──────────────────────
            # cx = m10/m00, cy = m01/m00. This is the CENTRE OF MASS of
            # the contour – more stable than bounding-box centre for
            # odd shapes (curves, forks). Falls back to bbox centre if
            # m00==0 (degenerate zero-area contour – shouldn't happen but safe).
            M = cv.moments(largest_contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = x + (w // 2), y + (h // 2)

            cv.drawContours(crop_bgr, [largest_contour], -1, draw_colour, 2)

            # ── PID CONTROLLER (x-axis only) ────────────────────
            # error > 0 -> line is LEFT of image centre -> we must turn LEFT
            # error < 0 -> line is RIGHT -> turn RIGHT
            # pid_out is added to right motor, subtracted from left -> steering.
            error = X_CENTRE_REF - cx
            now   = time.monotonic()
            dt    = now - pid_state['last_time']     # Real time between iterations – accounts for variable FPS
            pid_state['last_time'] = now

            P = KP * error                           # Proportional: how FAR off centre we are
            pid_state['integral'] += error * dt      # Integral accumulates – cancels steady drift
            I = KI * pid_state['integral']
            # Derivative gated: if dt<0.01 it's either our first frame or
            # jitter -> 1/dt explodes. Zero it out for safety.
            D = (KD * (error - pid_state['last_error']) / dt) if dt > 0.01 else 0.0
            pid_state['last_error'] = error
            pid_out = P + I + D                      # Sum = steering correction
        else:
            # No line this frame – reset last_time so when line returns,
            # dt isn't a huge number that would cause a derivative kick.
            pid_state['last_time'] = time.monotonic()

        # ── LANE MEMORY / RED VOTING ALGORITHM ─────────────────
        # Problem: when the robot arrives at a junction marked by RED
        # tape, we must remember which side of the track the red appeared
        # on so that when we exit back onto BLACK we can turn correctly.
        #
        # Solution: while on RED, compare red centroid (cx) with black
        # centroid (cx_blk). If red is left of black, vote LEFT, else RIGHT.
        # Majority vote over many frames = robust against single-frame noise.
        # Once >5 / >=6 votes reached, LOCK the decision (lane_memory).
        # When we leave RED for BLACK, emit a turn command (1=L, 2=R)
        # which the main process reads and executes as a physical turn.
        if follow_colour == "Red":
            if lane_memory is None:                          # Still voting – not locked yet
                cx_blk = X_CENTRE_REF                        # Default: assume black is at image centre if not visible
                if cnt_black is not None and area_black > 1000:
                    M_blk = cv.moments(cnt_black)
                    if M_blk['m00'] != 0:
                        cx_blk = int(M_blk['m10'] / M_blk['m00'])

                if cx < cx_blk: red_left_votes += 1          # Red-blob is LEFT of black-blob in image
                else: red_right_votes += 1                    # Red-blob is RIGHT

                if red_left_votes > 5:                       # Asymmetric threshold (>5 vs >=6) – tiny bias, practical difference small
                    lane_memory = "Left"
                    print(f"\n[line_worker] VOTING COMPLETE: Locked LEFT ({red_left_votes} votes)\n")
                elif red_right_votes >= 6:
                    lane_memory = "Right"
                    print(f"\n[line_worker] VOTING COMPLETE: Locked RIGHT ({red_right_votes} votes)\n")
            was_on_red = True                                # Remember for transition detection below

        elif follow_colour == "Black":
            # EDGE: we just transitioned Red -> Black (exited the junction).
            # Fire the turn command once, then reset voting state.
            if was_on_red:
                if lane_memory == "Left": out_turn_cmd.value = 1   # main sees 1 -> pivot LEFT
                elif lane_memory == "Right": out_turn_cmd.value = 2 # main sees 2 -> pivot RIGHT
                was_on_red = False; lane_memory = None; red_left_votes = 0; red_right_votes = 0
        else:
            # WARNING: this rebinds the CONSTANTS as LOCAL variables only –
            # it has NO effect on main's base speed. Effectively dead code.
            MOTOR_A_SPEED_NORMAL, MOTOR_B_SPEED_NORMAL = 21, 21
            was_on_red = False

        # ── DEBUG OVERLAYS on display frame ────────────────────
        cv.circle(crop_bgr, (X_CENTRE_REF, Y_CENTRE_REF), 5, (0, 255, 255), -1)   # Yellow dot = desired centre
        cv.circle(crop_bgr, (cx, cy), 5, (0, 0, 255), -1)                          # Red dot = actual line centroid
        cv.putText(crop_bgr, f"FPS: {int(smoothed_fps)}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Write annotated frame to display shared memory for main to show.
        with disp_lock:
            np.copyto(disp_buf, crop_bgr)

        # Publish all outputs back to main via shared mp.Value slots.
        out_pid.value = pid_out
        out_cx.value = cx
        out_cy.value = cy
        out_is_priority.value = (follow_colour in ["Red", "Yellow"])   # Tells main+img_worker to slow down / suppress shape detection
        out_lineArea.value = current_area
        out_has_line.value = has_line


# ══════════════════════════════════════════════════════════════
# IMAGE-RECOGNITION HELPERS
# ══════════════════════════════════════════════════════════════
def orb_match_symbol(bf, ref_entries, des_scene, threshold):
    """Try to match scene against every reference image of one symbol.
    Returns True on the FIRST reference that passes, so ordering matters
    (but with only 1 image per symbol here, it's effectively single-shot).

    LOWE'S RATIO TEST (0.72):
    - knnMatch k=2 returns the two best matches for each descriptor.
    - If best match is much closer than second-best (m[0] < 0.72*m[1]),
      it's a confident match. Otherwise it's ambiguous and rejected.
    - 0.72 is a tuned balance: 0.75 (standard) was too permissive here,
      0.65 would miss too many.
    - Threshold = minimum number of "good" matches to accept as detection.
    """
    for ref in ref_entries:
        if ref["des"] is None: continue                      # Skip refs that failed to load (e.g. typo'd filename)
        matches = bf.knnMatch(ref["des"], des_scene, k=2)    # k=2: get best + second best per descriptor
        good = sum(1 for m in matches if len(m) == 2 and m[0].distance < 0.72 * m[1].distance)
        if good >= threshold: return True, good              # Enough confident matches -> symbol found
    return False, 0

def _detect_shape(contour):
    """FALLBACK shape detector when ORB finds nothing.
    Specifically tuned to recognise ARROWS (concave polygons)."""
    area = cv.contourArea(contour)
    peri = cv.arcLength(contour, True)                       # True = closed contour

    # approxPolyDP simplifies the contour to ~corner points.
    # epsilon = 0.02*peri means "points within 2% of perimeter are merged".
    approx = cv.approxPolyDP(contour, 0.02 * peri, True)
    v = len(approx)                                          # Number of vertices after simplification

    # Solidity = (area) / (convex hull area).
    # Convex shapes have solidity ≈ 1.0.
    # Arrows are concave -> solidity between ~0.5-0.7.
    hull_area = cv.contourArea(cv.convexHull(contour))
    if hull_area == 0: return "Unknown", None

    solidity = area / hull_area
    is_convex = cv.isContourConvex(approx)                   # Arrows MUST NOT be convex
    circ = 4 * np.pi * area / (peri * peri)                  # Circularity: 1.0 for circle, lower for elongated shapes

    # ARROW signature: 7-10 vertices, concave, solidity 0.52-0.68, circ >= 0.15
    if 7 <= v <= 10 and not is_convex and 0.52 <= solidity <= 0.68 and circ >= 0.15:
        # Find arrow TIP = contour point FARTHEST from the centroid.
        M = cv.moments(contour)
        if M["m00"] == 0: return "Arrow", "Unknown"
        cx_ = int(M["m10"] / M["m00"])
        cy_ = int(M["m01"] / M["m00"])
        # Farthest-point algorithm: O(N) scan over every contour point.
        far = max(contour, key=lambda p: (p[0][0]-cx_)**2 + (p[0][1]-cy_)**2)
        dx, dy = far[0][0] - cx_, far[0][1] - cy_
        # Classify direction by which axis the tip is farther along.
        # NOTE: in image coords +x=right, +y=down. Label logic may be
        # inverted on purpose if the camera is mounted rotated.
        if abs(dx) > abs(dy): direction = "Left" if dx > 0 else "Right"
        else: direction = "Up" if dy > 0 else "Down"
        return "Arrow", direction
    return "Unknown", None


# ══════════════════════════════════════════════════════════════
# IMAGE-RECOGNITION WORKER PROCESS
# ══════════════════════════════════════════════════════════════
def image_worker(
    shm_name, frame_lock, shared_fid, my_fid,
    out_found, out_label, out_instruction, out_instruction_ready,
    disp_shm_name, disp_lock, out_is_priority,
):
    # Attach to the shared camera frame (same as line_worker).
    shm  = shared_memory.SharedMemory(name=shm_name)
    fbuf = np.ndarray(FRAME_SHAPE, dtype=np.uint8, buffer=shm.buf)

    disp_shm = shared_memory.SharedMemory(name=disp_shm_name)
    disp_buf = np.ndarray(IMG_DISP_SHAPE, dtype=np.uint8, buffer=disp_shm.buf)

    # ── ORB DETECTOR + MATCHER SETUP ───────────────────────────
    # nfeatures=500   : cap on keypoints per frame (more=slower, diminishing returns)
    # nlevels=8       : image pyramid levels -> scale invariance (symbol works at any distance)
    # fastThreshold=17: FAST corner detector sensitivity. Lower=more keypoints, noisier.
    #                    17 is a compromise: enough keypoints at normal lighting,
    #                    not overwhelmed by noise in shadow.
    orb = cv.ORB_create(nfeatures=500, nlevels=8, fastThreshold=17)

    # Brute-Force Matcher with Hamming distance (ORB uses BINARY descriptors).
    # crossCheck=False because we use Lowe's ratio test instead (compatible with knnMatch).
    bf  = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

    # ── LOAD REFERENCE IMAGES ONCE ─────────────────────────────
    # Compute keypoints+descriptors for each reference image up front.
    # Doing this every frame would be a massive waste – references never change.
    reference_data = []
    for symbol_id, (img_files, threshold) in SAMPLE_DICT.items():
        refs = []
        for img_file in img_files:
            img = cv.imread(img_file, cv.IMREAD_GRAYSCALE)   # ORB works on grayscale only
            if img is None:
                # Graceful degradation – log warning, keep running without this ref.
                print(f"[img_worker] WARNING: reference image '{img_file}' not found – skipping.")
                refs.append({"filename": img_file, "kp": None, "des": None})
                continue
            kp, des = orb.detectAndCompute(img, None)        # kp=keypoints, des=binary descriptors
            refs.append({"filename": img_file, "kp": kp, "des": des})
            print(f"[img_worker] Loaded '{img_file}' — {len(kp)} keypoints.")

        reference_data.append({"id": symbol_id, "name": SYMBOL_NAMES[symbol_id], "threshold": threshold, "refs": refs})

    # Dict lookup by symbol_id – used by the colour->symbol routing below.
    ref_by_id = {entry["id"]: entry for entry in reference_data}

    # ── DEBOUNCE STATE ─────────────────────────────────────────
    # maxlen=3 means: require 3 consecutive matching detections before
    # reporting a symbol. Kills single-frame false positives.
    label_history = deque(maxlen = 3)

    cooldown_counter = 0                                      # After a confirmed detection, ignore 8 frames (avoid re-firing same symbol)
    missed_frames = 0                                         # Frames since last positive – if >4, clear history

    while True:
        fid = shared_fid.value
        if fid == my_fid.value:
            time.sleep(0.002)
            continue
        my_fid.value = fid

        with frame_lock: frame = fbuf.copy()                 # Grab private copy so main isn't blocked
        display_bgr = cv.cvtColor(frame, cv.COLOR_RGB2BGR)   # For annotation + imshow

        # Per-frame working vars (reset each iteration).
        found = False
        label = ""
        instruction = ""
        best_contour_for_display = None

        # Is line_worker currently tracking a priority (red/yellow) lane?
        # If yes we skip shape detection to avoid false arrows on the lane tape.
        current_is_priority = bool(out_is_priority.value)

        # Precompute both colour spaces once – per-colour code below just picks the right one.
        blurred = cv.GaussianBlur(frame, (3, 3), 0)
        HSV = cv.cvtColor(blurred, cv.COLOR_RGB2HSV)          # Good for pure Green/Yellow
        LAB = cv.cvtColor(blurred, cv.COLOR_RGB2LAB)          # Better for shadows (Red/Purple/Blue/Orange)

        # ── CANDIDATE GENERATION ──────────────────────────────
        # For every configured colour: build its mask, find contours,
        # keep only blobs with area >= 1200 px (smaller = noise).
        all_candidates = []
        for colour_name, params in IMAGE_COLOUR_RANGES.items():
            src  = HSV if params["space"] == "HSV" else LAB
            mask = cv.inRange(src, params["lower"], params["upper"])

            for cnt in cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]:
                a = cv.contourArea(cnt)
                if a >= 1200: all_candidates.append((a, cnt, colour_name))

        # Prioritise the 3 LARGEST blobs – realistic symbols are sizeable.
        # Ignoring smaller ones saves ORB CPU time.
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = all_candidates[:3]

        # ── COLOUR → SYMBOL ROUTING ───────────────────────────
        # Each symbol has a known background colour. We only try ORB on
        # refs whose colour is likely present. Huge speedup vs matching
        # against ALL symbols for every blob.
        #   Yellow    -> Hazard(4)
        #   Blue/Teal -> Fingerprint(1) or QR(2)
        #   Purple    -> Fingerprint(1) or QR(2)
        #   Green     -> Button(0) or Recycle(3)
        colour_to_ids = {"Yellow": [4], "Blue/Teal": [1, 2], "Purple": [1, 2], "Green": [0, 3]}
        orb_eligible = any(colour in colour_to_ids for _, _, colour in top_candidates)

        # ORB on the scene is expensive – only compute it if at least one
        # candidate blob has a colour that maps to a known symbol.
        des_s = None
        if orb_eligible:
            gray_scene = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
            _, des_s   = orb.detectAndCompute(gray_scene, None)

        # ── DETECTION LOOP (per candidate, largest first) ─────
        for area, cnt, detected_colour in top_candidates:
            # Step 1 – try ORB symbol match if colour is known.
            if detected_colour in colour_to_ids and des_s is not None:
                for sym_id in colour_to_ids[detected_colour]:
                    entry = ref_by_id[sym_id]
                    matched, good_count = orb_match_symbol(bf, entry["refs"], des_s, entry["threshold"])
                    if matched:
                        label = entry["name"]; found = True; best_contour_for_display = cnt
                        break                                # Stop after first confident symbol

            # Step 2 – ARROW fallback via shape geometry.
            # Skipped when on priority lane (tape blobs would register as arrows).
            if not found and not current_is_priority:
                shape, direction = _detect_shape(cnt)
                if shape != "Unknown":
                    label = shape + (f" ({direction})" if direction else "")
                    found = True; best_contour_for_display = cnt
            if found: break                                  # One detection per frame is enough

        # ── CONFIRMATION / DEBOUNCE / COOLDOWN ─────────────────
        # Rationale: a single-frame detection could be noise. We require
        # 3 identical consecutive labels before accepting. After accepting
        # we enter an 8-frame cooldown so we don't re-trigger on the same
        # target. If we miss >4 frames we reset history (not same symbol).
        if found and cooldown_counter == 0:
            label_history.append(label)
            missed_frames = 0
            # Full buffer + all same label = confirmed detection.
            if len(label_history) == label_history.maxlen and len(set(label_history)) == 1:
                confirmed_label = label_history[0]
                instruction     = LABEL_TO_INSTRUCTION.get(confirmed_label, "")   # Map label -> driving command

                print(f"----------\n[img_worker] Detected : {confirmed_label}")
                if instruction: print(f"[img_worker] Instruction: {instruction}")
                print(f"----------")

                label_history.clear()
                cooldown_counter = 8                         # Ignore detections for next 8 frames
        else:
            if cooldown_counter > 0: cooldown_counter -= 1
            missed_frames += 1
            if missed_frames > 4:                            # Lost the target -> start fresh
                label_history.clear()
                instruction = ""

        # ── DISPLAY OVERLAYS ──────────────────────────────────
        if current_is_priority:
            # Visual indicator that ORB is effectively off right now.
            cv.putText(display_bgr, "Line Following Only", (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if best_contour_for_display is not None:
            # Draw bounding box around detected symbol with its label.
            x, y, w, h = cv.boundingRect(best_contour_for_display)
            cv.rectangle(display_bgr, (x, y), (x + w, y + h), (255, 0, 0), 3)
            if label:
                cv.putText(display_bgr, label, (x, max(10, y - 10)), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        elif label:
            # No bbox to draw on (unusual) – just show label top-left.
            cv.putText(display_bgr, label, (10, 80), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        with disp_lock: np.copyto(disp_buf, display_bgr)

        # Publish results to main.
        out_found.value = found
        _write_str(out_label, label, 64)
        # Only raise the "ready" flag if there's an actual instruction to act on.
        # Main resets this flag after consuming – classic handshake pattern.
        if instruction:
            _write_str(out_instruction, instruction, 32)
            out_instruction_ready.value = True


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    # ── SHARED MEMORY ALLOCATION ───────────────────────────────
    # create=True -> OS allocates a brand-new block. Workers attach via name.
    # Keeps one canonical copy of each frame; no IPC serialisation cost.
    shm  = shared_memory.SharedMemory(create=True, size=FRAME_NBYTES)
    fbuf = np.ndarray(FRAME_SHAPE, dtype=np.uint8, buffer=shm.buf)    # numpy view over raw bytes

    # Two additional shared blocks for the ANNOTATED display frames
    # (workers write these, main reads them for imshow).
    line_disp_shm = shared_memory.SharedMemory(create=True, size=LINE_DISP_NBYTES)
    img_disp_shm  = shared_memory.SharedMemory(create=True, size=IMG_DISP_NBYTES)
    line_disp_buf = np.ndarray(LINE_DISP_SHAPE, dtype=np.uint8, buffer=line_disp_shm.buf)
    img_disp_buf  = np.ndarray(IMG_DISP_SHAPE,  dtype=np.uint8, buffer=img_disp_shm.buf)
    line_disp_lock, img_disp_lock = mp.Lock(), mp.Lock()              # One lock per display buffer

    # Lock protecting the main camera frame (main writes, workers read).
    frame_lock = mp.Lock()

    # Frame-ID synchronisation primitives.
    # 'i' = signed int, 'd' = double, 'b' = bool (byte).
    shared_fid = mp.Value('i',  0)       # Main increments each new frame
    line_fid   = mp.Value('i', -1)       # line_worker's own counter (starts -1 so frame 0 is "new")
    img_fid    = mp.Value('i', -1)       # image_worker's own counter

    # ── CROSS-PROCESS OUTPUT SLOTS ─────────────────────────────
    # These are the "return channels" from workers back to main.
    out_pid = mp.Value('d', 0.0); out_cx = mp.Value('i', X_CENTRE_REF); out_cy = mp.Value('i', Y_CENTRE_REF)
    out_lineArea = mp.Value('d', 0.0); out_has_line = mp.Value('b', False)

    out_found = mp.Value('b', False); out_label = mp.Array('c', 64)       # label string up to 64 chars
    out_instruction = mp.Array('c', 32); out_instruction_ready = mp.Value('b', False)   # Handshake flag
    out_is_priority = mp.Value('b', False); out_turn_cmd = mp.Value('i', 0)             # 0=none,1=L,2=R exit

    # ── SPAWN WORKER PROCESSES ────────────────────────────────
    # daemon=True = kill workers automatically when main exits (safety net).
    p_line = mp.Process(
        target=line_worker,
        args=(shm.name, frame_lock, shared_fid, line_fid, out_pid, out_cx, out_cy, out_has_line, out_lineArea, out_is_priority, out_turn_cmd, line_disp_shm.name, line_disp_lock),
        daemon=True, name="LineWorker"
    )
    p_img = mp.Process(
        target=image_worker,
        args=(shm.name, frame_lock, shared_fid, img_fid, out_found, out_label, out_instruction, out_instruction_ready, img_disp_shm.name, img_disp_lock, out_is_priority),
        daemon=True, name="ImgWorker"
    )
    p_line.start(); p_img.start()
    print("[main] Workers started.")

    # Initialise motor driver AFTER spawning children (children inherit the
    # memory state BEFORE this call – they don't need GPIO and shouldn't have it).
    pwm_motor_a, pwm_motor_b = setup_gpio()

    # ── CAMERA INITIALISATION ─────────────────────────────────
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": (FRAME_W, FRAME_H)}))
    picam2.start()
    time.sleep(2.0)                                               # Let AE/AWB stabilise – first frames are bad
    print("[main] Camera ready. Press ESC to quit.")

    line_loss_counter = 0                                         # Counts consecutive frames without a line (recovery logic)
    active_instruction = ""                                       # Currently-executing manoeuvre
    
    try:
        while True:
            # ── CAPTURE NEW FRAME ──────────────────────────────
            RGB = picam2.capture_array()
            # Picamera2 sometimes returns RGBA (4-ch). Strip alpha so our
            # (H, W, 3) shared buffer doesn't crash on assignment.
            if RGB.ndim == 3 and RGB.shape[2] == 4: RGB = RGB[:, :, :3]

            # Atomic write to shared memory, then bump frame ID – workers
            # spinning on shared_fid will wake up and start processing.
            with frame_lock: np.copyto(fbuf, RGB)
            shared_fid.value += 1

            # ── READ LATEST WORKER OUTPUTS ─────────────────────
            pid, cx, cy = out_pid.value, out_cx.value, out_cy.value
            has_line, found = bool(out_has_line.value), bool(out_found.value)
            label = _read_str(out_label)
            current_is_priority = bool(out_is_priority.value)

            # Handshake: img_worker raises the flag when a NEW instruction
            # is ready. We consume it and clear the flag in one step.
            new_instr= ""
            if out_instruction_ready.value:
                new_instr = _read_str(out_instruction)
                out_instruction_ready.value = False

            # ── HIGH-PRIORITY: JUNCTION-EXIT TURN (from red lane) ─
            # line_worker sets turn_cmd when it exits a red priority lane
            # onto black. We execute the physical turn immediately (blocking).
            turn_cmd = out_turn_cmd.value
            if turn_cmd != 0:
                if turn_cmd == 1:
                    print("[main] Exiting priority line — turning LEFT")
                    move_forward(pwm_motor_a, pwm_motor_b, MOTOR_A_SPEED_PUSH, MOTOR_B_SPEED_PUSH); time.sleep(0.25)   # Push forward briefly to clear junction
                    move_forward(pwm_motor_a, pwm_motor_b, -40, 55); time.sleep(0.8)                                  # Pivot: left wheel reverses, right drives = rotate LEFT
                elif turn_cmd == 2:
                    print("[main] Exiting priority line — turning RIGHT")
                    move_forward(pwm_motor_a, pwm_motor_b, MOTOR_A_SPEED_PUSH, MOTOR_B_SPEED_PUSH); time.sleep(0.25)
                    move_forward(pwm_motor_a, pwm_motor_b, 55, -40); time.sleep(0.8)                                  # Mirror: left drives, right reverses = rotate RIGHT
                out_turn_cmd.value = 0                                                                                # Acknowledge – clear so we don't re-trigger

            # Latch new instruction (STOP / TURN_LEFT etc.) for execution below.
            if new_instr:
                active_instruction = new_instr
                print(f"[main] Instruction '{active_instruction}' confirmed.")
                _write_str(out_instruction, "", 32)           # Clear worker's buffer to avoid re-read

            # ── EXECUTE INSTRUCTION (blocking manoeuvre) ───────
            # Each branch: brief PID blend-out, then hard-coded motor pattern,
            # then stop_motors. After completion active_instruction = "" so
            # we fall back to normal line-following next iteration.
            if active_instruction:
                if active_instruction == "TURN_LEFT":
                    # Small "steer into" moment first, then sharp pivot.
                    move_forward(pwm_motor_a, pwm_motor_b, MOTOR_A_SPEED_NORMAL - pid, MOTOR_B_SPEED_NORMAL + pid); time.sleep(0.1)
                    move_forward(pwm_motor_a, pwm_motor_b, -45, 55); time.sleep(0.6)      # Pivot LEFT
                    stop_motors(pwm_motor_a, pwm_motor_b); active_instruction = ""
                elif active_instruction == "TURN_RIGHT":
                    move_forward(pwm_motor_a, pwm_motor_b, MOTOR_A_SPEED_NORMAL - pid, MOTOR_B_SPEED_NORMAL + pid); time.sleep(0.1)
                    move_forward(pwm_motor_a, pwm_motor_b, 55, -35); time.sleep(0.5)      # Pivot RIGHT (slightly asymmetric tuning)
                    stop_motors(pwm_motor_a, pwm_motor_b); active_instruction = ""
                elif active_instruction == "MOVE_FORWARD":
                    move_forward(pwm_motor_a, pwm_motor_b, MOTOR_A_SPEED_NORMAL, MOTOR_B_SPEED_NORMAL); time.sleep(1)   # Blind forward 1 s
                    active_instruction = ""
                elif active_instruction == "STOP":
                    stop_motors(pwm_motor_a, pwm_motor_b); time.sleep(2)                  # Full stop, 2 s pause
                    move_forward(pwm_motor_a, pwm_motor_b, 50, 50); time.sleep(0.2)       # Nudge forward to clear the symbol from view
                    active_instruction = ""
                elif active_instruction == "360-TURN":
                    move_forward(pwm_motor_a, pwm_motor_b, MOTOR_A_SPEED_NORMAL, MOTOR_B_SPEED_NORMAL); time.sleep(0.5) # Approach the marker
                    move_forward(pwm_motor_a, pwm_motor_b, -70, 70); time.sleep(1.8)                                    # In-place spin 1.8 s
                    stop_motors(pwm_motor_a, pwm_motor_b); active_instruction = ""
            else:
                # ── NORMAL LINE FOLLOWING MODE ─────────────────
                if current_is_priority:
                    # On red/yellow lane – slow right down so ORB has clean frames.
                    L_base, R_base = 24, 23
                else:
                    # When we've detected a symbol, push harder to close the gap;
                    # otherwise cruise at normal.
                    L_base = MOTOR_A_SPEED_PUSH if found else MOTOR_A_SPEED_NORMAL
                    R_base = MOTOR_B_SPEED_PUSH if found else MOTOR_B_SPEED_NORMAL

                if has_line:
                    # Differential steering: subtract PID on one side, add on the other.
                    move_forward(pwm_motor_a, pwm_motor_b, L_base - pid, R_base + pid)
                    line_loss_counter = 0                    # Reset – we still have the line
                else:
                    # ── LINE-LOST RECOVERY ─────────────────────
                    # Phase 1 (<=8 frames): keep going straight – probably momentary occlusion.
                    # Phase 2 (>8 frames): pivot toward the side the PID was last pushing
                    #                      (pid sign encodes last known line direction).
                    if line_loss_counter <= 8:
                        move_forward(pwm_motor_a, pwm_motor_b, MOTOR_A_SPEED_NORMAL, MOTOR_B_SPEED_NORMAL)
                        line_loss_counter += 1
                    else:
                        if pid > 0: move_forward(pwm_motor_a, pwm_motor_b, -45, 55)    # Search LEFT
                        else: move_forward(pwm_motor_a, pwm_motor_b,  55, -45)         # Search RIGHT

            # ── DISPLAY ────────────────────────────────────────
            # Copy annotated frames out of shared memory so workers can keep writing.
            with line_disp_lock: line_frame = line_disp_buf.copy()
            with img_disp_lock: img_frame = img_disp_buf.copy()

            if active_instruction:
                cv.putText(line_frame, f"CMD: {active_instruction}", (10, 220), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv.putText(img_frame, f"CMD: {active_instruction}", (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv.imshow("Line Following", line_frame)
            # cv.imshow("Image Detection", img_frame)          # Commented out to save display bandwidth/CPU

            # ESC key terminates the loop. waitKey(1) yields 1ms back to the OS
            # which is essential for cv to actually render the window.
            if cv.waitKey(1) & 0xFF == 27: break

    except KeyboardInterrupt: print("\n[main] Ctrl+C received — stopping.")
    except Exception as e:
        # Full traceback helps diagnose bugs without losing info.
        import traceback; print(f"\n[main] ERROR: {e}"); traceback.print_exc()
    finally:
        # ── ORDERED GRACEFUL SHUTDOWN ──────────────────────────
        # Each step wrapped in try/except: we want shutdown to continue
        # even if one step fails (e.g. GPIO already cleaned up elsewhere).
        print("[main] Shutting down…")

        # 1. Stop motors FIRST – robot must not keep running after crash.
        try: stop_motors(pwm_motor_a, pwm_motor_b)
        except: pass

        # 2. Stop PWM signal generators + release Python references.
        try: pwm_motor_a.stop(); pwm_motor_b.stop(); del pwm_motor_a, pwm_motor_b
        except: pass

        # 3. Kill worker processes. terminate() sends SIGTERM; join() waits.
        p_line.terminate(); p_line.join()
        p_img.terminate(); p_img.join()

        # 4. Release shared memory. .close() detaches, .unlink() deletes it
        #    from /dev/shm – otherwise it leaks until reboot.
        try: shm.close(); shm.unlink(); line_disp_shm.close(); line_disp_shm.unlink(); img_disp_shm.close(); img_disp_shm.unlink()
        except: pass

        # 5. GPIO cleanup – restores pins to input state (safe default).
        try: GPIO.cleanup()
        except: pass

        # 6. Stop camera pipeline.
        try: picam2.stop()
        except: pass

        # 7. Close any remaining OpenCV windows.
        cv.destroyAllWindows()
        print("[main] Done.")

if __name__ == "__main__":
    # ── START METHOD: forkserver ───────────────────────────────
    # Options: fork / spawn / forkserver.
    #   fork       = copy parent process (fast but duplicates camera/cv handles – unsafe)
    #   spawn      = fresh Python interpreter (slow startup, safest)
    #   forkserver = one-off spawn of a helper, then lightweight forks (balanced)
    # forkserver avoids the OpenCV/Picamera2 handle-duplication issues that
    # plague plain `fork` while still starting quickly.
    mp.set_start_method("forkserver")
    main()
