import time
import cv2
from picamera2 import Picamera2
from symbol_detector import SymbolDetector

detector = SymbolDetector(min_area=800)

picam2 = Picamera2()
picam2.configure(
    picam2.create_video_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
)
picam2.start()

time.sleep(2)

print("Running detection (Ctrl+C to stop)...")

last_detected = None
detection_cooldown = 0
COOLDOWN_TIME = 2.0  # seconds before detecting same object again

try:
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        results = detector.detect(frame)

        if results:
            for obj in results:
                label = obj["shape"]
                if obj["direction"]:
                    label += " " + obj["direction"]

                # Only print if different from last detection or cooldown expired
                current_time = time.time()
                if label != last_detected or current_time > detection_cooldown:
                    print("Detected:", label)
                    last_detected = label
                    detection_cooldown = current_time + COOLDOWN_TIME
        
        time.sleep(0.2)  # Slow down detection rate

except KeyboardInterrupt:
    print("Stopping...")

picam2.stop()