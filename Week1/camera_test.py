from picamera2 import Picamera2
import time
import cv2

picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration(main={"size": (640, 480), "format": "RGB888"}))
picam2.start()
time.sleep(1)

frame = picam2.capture_array()  # RGB
frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

cv2.imwrite("test_frame.jpg", frame_bgr)
print("Saved test_frame.jpg")

picam2.stop()