from picamera2 import Picamera2
import cv2
import numpy as np
import os

# Make sure the templates folder exists!
if not os.path.exists("templates"):
    os.makedirs("templates")

# Ask for the shape name BEFORE starting the camera
print("=======================================")
shape_name = input("What shape are you capturing? (e.g., star, packman, plus): ")
save_filename = f"templates/{shape_name.lower().strip()}.png"
print(f"Great! This will be saved as {save_filename}")
print("=======================================")

# Start Raspberry Pi Camera
camera = Picamera2()
config = camera.create_preview_configuration(main={"size": (640, 480)})
camera.configure(config)
camera.start()

print("1. Put your shape exactly inside the GREEN BOX.")
print("2. Look at the 'Threshold Preview' window to make sure the shape is solid white.")
print("3. Press 's' to capture and save.")
print("4. Press ESC or 'q' to quit.")

try:
    while True:
        # Capture frame
        frame = camera.capture_array()
        
        # Define a 350x350 pixel box in the center of the screen
        h, w = frame.shape[:2]
        box_size = 350
        x1 = int((w - box_size) /2)
        y1 = int((h - box_size) /2)
        x2 = x1 + box_size
        y2 = y1 + box_size

        # Crop the frame to ONLY look inside the box
        roi = frame[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Generate a live preview of the Otsu threshold
        blurred_roi = cv2.GaussianBlur(gray_roi, (11, 11), 0)
        _, thresh_roi = cv2.threshold(blurred_roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        # Draw the green targeting box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Show windows
        cv2.imshow("Template Capture - Put shape in box", frame)
        cv2.imshow("Threshold Preview (What the Pi sees)", thresh_roi)

        key = cv2.waitKey(1) & 0xFF

        # Press S to save image
        if key == ord('s'):
            cv2.imwrite(save_filename, gray_roi)
            print(f"? Template perfectly cropped and saved as {save_filename}!")
            break

        # ESC or 'q' to exit
        if key == 27 or key == ord('q'):
            print("Cancelled.")
            break
finally:
    camera.stop()
    cv2.destroyAllWindows()