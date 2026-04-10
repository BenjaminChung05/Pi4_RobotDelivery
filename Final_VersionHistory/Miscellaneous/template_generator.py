import cv2
import os
import numpy as np
from picamera2 import Picamera2

# Create templates directory if it doesn't exist
template_dir = "templates"
if not os.path.exists(template_dir):
    os.makedirs(template_dir)

# --- CAMERA SETUP ---
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "BGR888"})
picam2.configure(config)
picam2.start()

print("=====================================")
print("     TEMPLATE GENERATOR STARTED      ")
print("=====================================")
print("1. Place a symbol under the camera.")
print("2. A green box will highlight the grouped shape.")
print("3. Press 's' to save the shape.")
print("4. Press 'q' to quit.")
print("=====================================")

try:
    while True:
        frame = picam2.capture_array()
        frame = cv2.flip(frame, -1) 
        
        # 1. Base Image Processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        
        # Using your perfect 141 block size!
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 141, 15)
        
        # 2. THE GHOST LAYER (Extreme Dilation)
        # We use a massive 25x25 brush to bridge the gaps between disconnected symbol parts
        kernel_large = np.ones((25,25), np.uint8)
        thresh_merged = cv2.dilate(thresh, kernel_large, iterations=2)
        
        # 3. Find contours on the smudged GHOST layer, NOT the detailed layer
        contours, _ = cv2.findContours(thresh_merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        area = 0
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            if area > 1000:
                # Find the bounding box of the massive grouped shape
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "Target Group", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Template Generator", frame)
        # Showing the merged layer so you can see how it groups things!
        cv2.imshow("Ghost Layer (Merged)", thresh_merged) 
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            if len(contours) > 0 and area > 1000:
                print("\n>>> CAPTURED! <<<")
                shape_name = input("Type the name of this symbol (e.g., QR, Recycle) and press Enter: ")
                
                if shape_name.strip() != "":
                    # CRITICAL: We crop from the detailed 'thresh' layer, not the smudged ghost layer!
                    roi = thresh[y:y+h, x:x+w]
                    roi_resized = cv2.resize(roi, (50, 50))
                    
                    filepath = os.path.join(template_dir, f"{shape_name}.jpg")
                    cv2.imwrite(filepath, roi_resized)
                    print(f"Saved successfully as {filepath}!\n")
                else:
                    print("Name cannot be empty. Capture cancelled.\n")
            else:
                print("\nNo distinct shape found to capture.\n")
                
        elif key == ord('q'):
            break

finally:
    picam2.stop()
    picam2.close()
    cv2.destroyAllWindows()
    print("Generator closed.")