import cv2
import os
import numpy as np
from collections import Counter, deque

# Make sure you have a folder named 'templates' in the same directory
TEMPLATES_DIR = "templates"

class TemplateLibrary:
    def __init__(self, directory):
        self.directory = directory
        self.templates = {}

    def load(self):
        """Loads all .jpg and .png images from the templates folder into memory."""
        if not os.path.exists(self.directory):
            print(f"Warning: Directory '{self.directory}' not found!")
            print("Please create a folder named 'templates' and add your symbol images.")
            return self

        for filename in os.listdir(self.directory):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # The label name becomes the filename without the extension (e.g., "Recycle.jpg" -> "Recycle")
                label = os.path.splitext(filename)[0]
                path = os.path.join(self.directory, filename)
                
                # Load as grayscale for template matching
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    self.templates[label] = img
                else:
                    print(f"Failed to load template: {filename}")
                    
        print(f"Loaded {len(self.templates)} symbol templates from '{self.directory}'.")
        return self

class SymbolDetector:
    def __init__(self, library, min_area=1800, match_thresh=0.42):
        self.library = library
        self.min_area = min_area
        self.match_thresh = match_thresh

    def detect(self, frame):
        """Scans the camera frame for matches against the loaded templates."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = []
        
        for label, template in self.library.templates.items():
            h, w = template.shape
            
            # Prevent crashing if the camera frame is smaller than the template image
            if h > gray.shape[0] or w > gray.shape[1]:
                continue
                
            # Perform OpenCV Template Matching
            res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            # If the match score is higher than our threshold, it's a valid detection
            if max_val >= self.match_thresh:
                area = w * h
                if area >= self.min_area:
                    results.append({
                        "label": label,
                        "score": max_val,
                        "box": (max_loc[0], max_loc[1], w, h)
                    })
        
        # Sort the results so the most confident match is first
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        
        # The master script expects a tuple: (results, frame)
        return results, frame

class StableLabel:
    def __init__(self, window=12, require=8):
        """
        Prevents ghost readings. A symbol must be seen 'require' times 
        within the last 'window' frames to be considered a stable detection.
        """
        self.window = window
        self.require = require
        self.history = deque(maxlen=window)

    def update(self, label):
        """Adds the latest reading to history and returns the most stable label."""
        self.history.append(label)
        
        if not self.history:
            return None
            
        # Filter out "None" (frames where no symbol was seen)
        valid_labels = [lbl for lbl in self.history if lbl is not None]
        
        if not valid_labels:
            return None
            
        # Count the most frequent symbol in recent memory
        counts = Counter(valid_labels)
        most_common_label, count = counts.most_common(1)[0]
        
        if count >= self.require:
            return most_common_label
            
        return None