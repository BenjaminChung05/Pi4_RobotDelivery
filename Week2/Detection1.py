import cv2  # Install opencv-python
import numpy as np

# =============================================================================
# 1. LOAD TEMPLATES (Run this ONCE before your main loop)
# =============================================================================
def load_symbol_templates():
    """
    Loads your saved images, finds their outlines (contours), and stores 
    them in a dictionary for the robot to remember.
    """
    templates = {}
    
    # ---------------------------------------------------------
    # ALL SHAPES & SYMBOLS FROM THE PROJECT PDFS
    # Ensure you have cropped .png files saved in your folder with these exact names!
    # ---------------------------------------------------------
    files_to_load = {
        # Shapes (from Shape_2026.pdf)
        'Star': 'star.png',
        'Diamond': 'diamond.png',
        'Cross': 'cross.png',
        'Trapezoid': 'trapezoid.png',
        'Pacman': 'pacman.png', 
        'Semi-Circle': 'semi_circle.png',
        'Octagon': 'octagon.png',

        # Directional Arrows (from Symbol_2026.pdf)
        'Arrow-Up': 'arrow_up.png',       # Green up arrow
        'Arrow-Right': 'arrow_right.png', # Blue right arrow
        'Arrow-Down': 'arrow_down.png',   # Red down arrow
        'Arrow-Left': 'arrow_left.png',   # Orange left arrow

        # Task-Specific Markers (from Symbol_2026.pdf)
        'Four-Squares': 'four_squares.png',     # The 4 blue squares
        'Recycle': 'recycle.png',               # The green recycle arrows
        'Warning': 'warning_triangle.png',      # The yellow exclamation triangle
        'Push-Button': 'push_button.png'        # The green hand/button symbol
    }

    for name, filename in files_to_load.items():
        # Load image in grayscale
        img = cv2.imread(filename, 0)
        
        if img is None:
            print(f"WARNING: Could not find {filename}. Make sure it is in the same folder!")
            continue
            
        # Threshold the image 
        # (Assuming your templates have white backgrounds and dark shapes)
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Find the contour of your template
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Grab the largest contour (the shape itself) and save it
            largest_cnt = max(contours, key=cv2.contourArea)
            templates[name] = largest_cnt
            print(f"Successfully loaded memory for: {name}")

    return templates