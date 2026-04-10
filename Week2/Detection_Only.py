import os
import sys
import time
import cv2

from symbol_detector import TemplateLibrary, SymbolDetector

TEMPLATES_DIR = "templates"
DEBUG_DIR = "debug"

SHOW_PREVIEW = True
SHOW_EDGES = True
ROTATE_180 = False
USE_ROI = True

FRAME_SIZE = (640, 480)
ROI_MARGIN = 0.20  # 20% margin on each side


def open_picamera2():
    try:
        from picamera2 import Picamera2
    except Exception:
        return None, None

    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": FRAME_SIZE, "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)
    return picam2, "picamera2"


def open_gstreamer():
    pipeline = (
        "libcamerasrc ! video/x-raw, width=640, height=480, framerate=30/1 ! "
        "videoconvert ! appsink"
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        return None, None
    return cap, "gstreamer"


def main():
    if not os.path.isdir(TEMPLATES_DIR):
        print(f"CRITICAL ERROR: Templates folder not found: {TEMPLATES_DIR}")
        sys.exit(1)

    try:
        lib = TemplateLibrary(TEMPLATES_DIR).load()
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load templates: {e}")
        sys.exit(1)

    detector = SymbolDetector(
        lib,
        min_area=1800,
        match_thresh=0.40,
        cross_shape_thresh=0.80,
        fp_min_good=10,
        fp_ratio=0.75,
    )

    cam, cam_type = open_picamera2()
    if cam is None:
        cam, cam_type = open_gstreamer()

    if cam is None:
        print("FAILED TO START CAMERA.")
        print("If you have a Pi Camera, verify with `libcamera-hello`.")
        sys.exit(1)

    print(f"Loaded {len(lib.shape_templates) + len(lib.feature_templates)} templates.")
    print("Running detection... press 'q' to quit.")
    print("Press 'd' to dump debug images.")

    try:
        while True:
            if cam_type == "picamera2":
                frame_rgb = cam.capture_array()
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            else:
                ret, frame_bgr = cam.read()
                if not ret:
                    print("Empty frame received.")
                    break

            if ROTATE_180:
                frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_180)

            display = frame_bgr.copy()
            h, w = display.shape[:2]

            roi = None
            if USE_ROI:
                x1 = int(ROI_MARGIN * w)
                y1 = int(ROI_MARGIN * h)
                x2 = int((1.0 - ROI_MARGIN) * w)
                y2 = int((1.0 - ROI_MARGIN) * h)
                roi = (x1, y1, x2, y2)
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            results, edges = detector.detect(frame_bgr, roi=roi)

            if results:
                top = results[0]
                label = f'{top["label"]} ({top["score"]:.3f})'
                cv2.putText(display, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            for r in results[:4]:
                x, y, w_box, h_box = r["bbox"]
                label = f'{r["label"]} ({r["score"]:.3f})'
                cv2.rectangle(display, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)
                cv2.putText(display, label, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if SHOW_PREVIEW:
                cv2.imshow("Shapes & Symbols Detection", display)
                if SHOW_EDGES and edges is not None:
                    cv2.imshow("Edges", edges)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("d"):
                    os.makedirs(DEBUG_DIR, exist_ok=True)
                    ts = int(time.time() * 1000)
                    dbg_results = detector.detect(frame_bgr, roi=roi, debug=True)
                    _, dbg_edges, dbg_items = dbg_results
                    cv2.imwrite(os.path.join(DEBUG_DIR, f"{ts}_frame.jpg"), frame_bgr)
                    if dbg_edges is not None:
                        cv2.imwrite(os.path.join(DEBUG_DIR, f"{ts}_edges.png"), dbg_edges)
                    for i, item in enumerate(dbg_items[:6]):
                        cand = item.get("cand_bin")
                        templ = item.get("best_bin_img")
                        if cand is not None:
                            cv2.imwrite(os.path.join(DEBUG_DIR, f"{ts}_cand_{i}.png"), cand)
                        if templ is not None:
                            cv2.imwrite(os.path.join(DEBUG_DIR, f"{ts}_templ_{i}.png"), templ)
                    print(f"Saved debug images to {DEBUG_DIR}/")

    except KeyboardInterrupt:
        print("Stopping...")

    if cam_type == "picamera2":
        cam.stop()
    else:
        cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
