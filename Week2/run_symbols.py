# run_symbol.py
import time
import cv2
from picamera2 import Picamera2
from collections import Counter, deque

from symbol_detector import TemplateLibrary, SymbolDetector

TEMPLATES_DIR = "templates"

SHOW_PREVIEW = True
ROTATE_180 = False
USE_ROI = True

WINDOW = 12
REQUIRE = 8


class StableLabel:
    def __init__(self, window=12, require=8):
        self.buf = deque(maxlen=window)
        self.window = window
        self.require = require

    def update(self, label):
        self.buf.append(label)
        if len(self.buf) < self.window:
            return None
        counts = Counter([x for x in self.buf if x is not None])
        if not counts:
            return None
        best, n = counts.most_common(1)[0]
        return best if n >= self.require else None


def main():
    lib = TemplateLibrary(TEMPLATES_DIR, swap_octagon_hazard=True).load()

    detector = SymbolDetector(
        lib,
        min_area=1800,
        match_thresh=0.42,
        cross_shape_thresh=0.85,
        fp_min_good=10,
        fp_ratio=0.75,
    )

    stable = StableLabel(window=WINDOW, require=REQUIRE)

    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"}))
    picam2.start()

    time.sleep(1.5)

    roi = None
    last_print = 0
    last_label = None

    print("Running recognition... press 'q' to quit preview, Ctrl+C to stop.")

    try:
        while True:
            frame_rgb = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            if ROTATE_180:
                frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_180)

            display = frame_bgr.copy()
            h, w = frame_bgr.shape[:2]

            # Crosshair
            cv2.line(display, (w // 2 - 20, h // 2), (w // 2 + 20, h // 2), (0, 255, 0), 2)
            cv2.line(display, (w // 2, h // 2 - 20), (w // 2, h // 2 + 20), (0, 255, 0), 2)

            # ROI
            if USE_ROI and roi is None:
                roi = (int(0.2 * w), int(0.2 * h), int(0.8 * w), int(0.8 * h))

            if USE_ROI and roi is not None:
                x1, y1, x2, y2 = roi
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            results, edges = detector.detect(frame_bgr, roi=roi)

            raw_label = results[0]["label"] if results else None
            stable_label = stable.update(raw_label)

            now = time.time()
            if stable_label and (now - last_print) > 1.0:
                if stable_label != last_label:
                    print("STABLE DETECTED:", stable_label)
                    last_label = stable_label
                last_print = now

            for r in results[:3]:
                x, y, w_box, h_box = r["bbox"]
                label = f'{r["label"]} ({r["score"]:.3f})'
                cv2.rectangle(display, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)
                cv2.putText(display, label, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if SHOW_PREVIEW:
                cv2.imshow("Symbol Detection", display)
                cv2.imshow("Edges", edges)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("Stopping...")

    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    
