import socket
import struct
import threading
import cv2
import numpy as np

from symbol_detector import TFLiteSymbolDetector, SymbolDetectorConfig


HOST = "0.0.0.0"
PORT = 5001

# labels from your old detector:
# STOP, LEFT, RIGHT, RECYCLE, QR, FINGERPRINT
LABEL_TO_COMMAND = {
    "LEFT": "LEFT",
    "RIGHT": "RIGHT",
    "STOP": "STOP",
    "RECYCLE": "ROTATE_360",
    "QR": "PRINT_QR",
    "FINGERPRINT": "PRINT_FINGERPRINT",
}
DEFAULT_COMMAND = "FORWARD"

detector = TFLiteSymbolDetector(
    SymbolDetectorConfig(
        templates_dir="templates",
        confidence_threshold=0.60,
    )
)

if not detector.enabled:
    raise RuntimeError(f"Detector failed to load: {detector.reason}")


def recv_exact(conn, n):
    data = b""
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def detect_frame(frame_bgr, mode):
    """
    mode:
      A = arrow mode
      S = symbol mode
    """
    candidate = detector.fast_filter(frame_bgr)
    result = None

    if candidate.found:
        result = detector.classify(frame_bgr, candidate)

    if result is None or not result.accepted:
        result = detector.probe_symbol(frame_bgr)

    label = None
    bbox = None
    conf = 0.0

    if result is not None and result.accepted and result.label is not None:
        label = result.label
        bbox = result.bbox
        conf = float(result.confidence)

    if mode == "A":
        if label not in {"LEFT", "RIGHT"}:
            label = None
    else:
        if label not in {"STOP", "RECYCLE", "QR", "FINGERPRINT"}:
            label = None

    command = LABEL_TO_COMMAND.get(label, DEFAULT_COMMAND)
    return command, label, bbox, conf, candidate


def draw_debug(frame, command, label, bbox, conf, mode, candidate):
    disp = frame.copy()

    if candidate is not None and candidate.roi_bbox is not None:
        x, y, w, h = candidate.roi_bbox
        cv2.rectangle(disp, (x, y), (x + w, y + h), (255, 200, 0), 1)

    if candidate is not None and candidate.bbox is not None:
        x, y, w, h = candidate.bbox
        cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 165, 255), 1)

    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 2)

    text1 = f"MODE: {'ARROW' if mode == 'A' else 'SYMBOL'}"
    text2 = f"LABEL: {label if label else 'NONE'}  CONF: {conf:.2f}"
    text3 = f"CMD: {command}"

    cv2.putText(disp, text1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(disp, text2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(disp, text3, (10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Laptop Detector", disp)
    cv2.waitKey(1)


def handle_client(conn, addr):
    print(f"[SERVER] Connected: {addr}")
    try:
        while True:
            header = recv_exact(conn, 5)
            if header is None:
                break

            mode = header[:1].decode("utf-8", errors="ignore")
            length = struct.unpack(">I", header[1:])[0]

            jpg_bytes = recv_exact(conn, length)
            if jpg_bytes is None:
                break

            arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                reply = DEFAULT_COMMAND.encode("utf-8")
                conn.sendall(struct.pack(">I", len(reply)) + reply)
                continue

            command, label, bbox, conf, candidate = detect_frame(frame, mode)
            draw_debug(frame, command, label, bbox, conf, mode, candidate)

            reply = command.encode("utf-8")
            conn.sendall(struct.pack(">I", len(reply)) + reply)

    except Exception as e:
        print(f"[SERVER] Client error: {e}")
    finally:
        conn.close()
        print(f"[SERVER] Disconnected: {addr}")


def main():
    print(f"[SERVER] Listening on {HOST}:{PORT}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)

        while True:
            conn, addr = s.accept()
            t = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
            t.start()


if __name__ == "__main__":
    main()