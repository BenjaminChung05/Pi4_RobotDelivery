"""TCP server module — runs on MacBook.

Listens on a port, receives length-prefixed JPEG frames from a Raspberry Pi
client, passes each frame to a pluggable detection callback, and sends back
the resulting command string.
"""

import socket
import struct
import threading
import traceback

import cv2
import numpy as np

DEFAULT_PORT = 5001


# ======================================================================
# Stub detection function — replaced by the real detector in production
# ======================================================================
def _stub_detect(frame: np.ndarray) -> str:
    """Placeholder detection — always returns FORWARD."""
    print(f"[stub_detect] Received frame {frame.shape}")
    return "FORWARD"


# ======================================================================
# Server
# ======================================================================
class MacServer:
    """Single-threaded TCP server that handles one Pi client at a time,
    with reconnection support (the client can drop and reconnect).

    Parameters
    ----------
    detect_fn : callable(np.ndarray) -> str
        A function that takes a BGR frame and returns a command string.
    port : int
        Port to listen on.
    """

    def __init__(self, detect_fn=None, port=DEFAULT_PORT):
        self.detect_fn = detect_fn or _stub_detect
        self.port = port
        self._running = False
        self._server_sock: socket.socket | None = None

    # ------------------------------------------------------------------
    def start(self):
        """Start listening and serving (blocking)."""
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind(("0.0.0.0", self.port))
        self._server_sock.listen(1)
        self._running = True
        print(f"[MacServer] Listening on 0.0.0.0:{self.port}")

        while self._running:
            try:
                print("[MacServer] Waiting for Pi to connect …")
                conn, addr = self._server_sock.accept()
                print(f"[MacServer] Connection from {addr}")
                self._handle_client(conn)
            except OSError:
                if self._running:
                    traceback.print_exc()

    # ------------------------------------------------------------------
    def _handle_client(self, conn: socket.socket):
        """Serve one connected client until it disconnects."""
        conn.settimeout(30)
        try:
            while self._running:
                # Receive 4-byte length prefix
                raw_len = self._recvall(conn, 4)
                if raw_len is None:
                    break
                msg_len = struct.unpack(">I", raw_len)[0]
                if msg_len > 10_000_000:       # sanity: reject >10 MB
                    print("[MacServer] Frame too large, dropping connection")
                    break

                # Receive JPEG data
                jpg_data = self._recvall(conn, msg_len)
                if jpg_data is None:
                    break

                # Decode
                frame = cv2.imdecode(
                    np.frombuffer(jpg_data, dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if frame is None:
                    print("[MacServer] Failed to decode JPEG, skipping")
                    continue

                # Detect
                command = self.detect_fn(frame)

                # Send response (length-prefixed UTF-8 string)
                resp = command.encode("utf-8")
                conn.sendall(struct.pack(">I", len(resp)))
                conn.sendall(resp)

        except (OSError, ConnectionError) as exc:
            print(f"[MacServer] Client disconnected: {exc}")
        finally:
            conn.close()
            print("[MacServer] Connection closed, waiting for reconnect …")

    # ------------------------------------------------------------------
    @staticmethod
    def _recvall(sock: socket.socket, n: int):
        """Receive exactly *n* bytes.  Returns ``None`` on clean disconnect."""
        chunks = []
        remaining = n
        while remaining > 0:
            chunk = sock.recv(remaining)
            if not chunk:
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    # ------------------------------------------------------------------
    def stop(self):
        self._running = False
        if self._server_sock:
            try:
                self._server_sock.close()
            except OSError:
                pass


# ======================================================================
# Quick standalone test
# ======================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MacServer (stub)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    server = MacServer(detect_fn=_stub_detect, port=args.port)
    try:
        server.start()
    except KeyboardInterrupt:
        print("\n[MacServer] Shutting down")
        server.stop()
