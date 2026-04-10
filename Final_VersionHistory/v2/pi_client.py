"""TCP client module — runs on Raspberry Pi.

Captures a frame (or accepts one), compresses it as JPEG, sends it to the
MacBook server, waits for a command string response, and returns it.
"""

import socket
import struct
import time

import cv2
import numpy as np

DEFAULT_SERVER_IP = "10.134.9.249"   # ← change to your MacBook's IP
DEFAULT_PORT = 5001
JPEG_QUALITY = 80
MAX_RETRIES = 1        # one shot per control cycle; retry next frame instead
RETRY_DELAY = 0.05     # retained for symmetry with send loop
SOCKET_TIMEOUT = 0.45  # keep steering responsive even if the Mac stalls
RECONNECT_COOLDOWN = 0.75


class PiClient:
    """Persistent TCP client with automatic reconnection."""

    def __init__(self, server_ip=DEFAULT_SERVER_IP, port=DEFAULT_PORT):
        self.server_ip = server_ip
        self.port = port
        self._sock: socket.socket | None = None
        self._next_connect_time = 0.0

    # ------------------------------------------------------------------
    def _connect(self):
        """Establish (or re-establish) the TCP connection."""
        now = time.monotonic()
        if now < self._next_connect_time:
            raise ConnectionError(
                f"Reconnect cooldown active for "
                f"{self._next_connect_time - now:.2f}s"
            )

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if self._sock:
                    try:
                        self._sock.close()
                    except OSError:
                        pass
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._sock.settimeout(SOCKET_TIMEOUT)
                self._sock.connect((self.server_ip, self.port))
                self._next_connect_time = 0.0
                print(f"[PiClient] Connected to {self.server_ip}:{self.port}")
                return
            except OSError as exc:
                self._next_connect_time = time.monotonic() + RECONNECT_COOLDOWN
                print(f"[PiClient] Connection attempt {attempt}/{MAX_RETRIES} "
                      f"failed: {exc}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
        raise ConnectionError(
            f"Could not connect to {self.server_ip}:{self.port} "
            f"after {MAX_RETRIES} attempts"
        )

    # ------------------------------------------------------------------
    def send_frame(self, frame: np.ndarray) -> str:
        """Compress *frame* as JPEG, send it, and return the server's
        command string (e.g. ``"LEFT"``, ``"STOP"``).

        Protocol
        --------
        Client → Server : 4-byte big-endian length + JPEG bytes
        Server → Client : 4-byte big-endian length + UTF-8 command string
        """
        # Encode
        ok, buf = cv2.imencode(
            ".jpg", frame,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
        )
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        data = buf.tobytes()

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if self._sock is None:
                    self._connect()

                # Send length-prefixed JPEG
                self._sock.sendall(struct.pack(">I", len(data)))
                self._sock.sendall(data)

                # Receive length-prefixed response
                raw_len = self._recvall(4)
                resp_len = struct.unpack(">I", raw_len)[0]
                resp_data = self._recvall(resp_len)
                return resp_data.decode("utf-8")

            except (OSError, ConnectionError, struct.error) as exc:
                print(f"[PiClient] Send/recv failed (attempt "
                      f"{attempt}/{MAX_RETRIES}): {exc}")
                if self._sock:
                    try:
                        self._sock.close()
                    except OSError:
                        pass
                self._sock = None          # force reconnect
                self._next_connect_time = time.monotonic() + RECONNECT_COOLDOWN
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)

        raise ConnectionError("Failed to send frame after retries")

    # ------------------------------------------------------------------
    def _recvall(self, n: int) -> bytes:
        """Receive exactly *n* bytes from the socket."""
        chunks = []
        remaining = n
        while remaining > 0:
            chunk = self._sock.recv(remaining)
            if not chunk:
                raise ConnectionError("Connection closed by server")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    # ------------------------------------------------------------------
    def close(self):
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None


# ======================================================================
# Quick standalone test
# ======================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PiClient test")
    parser.add_argument("--ip", default=DEFAULT_SERVER_IP,
                        help="MacBook server IP")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    client = PiClient(args.ip, args.port)

    # Create a dummy test frame (640×480 black image with text)
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_frame, "TEST FRAME", (160, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    try:
        response = client.send_frame(test_frame)
        print(f"[PiClient] Server responded: {response}")
    finally:
        client.close()
