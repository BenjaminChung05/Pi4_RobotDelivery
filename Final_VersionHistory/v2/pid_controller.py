"""PID controller for centroid-based robot line following."""

import csv
import time


class PIDController:
    """PID controller that converts horizontal pixel offset into a differential
    motor speed adjustment value.

    Parameters
    ----------
    kp, ki, kd : float
        Proportional, integral, and derivative gains.
    max_output : float
        Clamp the output to [-max_output, max_output].
    log_path : str or None
        If set, every ``update`` call writes a row to this CSV file
        (timestamp, error, P, I, D, output).
    """

    def __init__(self, kp=0.4, ki=0.0, kd=0.1, max_output=100.0,
                 log_path=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output

        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None

        # Optional CSV tuning log
        self._log_file = None
        self._csv_writer = None
        if log_path:
            self._log_file = open(log_path, "w", newline="")
            self._csv_writer = csv.writer(self._log_file)
            self._csv_writer.writerow(
                ["timestamp", "error", "P", "I", "D", "output"]
            )

    # ------------------------------------------------------------------
    def update(self, error: float) -> float:
        """Compute a new output given the current error (pixels from centre).

        Returns a differential speed value in [-max_output, max_output].
        Positive  -> steer right,  Negative -> steer left.
        """
        now = time.monotonic()
        dt = (now - self._prev_time) if self._prev_time is not None else 0.0
        self._prev_time = now

        p_term = self.kp * error

        self._integral += error * dt
        i_term = self.ki * self._integral

        d_term = self.kd * ((error - self._prev_error) / dt) if dt > 0 else 0.0
        self._prev_error = error

        output = p_term + i_term + d_term
        output = max(-self.max_output, min(self.max_output, output))

        if self._csv_writer:
            self._csv_writer.writerow(
                [f"{now:.4f}", f"{error:.2f}", f"{p_term:.4f}",
                 f"{i_term:.4f}", f"{d_term:.4f}", f"{output:.4f}"]
            )
            self._log_file.flush()

        return output

    # ------------------------------------------------------------------
    def reset_integral(self):
        """Zero the integral accumulator (call at intersections)."""
        self._integral = 0.0

    def reset(self):
        """Full reset — integral, derivative memory, and timing."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None

    # ------------------------------------------------------------------
    def close(self):
        """Flush and close the CSV log file if one is open."""
        if self._log_file:
            self._log_file.close()
            self._log_file = None
            self._csv_writer = None
