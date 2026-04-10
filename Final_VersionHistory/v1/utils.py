from __future__ import annotations

import time


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class FpsCounter:
    def __init__(self, report_period_seconds: float = 1.0) -> None:
        self.report_period_seconds = report_period_seconds
        self.last_report = time.perf_counter()
        self.frames = 0
        self.current_fps = 0.0

    def update(self) -> float:
        self.frames += 1
        now = time.perf_counter()
        elapsed = now - self.last_report
        if elapsed >= self.report_period_seconds:
            self.current_fps = self.frames / elapsed
            self.frames = 0
            self.last_report = now
        return self.current_fps
