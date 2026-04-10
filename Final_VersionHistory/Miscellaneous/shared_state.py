from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock


@dataclass
class SharedState:
    """
    Sequential code updates this directly today.

    A threaded version can keep the same structure and guard access with
    `with shared.lock:` in camera, perception, and control worker threads.
    """

    lock: Lock = field(default_factory=Lock)
    frame_index: int = 0
    fps: float = 0.0
    robot_state: str = "LINE_FOLLOW"
    last_symbol: str | None = None
    last_symbol_score: float = 0.0
    last_shape: str | None = None
    last_shape_score: float = 0.0
    line_error_px: float = 0.0
    line_found: bool = False
    left_pwm: float = 0.0
    right_pwm: float = 0.0
    line_ms: float = 0.0
    symbol_ms: float = 0.0
    control_ms: float = 0.0
    loop_ms: float = 0.0
