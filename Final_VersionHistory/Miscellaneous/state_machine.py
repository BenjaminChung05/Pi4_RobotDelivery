from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RobotState(str, Enum):
    LINE_FOLLOW = "LINE_FOLLOW"
    EXECUTE_ACTION = "EXECUTE_ACTION"
    RECOVER = "RECOVER"


@dataclass
class StateMachineConfig:
    symbol_cooldown_seconds: float = 2.0
    action_duration_seconds: float = 0.8
    recover_duration_seconds: float = 0.5


class RobotStateMachine:
    """
    Small FSM for symbol-driven actions.

    Line following still runs every loop. The FSM only decides whether the
    nominal steering command should be overridden for a short period.
    """

    def __init__(self, config: StateMachineConfig | None = None) -> None:
        self.config = config or StateMachineConfig()
        self.state = RobotState.LINE_FOLLOW
        self.current_symbol: str | None = None
        self.last_detected_symbol: str | None = None
        self.state_until = 0.0
        self.cooldown_until = 0.0

    def can_accept_symbol(self, now: float) -> bool:
        return self.state == RobotState.LINE_FOLLOW and now >= self.cooldown_until

    def handle_symbol(self, label: str, now: float, duration_seconds: float | None = None) -> None:
        self.current_symbol = label
        self.last_detected_symbol = label
        self.state = RobotState.EXECUTE_ACTION
        action_duration = duration_seconds if duration_seconds is not None else self.config.action_duration_seconds
        self.state_until = now + action_duration
        self.cooldown_until = now + self.config.symbol_cooldown_seconds

    def mark_output_symbol(self, label: str, now: float) -> None:
        self.current_symbol = None
        self.last_detected_symbol = label
        self.cooldown_until = now + self.config.symbol_cooldown_seconds

    def update(self, now: float) -> None:
        if self.state == RobotState.EXECUTE_ACTION and now >= self.state_until:
            self.state = RobotState.RECOVER
            self.state_until = now + self.config.recover_duration_seconds
            return

        if self.state == RobotState.RECOVER and now >= self.state_until:
            self.state = RobotState.LINE_FOLLOW
            self.current_symbol = None
            self.state_until = 0.0
