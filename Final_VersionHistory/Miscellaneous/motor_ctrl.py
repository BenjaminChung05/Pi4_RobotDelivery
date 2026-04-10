from __future__ import annotations

from dataclasses import dataclass

try:
    import RPi.GPIO as GPIO
except ImportError:
    GPIO = None


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass
class MotorConfig:
    ena: int = 12
    in1: int = 17
    in2: int = 18
    enb: int = 13
    in3: int = 22
    in4: int = 23
    pwm_frequency: int = 1000
    left_inverted: bool = False
    right_inverted: bool = False
    min_drive_pwm: float = 35.0


class MotorController:
    """
    GPIO/PWM motor driver for a two-motor differential robot.

    If RPi.GPIO is unavailable the controller falls back to a dry-run mode so
    the rest of the pipeline can still be tested on a laptop.
    """

    def __init__(self, config: MotorConfig | None = None, dry_run: bool = False) -> None:
        self.config = config or MotorConfig()
        self.dry_run = dry_run or GPIO is None
        self._pwm_left = None
        self._pwm_right = None
        self.last_command = (0.0, 0.0)

    @property
    def available(self) -> bool:
        return not self.dry_run

    def setup(self) -> None:
        if self.dry_run:
            return

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        for pin in (
            self.config.ena,
            self.config.in1,
            self.config.in2,
            self.config.enb,
            self.config.in3,
            self.config.in4,
        ):
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)

        self._pwm_left = GPIO.PWM(self.config.ena, self.config.pwm_frequency)
        self._pwm_right = GPIO.PWM(self.config.enb, self.config.pwm_frequency)
        self._pwm_left.start(0)
        self._pwm_right.start(0)

    def set_motor(self, left_pwm: float, right_pwm: float) -> None:
        left_pwm = _clamp(float(left_pwm), -100.0, 100.0)
        right_pwm = _clamp(float(right_pwm), -100.0, 100.0)

        if self.config.left_inverted:
            left_pwm = -left_pwm
        if self.config.right_inverted:
            right_pwm = -right_pwm

        if 0.0 < abs(left_pwm) < self.config.min_drive_pwm:
            left_pwm = self.config.min_drive_pwm if left_pwm > 0 else -self.config.min_drive_pwm
        if 0.0 < abs(right_pwm) < self.config.min_drive_pwm:
            right_pwm = self.config.min_drive_pwm if right_pwm > 0 else -self.config.min_drive_pwm

        self.last_command = (left_pwm, right_pwm)
        if self.dry_run:
            return

        self._apply_motor(
            pwm_value=left_pwm,
            forward_pin=self.config.in1,
            reverse_pin=self.config.in2,
            pwm=self._pwm_left,
        )
        self._apply_motor(
            pwm_value=right_pwm,
            forward_pin=self.config.in3,
            reverse_pin=self.config.in4,
            pwm=self._pwm_right,
        )

    def set_pwm(self, left_pwm: float, right_pwm: float) -> None:
        self.set_motor(left_pwm, right_pwm)

    def _apply_motor(self, pwm_value: float, forward_pin: int, reverse_pin: int, pwm) -> None:
        if pwm_value >= 0:
            GPIO.output(forward_pin, GPIO.LOW)
            GPIO.output(reverse_pin, GPIO.HIGH)
        else:
            GPIO.output(forward_pin, GPIO.HIGH)
            GPIO.output(reverse_pin, GPIO.LOW)
        pwm.ChangeDutyCycle(abs(pwm_value))

    def stop(self) -> None:
        self.set_motor(0.0, 0.0)
        if self.dry_run:
            return
        GPIO.output(self.config.in1, GPIO.LOW)
        GPIO.output(self.config.in2, GPIO.LOW)
        GPIO.output(self.config.in3, GPIO.LOW)
        GPIO.output(self.config.in4, GPIO.LOW)

    def cleanup(self) -> None:
        self.stop()
        if self.dry_run:
            return
        if self._pwm_left is not None:
            self._pwm_left.stop()
        if self._pwm_right is not None:
            self._pwm_right.stop()
        GPIO.cleanup()
