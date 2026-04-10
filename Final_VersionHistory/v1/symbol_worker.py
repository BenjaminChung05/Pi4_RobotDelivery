"""Background threaded symbol detection for the robot controller.

Runs fast_filter -> classify / probe in a daemon thread so the main control
loop can do uninterrupted line following.

BossLevelV extensibility
------------------------
Register callbacks via ``worker.on_symbol("FINGERPRINT", fn)`` to trigger
additional processing (face recognition, shape analysis, etc.) when a
specific symbol is detected.  Callbacks execute in the worker thread.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from symbol_detector import (
    SymbolCandidate,
    SymbolResult,
    TFLiteSymbolDetector,
)

logger = logging.getLogger(__name__)


@dataclass
class SymbolWorkerConfig:
    """Tuning knobs for the background detection thread."""

    # Master switch — set False to fall back to synchronous detection.
    threaded: bool = True

    # Minimum gap between classify attempts (seconds).  0 = no limit.
    min_detect_interval_s: float = 0.0

    # Probe (expensive whole-image scan when fast_filter rejects).
    enable_probe: bool = True
    min_probe_interval_s: float = 0.3

    # Thread idle sleep when waiting for a fresh frame.
    idle_sleep_s: float = 0.005


class SymbolWorker:
    """Background symbol detection worker.

    Usage::

        worker = SymbolWorker(detector, config)
        worker.start()          # spawns daemon thread

        # In main loop:
        worker.post_frame(frame)                     # non-blocking
        result, cand, is_new = worker.get_result()   # non-blocking

        worker.pause()   # suppress detection during hard turns
        worker.resume()

        worker.stop()    # clean shutdown
    """

    def __init__(
        self,
        detector: TFLiteSymbolDetector,
        config: SymbolWorkerConfig | None = None,
    ) -> None:
        self._detector = detector
        self.config = config or SymbolWorkerConfig()

        # --- Shared state (protected by _lock) ---
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._frame_seq: int = 0
        self._processed_seq: int = 0

        # Latest result published by the worker
        self._result: SymbolResult | None = None
        self._candidate: SymbolCandidate | None = None
        self._result_seq: int = 0
        # Sequence last consumed by the main loop
        self._consumed_seq: int = 0

        # --- Control ---
        self._active = threading.Event()
        self._active.set()  # start unpaused
        self._stop_flag = threading.Event()
        self._thread: threading.Thread | None = None

        # --- BossLevel extension hooks ---
        self._hooks: dict[str, list[Callable[[SymbolResult], None]]] = {}

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Spawn the background detection thread (no-op if threaded=False)."""
        if not self.config.threaded:
            return
        t = threading.Thread(target=self._run, name="SymbolWorker", daemon=True)
        self._thread = t
        t.start()
        logger.info("SymbolWorker thread started")

    def stop(self) -> None:
        """Signal the thread to exit and wait for it."""
        self._stop_flag.set()
        self._active.set()  # unblock if paused
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        logger.info("SymbolWorker stopped")

    def pause(self) -> None:
        """Pause detection (e.g., during hard turns)."""
        self._active.clear()

    def resume(self) -> None:
        """Resume detection."""
        self._active.set()

    @property
    def paused(self) -> bool:
        return not self._active.is_set()

    # ------------------------------------------------------------------ #
    # Frame input  (called from main loop — non-blocking)
    # ------------------------------------------------------------------ #

    def post_frame(self, frame: np.ndarray) -> None:
        """Post the latest camera frame.  Overwrites any unprocessed frame."""
        with self._lock:
            self._frame = frame
            self._frame_seq += 1

    # ------------------------------------------------------------------ #
    # Result output  (called from main loop — non-blocking)
    # ------------------------------------------------------------------ #

    def get_result(self) -> tuple[SymbolResult | None, SymbolCandidate | None, bool]:
        """Return ``(result, candidate, is_new)``.

        ``is_new`` is ``True`` only the first time a particular result is
        returned, so the main loop does not double-count confirmations.
        """
        with self._lock:
            is_new = self._result_seq > self._consumed_seq
            if is_new:
                self._consumed_seq = self._result_seq
            return self._result, self._candidate, is_new

    # ------------------------------------------------------------------ #
    # Synchronous fallback  (when threaded=False)
    # ------------------------------------------------------------------ #

    def detect_sync(
        self, frame: np.ndarray,
    ) -> tuple[SymbolResult, SymbolCandidate]:
        """Run detection synchronously.  Used when threading is disabled."""
        candidate = self._detector.fast_filter(frame)
        if candidate.found:
            try:
                result = self._detector.classify(frame, candidate)
            except Exception:
                result = SymbolResult(
                    enabled=self._detector.enabled,
                    accepted=False,
                    reason="Classify error",
                )
        else:
            if self.config.enable_probe:
                result = self._detector.probe_symbol(frame)
            else:
                result = SymbolResult(
                    enabled=self._detector.enabled,
                    accepted=False,
                    reason="No candidate",
                )
            candidate = SymbolCandidate(found=False)
        return result, candidate

    # ------------------------------------------------------------------ #
    # BossLevel callback hooks
    # ------------------------------------------------------------------ #

    def on_symbol(
        self, label: str, callback: Callable[[SymbolResult], None],
    ) -> None:
        """Register *callback* to fire when *label* is detected.

        Callbacks run in the worker thread (threaded mode) or inline
        (synchronous mode).  Keep them fast or offload heavy work to
        another thread/process.
        """
        self._hooks.setdefault(label, []).append(callback)

    def _fire_hooks(self, result: SymbolResult) -> None:
        label = result.action_label or result.label
        if label and label in self._hooks:
            for cb in self._hooks[label]:
                try:
                    cb(result)
                except Exception as exc:
                    logger.warning("Hook error (%s): %s", label, exc)

    # ------------------------------------------------------------------ #
    # Worker thread body
    # ------------------------------------------------------------------ #

    def _run(self) -> None:  # noqa: C901
        last_detect = 0.0
        last_probe = 0.0
        cfg = self.config

        while not self._stop_flag.is_set():
            # Respect pause
            self._active.wait(timeout=0.1)
            if self._stop_flag.is_set():
                break

            # Grab latest frame (only if newer than what we last processed)
            with self._lock:
                if self._frame_seq == self._processed_seq:
                    no_frame = True
                else:
                    frame = self._frame.copy()
                    seq = self._frame_seq
                    self._processed_seq = seq
                    no_frame = False

            if no_frame:
                time.sleep(cfg.idle_sleep_s)
                continue

            now = time.perf_counter()

            # Rate-limit
            if cfg.min_detect_interval_s > 0 and (now - last_detect) < cfg.min_detect_interval_s:
                time.sleep(cfg.idle_sleep_s)
                continue

            # --- Detection ---
            try:
                cand = self._detector.fast_filter(frame)
                result: SymbolResult | None = None

                if cand.found:
                    result = self._detector.classify(frame, cand)
                    last_detect = time.perf_counter()
                elif cfg.enable_probe and (now - last_probe) >= cfg.min_probe_interval_s:
                    result = self._detector.probe_symbol(frame)
                    last_probe = time.perf_counter()
                    cand = SymbolCandidate(found=False)

                if result is not None:
                    with self._lock:
                        self._result = result
                        self._candidate = cand
                        self._result_seq = seq

                    # Fire BossLevel hooks
                    if result.accepted and result.label:
                        self._fire_hooks(result)

            except Exception as exc:
                logger.debug("SymbolWorker error: %s", exc)

            time.sleep(cfg.idle_sleep_s)
