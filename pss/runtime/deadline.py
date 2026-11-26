"""Deadline guard for anytime runtime calls.

A synchronous Python call cannot be preempted.  This guard runs the call on a
worker thread, bounds the *caller's* wait, and returns a degraded result when
the deadline is missed.  The worker is joined before the next call, so the
runtime is never mutated by two frames at once; the cost is that a late frame
is caught up before the next one starts.

The guard is deliberately conservative: it reserves a little headroom for
thread start and wake-up, so the measured caller latency stays under the
budget even when the worker overruns.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


@dataclass
class GuardSnapshot:
    budget_ms: float
    overhead_target_ms: float
    spin: bool
    switch_interval: float | None
    frames: int
    timeouts: int
    timeout_rate: float
    max_consecutive_timeouts: int
    degraded_cooldown_frames: int
    degraded_frames: int
    elapsed_ms: dict[str, float]
    worker_late_ms: dict[str, float]
    errors: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget_ms": float(self.budget_ms),
            "overhead_target_ms": float(self.overhead_target_ms),
            "spin": bool(self.spin),
            "switch_interval": self.switch_interval,
            "frames": int(self.frames),
            "timeouts": int(self.timeouts),
            "timeout_rate": float(self.timeout_rate),
            "max_consecutive_timeouts": int(self.max_consecutive_timeouts),
            "degraded_cooldown_frames": int(self.degraded_cooldown_frames),
            "degraded_frames": int(self.degraded_frames),
            "elapsed_ms": dict(self.elapsed_ms),
            "worker_late_ms": dict(self.worker_late_ms),
            "errors": int(self.errors),
        }


@dataclass
class DeadlineGuard:
    """Run a callable under a wall-clock deadline.

    ``budget_ms`` is the contract bound.  The caller waits at most
    ``budget_ms - reserve_ms`` so thread start and wake-up overhead do not push
    the measured latency past the budget.  A timed-out call returns
    ``(None, True, elapsed_ms)``; the caller substitutes a degraded result.
    """

    budget_ms: float = 10.0
    reserve_ms: float = 1.5
    max_consecutive_timeouts_before_abstain: int = 5
    degraded_cooldown_frames: int = 3
    spin: bool = True
    switch_interval: float | None = 0.0005
    _frames: int = field(default=0, init=False)
    _timeouts: int = field(default=0, init=False)
    _consecutive: int = field(default=0, init=False)
    _max_consecutive: int = field(default=0, init=False)
    _degraded_frames: int = field(default=0, init=False)
    _elapsed: list[float] = field(default_factory=list, init=False)
    _worker_late: list[float] = field(default_factory=list, init=False)
    _errors: int = field(default=0, init=False)
    _pending: threading.Thread | None = field(default=None, init=False)
    _pending_start: float = field(default=0.0, init=False)
    _pending_box: dict[str, Any] | None = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        self.budget_ms = float(self.budget_ms)
        self.reserve_ms = max(0.0, float(self.reserve_ms))
        if self.budget_ms <= self.reserve_ms:
            raise ValueError("budget_ms must be larger than reserve_ms")

    @property
    def wait_ms(self) -> float:
        return self.budget_ms - self.reserve_ms

    @property
    def consecutive_timeouts(self) -> int:
        return int(self._consecutive)

    @property
    def degraded_frames(self) -> int:
        return int(self._degraded_frames)

    @property
    def should_shed(self) -> bool:
        """True for a short cooldown after a deadline miss."""
        return self._degraded_frames > 0

    @property
    def should_abstain(self) -> bool:
        return self._consecutive >= int(self.max_consecutive_timeouts_before_abstain)

    def call(
        self,
        fn: Callable[[], Any],
        *,
        frame: int | None = None,
    ) -> tuple[Any, bool, float]:
        """Return ``(value, timed_out, elapsed_ms)``.

        ``value`` is ``None`` when the call timed out.  The worker keeps
        running; call :meth:`wait_pending` before the next call.
        """
        self.wait_pending()
        box: dict[str, Any] = {}
        done = threading.Event()
        thread_frame = frame

        def target() -> None:
            try:
                box["value"] = fn()
            except BaseException as exc:  # noqa: BLE001 - propagate to caller
                box["error"] = exc
            finally:
                done.set()

        thread = threading.Thread(
            target=target,
            name=f"pss-deadline-{thread_frame}" if thread_frame is not None else "pss-deadline",
            daemon=True,
        )
        started = time.perf_counter()
        previous_interval = None
        if self.switch_interval is not None:
            previous_interval = sys.getswitchinterval()
            sys.setswitchinterval(float(self.switch_interval))
        try:
            thread.start()
            if self.spin:
                # Windows Event.wait() quantizes timeouts to ~15.6 ms, which
                # would make a 10 ms guard useless.  A yielding spin-wait keeps
                # ~1 ms precision.  A short GIL switch interval lets the main
                # thread wake even if the worker is CPU-bound.
                deadline = started + self.wait_ms / 1000.0
                while not done.is_set():
                    if time.perf_counter() >= deadline:
                        break
                    time.sleep(0)
                finished = done.is_set()
            else:
                finished = done.wait(self.wait_ms / 1000.0)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
        finally:
            if previous_interval is not None:
                sys.setswitchinterval(previous_interval)
        self._frames += 1
        self._elapsed.append(elapsed_ms)
        if finished:
            self._consecutive = 0
            self._degraded_frames = max(0, self._degraded_frames - 1)
            if "error" in box:
                raise box["error"]
            return box.get("value"), False, elapsed_ms
        self._timeouts += 1
        self._consecutive += 1
        self._degraded_frames = max(
            self._degraded_frames, int(self.degraded_cooldown_frames)
        )
        self._max_consecutive = max(self._max_consecutive, self._consecutive)
        self._pending = thread
        self._pending_start = started
        self._pending_box = box
        return None, True, elapsed_ms

    def wait_pending(self) -> None:
        """Join the worker from a timed-out call, if any."""
        with self._lock:
            thread = self._pending
            if thread is None:
                return
            started = self._pending_start
            box = self._pending_box
            self._pending = None
            self._pending_box = None
        thread.join()
        if box is not None and "error" in box:
            self._errors += 1
        late_ms = (time.perf_counter() - started) * 1000.0
        self._worker_late.append(late_ms)

    def close(self) -> None:
        self.wait_pending()

    def snapshot(self) -> GuardSnapshot:
        self.wait_pending()
        elapsed = dict(
            p50=_percentile(self._elapsed, 50),
            p95=_percentile(self._elapsed, 95),
            p99=_percentile(self._elapsed, 99),
            p999=_percentile(self._elapsed, 99.9),
            max=max(self._elapsed) if self._elapsed else 0.0,
        )
        late = dict(
            p50=_percentile(self._worker_late, 50),
            p95=_percentile(self._worker_late, 95),
            p99=_percentile(self._worker_late, 99),
            max=max(self._worker_late) if self._worker_late else 0.0,
        )
        return GuardSnapshot(
            budget_ms=self.budget_ms,
            overhead_target_ms=self.reserve_ms,
            spin=bool(self.spin),
            switch_interval=self.switch_interval,
            frames=self._frames,
            timeouts=self._timeouts,
            timeout_rate=(self._timeouts / self._frames) if self._frames else 0.0,
            max_consecutive_timeouts=self._max_consecutive,
            degraded_cooldown_frames=int(self.degraded_cooldown_frames),
            degraded_frames=int(self._degraded_frames),
            elapsed_ms=elapsed,
            worker_late_ms=late,
            errors=self._errors,
        )


class LatencyIsolation:
    """Best-effort OS tuning for tail latency.

    On Windows this raises the timer resolution to 1 ms and the calling
    thread's priority to ``THREAD_PRIORITY_HIGHEST``.  On other platforms it
    tries a modest ``nice`` decrease.  It is not a hard real-time guarantee;
    it reduces the chance that the guard thread itself is descheduled.
    """

    def __init__(self) -> None:
        self._restores: list[Callable[[], None]] = []

    def __enter__(self) -> "LatencyIsolation":
        if sys.platform == "win32":
            try:
                import ctypes

                winmm = ctypes.WinDLL("winmm")
                if winmm.timeBeginPeriod(1) == 0:
                    self._restores.append(lambda: winmm.timeEndPeriod(1))
                kernel32 = ctypes.WinDLL("kernel32")
                handle = kernel32.GetCurrentThread()
                old_priority = kernel32.GetThreadPriority(handle)
                if old_priority != 0x7FFFFFFF:  # THREAD_PRIORITY_ERROR_RETURN
                    if kernel32.SetThreadPriority(
                        handle, 2  # THREAD_PRIORITY_HIGHEST
                    ):
                        self._restores.append(
                            lambda: kernel32.SetThreadPriority(handle, old_priority)
                        )
            except Exception:
                pass
        else:
            try:
                old_nice = os.nice(0)
                os.nice(-5)
                self._restores.append(lambda: os.nice(old_nice))
            except Exception:
                pass
        return self

    def __exit__(self, *exc_info) -> None:
        while self._restores:
            restore = self._restores.pop()
            try:
                restore()
            except Exception:
                pass

