"""GPU thermal guard to prevent overheating / system crashes.

The Dell Pro Max GB10 runs vision models that can spike GPU temperature during
sustained inference. A single pre-inference temperature check is insufficient
because the spike happens *during* the (blocking) inference call and cools
before the next check. This module therefore runs a **continuous background
monitor** that samples temperature every ~2s, tracks the peak seen since the
last page, and forces an adaptive cooldown when the peak exceeded a critical
threshold.

Fail-soft: if `nvidia-smi` is unavailable (non-NVIDIA host, missing driver), the
guard is a no-op and the pipeline runs normally.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import time
from typing import Any

from config import Config, get_config

logger = logging.getLogger(__name__)


class ThermalGuard:
    """Continuously monitors GPU temperature and gates work to avoid shutdown."""

    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self._has_nvidia = shutil.which("nvidia-smi") is not None
        if not self._has_nvidia:
            logger.info("nvidia-smi not found; thermal guard disabled")

        # Peak temperature observed since the last reset (updated by the monitor).
        self._peak_temp: float = 0.0
        self._peak_lock = asyncio.Lock()
        self._monitor_task: asyncio.Task | None = None

    def _read_temp(self) -> float | None:
        """Return current GPU temperature in Celsius, or None if unavailable."""
        if not self._has_nvidia:
            return None
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if out.returncode != 0:
                return None
            temps = [float(x) for x in out.stdout.strip().splitlines() if x.strip()]
            return max(temps) if temps else None
        except (subprocess.SubprocessError, ValueError, OSError):
            return None

    async def _monitor_loop(self) -> None:
        """Continuously sample temperature and track the rolling peak."""
        tc = self.config.pipeline.thermal
        while True:
            temp = self._read_temp()
            if temp is not None:
                async with self._peak_lock:
                    if temp > self._peak_temp:
                        self._peak_temp = temp
            await asyncio.sleep(tc.monitor_interval)

    def start(self) -> None:
        """Start the background temperature monitor (idempotent)."""
        if not self._has_nvidia or not self.config.pipeline.thermal.enabled:
            return
        if self._monitor_task is None or self._monitor_task.done():
            self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop(self) -> None:
        """Stop the background monitor."""
        if self._monitor_task is not None and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

    async def _get_peak(self) -> float:
        async with self._peak_lock:
            return self._peak_temp

    async def _reset_peak(self) -> None:
        async with self._peak_lock:
            self._peak_temp = 0.0

    async def wait_if_hot(self) -> None:
        """Block until the GPU is below the safe threshold (or guard disabled)."""
        tc = self.config.pipeline.thermal
        if not tc.enabled or not self._has_nvidia:
            return

        while True:
            temp = self._read_temp()
            if temp is None:
                return  # can't read temp; don't block
            if temp < tc.max_temp:
                return
            logger.warning(
                "GPU temperature %.0f°C >= threshold %.0f°C — pausing %.0fs to cool down",
                temp,
                tc.max_temp,
                tc.poll_interval,
            )
            await asyncio.sleep(tc.poll_interval)

    async def cooldown_between_pages(self) -> None:
        """Adaptive cooldown after each page.

        If the background monitor observed a peak above the critical threshold
        during the last page's inference, wait for the GPU to cool back below
        the safe threshold before proceeding. Otherwise apply the base delay.
        """
        tc = self.config.pipeline.thermal
        if not tc.enabled or not self._has_nvidia:
            return

        peak = await self._get_peak()
        await self._reset_peak()

        if peak >= tc.critical_temp:
            logger.warning(
                "GPU peaked at %.0f°C during last page (critical >= %.0f°C) — cooling down before next page",
                peak,
                tc.critical_temp,
            )
            # Wait until temperature drops below the safe threshold.
            while True:
                temp = self._read_temp()
                if temp is None or temp < tc.max_temp:
                    break
                await asyncio.sleep(tc.poll_interval)
            return

        # Base breathing room between pages.
        if tc.inter_page_delay > 0:
            await asyncio.sleep(tc.inter_page_delay)
