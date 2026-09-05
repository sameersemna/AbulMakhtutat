"""GPU thermal guard to prevent overheating / system crashes.

The Dell Pro Max GB10 runs a 35 GB vision model (`qwen3-vl:32b`). Firing many
concurrent inference requests can spike GPU power draw and temperature, causing
thermal shutdown. This module polls GPU temperature via `nvidia-smi` and blocks
new OCR work while the GPU is above a safe threshold, resuming once it cools.

Fail-soft: if `nvidia-smi` is unavailable (non-NVIDIA host, missing driver), the
guard is a no-op and the pipeline runs normally.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
from typing import Any

from config import Config, get_config

logger = logging.getLogger(__name__)


class ThermalGuard:
    """Polls GPU temperature and gates work to avoid thermal shutdown."""

    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self._has_nvidia = shutil.which("nvidia-smi") is not None
        if not self._has_nvidia:
            logger.info("nvidia-smi not found; thermal guard disabled")

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
        """Optional small delay between pages to let the GPU breathe."""
        delay = self.config.pipeline.thermal.inter_page_delay
        if delay > 0:
            await asyncio.sleep(delay)
