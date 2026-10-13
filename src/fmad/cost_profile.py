"""Small, opt-in GPU memory sampler for the inference cost profile."""

from __future__ import annotations

import atexit
import csv
import os
import shutil
import subprocess
import threading


class NvidiaSmiPeakSampler:
    """Sample total used memory on the explicitly selected physical GPU."""

    def __init__(self, gpu_uuid: str, interval_ms: int = 100) -> None:
        if not gpu_uuid.startswith("GPU-"):
            raise ValueError("cost profiling requires CUDA_VISIBLE_DEVICES to contain a GPU UUID")
        self.gpu_uuid = gpu_uuid
        self.interval_ms = interval_ms
        self.peak_mib: int | None = None
        self.sample_count = 0
        self._ready = threading.Event()
        self._error: Exception | None = None
        self._process: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None

    @classmethod
    def from_environment(cls, interval_ms: int = 100) -> "NvidiaSmiPeakSampler":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        gpu_uuid = visible.split(",", 1)[0].strip()
        return cls(gpu_uuid, interval_ms=interval_ms)

    def _sample_output(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        try:
            for line in self._process.stdout:
                fields = next(csv.reader([line]))
                if len(fields) != 2 or fields[0].strip() != self.gpu_uuid:
                    continue
                used_mib = int(fields[1].strip())
                self.peak_mib = used_mib if self.peak_mib is None else max(self.peak_mib, used_mib)
                self.sample_count += 1
                self._ready.set()
        except Exception as error:  # surfaced synchronously by stop()
            self._error = error
            self._ready.set()

    def start(self) -> None:
        executable = shutil.which("nvidia-smi")
        if executable is None:
            raise RuntimeError("nvidia-smi is required for cost-profile VRAM sampling")
        self._process = subprocess.Popen(
            [
                executable,
                "--query-gpu=uuid,memory.used",
                "--format=csv,noheader,nounits",
                f"--loop-ms={self.interval_ms}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        atexit.register(self._cleanup)
        self._thread = threading.Thread(target=self._sample_output, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=10):
            self.stop()
            raise RuntimeError(f"nvidia-smi did not report selected GPU {self.gpu_uuid}")
        if self._error is not None:
            self.stop()
            raise RuntimeError("failed to sample nvidia-smi output") from self._error

    def stop(self) -> dict:
        if self._process is None:
            raise RuntimeError("nvidia-smi memory sampler was not started")
        self._cleanup()
        if self._error is not None:
            raise RuntimeError("failed to sample nvidia-smi output") from self._error
        if self.peak_mib is None or self.sample_count == 0:
            raise RuntimeError(f"nvidia-smi returned no samples for selected GPU {self.gpu_uuid}")
        return {
            "gpu_uuid": self.gpu_uuid,
            "peak_used_mib": self.peak_mib,
            "peak_used_gib": self.peak_mib / 1024.0,
            "sample_count": self.sample_count,
            "interval_ms": self.interval_ms,
        }

    def _cleanup(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
        try:
            self._process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=2)
        if self._thread is not None:
            self._thread.join(timeout=2)
