"""GPU monitoring via background nvidia-smi sampling."""
import json
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import List


@dataclass
class GPUSample:
    timestamp: float
    gpu_util: float  # percentage
    memory_used_mb: float
    memory_total_mb: float


@dataclass
class GPUMetrics:
    samples: List[GPUSample] = field(default_factory=list)

    def mean_util(self) -> float:
        if not self.samples:
            return 0.0
        return sum(s.gpu_util for s in self.samples) / len(self.samples)

    def p95_util(self) -> float:
        if not self.samples:
            return 0.0
        sorted_vals = sorted(s.gpu_util for s in self.samples)
        idx = int(len(sorted_vals) * 0.95)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    def peak_vram_gb(self) -> float:
        if not self.samples:
            return 0.0
        return max(s.memory_used_mb for s in self.samples) / 1024.0


class GPUMonitor:
    """Sample nvidia-smi in a background thread."""

    def __init__(self, interval_sec: float = 1.0, device_id: int = 0):
        self.interval_sec = interval_sec
        self.device_id = device_id
        self.samples: List[GPUSample] = []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def _sample(self) -> GPUSample | None:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=json",
                    f"--id={self.device_id}",
                ],
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            data = json.loads(result.stdout)
            if not data:
                return None
            gpu = data[0]
            return GPUSample(
                timestamp=time.time(),
                gpu_util=float(gpu["utilization.gpu [%]"].replace(" %", "").strip()),
                memory_used_mb=float(gpu["memory.used [MiB]"].replace(" MiB", "").strip()),
                memory_total_mb=float(gpu["memory.total [MiB]"].replace(" MiB", "").strip()),
            )
        except Exception:
            return None

    def _loop(self):
        while not self._stop_event.is_set():
            sample = self._sample()
            if sample:
                self.samples.append(sample)
            self._stop_event.wait(self.interval_sec)

    def start(self):
        self._stop_event.clear()
        self.samples = []
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> GPUMetrics:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        return GPUMetrics(samples=self.samples)
