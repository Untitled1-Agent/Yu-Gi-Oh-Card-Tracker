"""Compare batch scheduling overhead without network or disk latency.

Run from the repository root:
    python scripts/benchmark_image_downloads.py --images 15000 --concurrency 20

By default the baseline source is read from the upstream commit using git show.
For shallow checkouts, supply --baseline /path/to/original/image_manager.py.
No NiceGUI server, HTTP requests, or user-data writes are performed. Timings are
synthetic scheduler measurements, not real image-download throughput.
"""

import argparse
import asyncio
import gc
import json
import statistics
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


BASELINE_REF = "e892514af66d914ef9facaa06fceecd4e77ba502"
SOURCE_PATH = "src/services/image_manager.py"
ROOT = Path(__file__).resolve().parents[1]


class FakeSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


def load_manager(source, label):
    """Load each version with equivalent offline dependencies and no side effects."""
    module = ModuleType(label)
    mocks = {
        "nicegui": SimpleNamespace(run=SimpleNamespace(io_bound=asyncio.to_thread)),
        "aiohttp": SimpleNamespace(ClientSession=FakeSession),
        "PIL": SimpleNamespace(Image=None),
    }
    with patch.dict(sys.modules, mocks), patch("os.makedirs"):
        exec(compile(source, label, "exec"), module.__dict__)
        manager = module.ImageManager()
    manager.image_exists = lambda card_id, high_res=False: False
    return manager


async def measure(manager, urls, concurrency):
    tasks_before = asyncio.all_tasks()
    worker_tasks = 0
    sampled = False
    attempts = 0

    async def download(session, card_id, url, path):
        nonlocal worker_tasks, sampled, attempts
        if not sampled:
            worker_tasks = len(asyncio.all_tasks() - tasks_before)
            sampled = True
        attempts += 1
        # Yield once, like an asynchronous request, without contacting a server.
        await asyncio.sleep(0)
        return path

    manager._download_with_session = download
    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()
    try:
        await manager.download_batch(urls, concurrency=concurrency)
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert attempts == len(urls)
    return {"seconds": elapsed, "peak_mib": peak / (1024 * 1024), "worker_tasks": worker_tasks}


async def benchmark(baseline, optimized, images, concurrency, runs):
    urls = {i: f"offline:{i}" for i in range(images)}
    managers = {
        "baseline": load_manager(baseline, "baseline_image_manager"),
        "optimized": load_manager(optimized, "optimized_image_manager"),
    }
    # Warm the executor before tracing or timing either implementation.
    await asyncio.to_thread(lambda: None)
    results = {name: [] for name in managers}
    for run in range(runs):
        # Alternate order to reduce systematic warm-up/order effects.
        names = list(managers) if run % 2 == 0 else list(reversed(managers))
        for name in names:
            results[name].append(await measure(managers[name], urls, concurrency))
    return {
        "python": sys.version.split()[0],
        "images": images,
        "concurrency": concurrency,
        "runs": runs,
        "method": "mocked downloads and cache checks; real asyncio scheduling; tracemalloc enabled",
        "results": {
            name: {
                "median_seconds": statistics.median(row["seconds"] for row in rows),
                "median_peak_mib": statistics.median(row["peak_mib"] for row in rows),
                "worker_tasks": max(row["worker_tasks"] for row in rows),
            }
            for name, rows in results.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, help="Original image_manager.py (instead of git show)")
    parser.add_argument("--images", type=int, default=15000)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    if min(args.images, args.concurrency, args.runs) < 1:
        parser.error("images, concurrency, and runs must all be positive")
    try:
        baseline = args.baseline.read_text(encoding="utf-8") if args.baseline else subprocess.check_output(
            ["git", "show", f"{BASELINE_REF}:{SOURCE_PATH}"], cwd=ROOT, text=True, encoding="utf-8",
        )
        optimized = (ROOT / SOURCE_PATH).read_text(encoding="utf-8")
    except (OSError, subprocess.CalledProcessError) as exc:
        parser.error(f"Cannot load source: {exc}. Use --baseline to supply the original file.")
    result = asyncio.run(benchmark(baseline, optimized, args.images, args.concurrency, args.runs))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
