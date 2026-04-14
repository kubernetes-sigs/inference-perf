import asyncio
import json
import logging
import time
from collections import deque
import numpy as np
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.progress import Progress, BarColumn, TextColumn
from inference_perf.distributed.redis_client import RedisClient

logger = logging.getLogger(__name__)


class TUIDashboard:
    def __init__(self, redis_client: RedisClient, channel: str, config=None):
        self.redis = redis_client
        self.channel = channel
        self.config = config
        self.console = Console()
        self.layout = Layout()
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        )

        total_all_reqs = 100
        if config and config.load:
            total_all_reqs = sum(int(stage.rate * stage.duration) for stage in config.load.stages)

        self.overall_task = self.progress.add_task("Overall Progress", total=total_all_reqs)

        self.stage_tasks = []
        if config and config.load:
            for i, stage in enumerate(config.load.stages):
                total_reqs = int(stage.rate * stage.duration)
                t = self.progress.add_task(f"Stage {i}", total=total_reqs)
                self.stage_tasks.append(t)

        # Stats tracing
        self.latencies = deque(maxlen=1000)  # Keep last 1000 latencies for P99
        self.request_times = deque(maxlen=1000)  # For RPS calculation
        self.success_count = 0
        self.fail_count = 0
        self.ttfts = deque(maxlen=1000)  # Keep last 1000 for percentiles
        self.itls = deque(maxlen=1000)  # Keep last 1000 for percentiles

    def make_layout(self) -> Layout:
        progress_size = 3
        if self.config and self.config.load:
            progress_size = len(self.config.load.stages) + 3

        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="progress_panel", size=progress_size),
            Layout(name="footer", size=3),
        )
        self.layout["main"].split_row(Layout(name="metrics"))
        return self.layout

    def calculate_rps(self) -> float:
        if len(self.request_times) < 2:
            return 0.0
        duration = self.request_times[-1] - self.request_times[0]
        if duration == 0:
            return 0.0
        return len(self.request_times) / duration

    def calculate_p99(self) -> float:
        if not self.latencies:
            return 0.0
        return float(np.percentile(list(self.latencies), 99))

    async def run(self):
        await self.redis.connect()
        pubsub = await self.redis.get_subscriber()
        await pubsub.subscribe(self.channel)

        self.make_layout()
        self.layout["header"].update(Panel("Inference Perf Distributed Dashboard", style="bold green"))
        self.layout["footer"].update(Panel("Press Ctrl+C to exit", style="italic"))
        self.layout["progress_panel"].update(Panel(self.progress, title="Progress"))

        last_update = 0.0
        with Live(self.layout, refresh_per_second=4, console=self.console):
            while True:
                try:
                    message = await pubsub.get_message(ignore_subscribe_messages=True)

                    now = time.time()

                    if message:
                        data = json.loads(message["data"])

                        # Extract data from worker telemetry
                        status = data.get("status")
                        latency = data.get("latency", 0.0)
                        ttft = data.get("ttft")
                        itl = data.get("itl")

                        self.request_times.append(now)
                        self.latencies.append(latency)

                        if ttft is not None:
                            self.ttfts.append(ttft)
                        if itl is not None:
                            self.itls.append(itl)

                        if status == "success":
                            self.success_count += 1
                        else:
                            self.fail_count += 1

                    # Update UI at most 4 times per second
                    if now - last_update >= 0.25:
                        last_update = now

                        # Calculate live metrics
                        rps = self.calculate_rps()
                        p99 = self.calculate_p99()

                        # Calculate percentiles for TTFT and ITL
                        ttft_list = list(self.ttfts)
                        ttft_p50 = float(np.percentile(ttft_list, 50)) if ttft_list else 0.0
                        ttft_p90 = float(np.percentile(ttft_list, 90)) if ttft_list else 0.0
                        ttft_p99 = float(np.percentile(ttft_list, 99)) if ttft_list else 0.0

                        itl_list = list(self.itls)
                        itl_p50 = float(np.percentile(itl_list, 50)) if itl_list else 0.0
                        itl_p90 = float(np.percentile(itl_list, 90)) if itl_list else 0.0
                        itl_p99 = float(np.percentile(itl_list, 99)) if itl_list else 0.0

                        metrics_text = (
                            f"Live RPS: {rps:.2f}\n"
                            f"P99 Latency: {p99:.3f}s\n"
                            f"TTFT (ms): P50: {ttft_p50 * 1000:.1f} | P90: {ttft_p90 * 1000:.1f} | P99: {ttft_p99 * 1000:.1f}\n"
                            f"ITL (ms): P50: {itl_p50 * 1000:.1f} | P90: {itl_p90 * 1000:.1f} | P99: {itl_p99 * 1000:.1f}\n"
                            f"Failures: {self.fail_count}\n"
                            f"Success: {self.success_count}"
                        )
                        self.layout["metrics"].update(Panel(metrics_text, title="Live Metrics"))

                        # Update progress
                        completed_str = await self.redis.redis.get("completed_requests")
                        if completed_str:
                            completed = int(completed_str)
                            self.progress.update(self.overall_task, completed=completed)

                        # Update per-stage progress
                        if self.config and self.config.load:
                            for i, task_id in enumerate(self.stage_tasks):
                                completed_str = await self.redis.redis.get(f"completed_requests_stage_{i}")
                                if completed_str:
                                    completed = int(completed_str)
                                    self.progress.update(task_id, completed=completed)

                    await asyncio.sleep(0.01)  # Small sleep to yield
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in TUI: {e}")
                    await asyncio.sleep(1)

        await pubsub.unsubscribe(self.channel)
