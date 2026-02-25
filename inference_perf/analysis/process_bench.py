import multiprocessing as mp
import time
import os


# A simple target function for the processes to run.
# It does nothing to isolate the startup overhead.
def worker():
    pass


def run_benchmark():
    num_processes = 128
    processes = []

    # --- Start timing ---
    # process_time: Measures CPU time of the current process.
    # perf_counter: Measures wall-clock time.
    start_cpu_time = time.process_time()
    start_wall_time = time.perf_counter()

    # Create and start all 128 processes
    for _ in range(num_processes):
        p = mp.Process(target=worker)
        processes.append(p)
        p.start()

    # Wait for all processes to finish starting and complete their work
    for p in processes:
        p.join()

    # --- Stop timing ---
    end_cpu_time = time.process_time()
    end_wall_time = time.perf_counter()

    # --- Calculate and print results ---
    cpu_time_ms = (end_cpu_time - start_cpu_time) * 1000
    wall_time_ms = (end_wall_time - start_wall_time) * 1000

    print(f"Operating System: {os.name} ({mp.get_start_method()})")
    print(f"Spawning {num_processes} processes took:")
    print(f"  - CPU Time: {cpu_time_ms:.2f} ms")
    print(f"  - Wall-Clock Time: {wall_time_ms:.2f} ms")


if __name__ == "__main__":
    # On Windows/macOS, it's essential to protect the main script
    # with this `if` block to prevent infinite process spawning.
    run_benchmark()
