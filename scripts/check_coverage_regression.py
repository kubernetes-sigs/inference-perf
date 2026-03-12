import json
import subprocess
import sys
import shutil
import os
from pathlib import Path
import argparse

# Ensure PDM doesn't reuse the current virtualenv when we are trying to use another one (e.g. for baseline)
os.environ["PDM_IGNORE_ACTIVE_VENV"] = "1"


def get_coverage_data(report_path):
    if not Path(report_path).exists():
        return None
    with open(report_path) as f:
        return json.load(f)


def run_command(cmd, cwd=None):
    """Helper to run shell commands."""
    try:
        return subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Command failed with exit code {e.returncode}: {cmd}")
        if e.stdout:
            print(f"--- STDOUT ---\n{e.stdout}")
        if e.stderr:
            print(f"--- STDERR ---\n{e.stderr}")
        raise


def generate_baseline(output_path: Path):
    """Uses git worktree to generate coverage for the main branch."""
    # Create temp dir inside current project to ensure parent-path relative logic works
    project_root = Path.cwd()
    temp_dir = project_root / ".baseline_temp"

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    print("--- Generating baseline coverage from 'main' ---")
    try:
        # 1. Create a temporary worktree of the main branch
        try:
            print("Fetching latest 'main' branch from origin...")
            run_command("git fetch origin main")
            run_command(f"git worktree add -d {temp_dir} origin/main")
        except subprocess.CalledProcessError:
            print("⚠️ Could not fetch origin/main, falling back to local 'main' branch...")
            run_command(f"git worktree add -d {temp_dir} main")

        # 2. Setup a fresh virtual environment to avoid contamination
        print("Setting up fresh virtualenv for baseline...")
        run_command("python -m venv .venv", cwd=temp_dir)

        print("Installing dependencies in baseline environment...")
        # We must make sure we don't accidentally use the parent's __pypackages__ if configured that way,
        # but standard behavior with .venv presence is to use it.
        run_command("pdm install -dG test", cwd=temp_dir)

        # 3. Run tests
        print("Running tests on main branch...")
        # Use the current project's pyproject.toml for coverage config to ensure consistent settings (e.g. omit rules)
        # We need to map the project_root to the absolute path
        config_path = project_root / "pyproject.toml"
        run_command(
            f"pdm run pytest --cov=inference_perf --cov-config={config_path.absolute()} --cov-report=json:{output_path.absolute()} tests/",
            cwd=temp_dir,
        )

        print(f"✅ Baseline generated: {output_path.name}")
    finally:
        # 5. Cleanup: remove the worktree folder and the git reference
        try:
            run_command(f"git worktree remove {temp_dir} --force")
        except Exception:
            # If cleanup fails, we don't want to mask the original error
            pass


def generate_current(output_path: Path):
    """Generates coverage for the current branch/environment."""
    print("--- Generating current coverage ---")
    # Use the current python executable to run pytest directly
    # This avoids issues with nested 'pdm run' calls and ensures we use the same environment
    cmd = f"{sys.executable} -m pytest --cov=inference_perf --cov-report=json:{output_path.absolute()} tests/"
    run_command(cmd)
    print(f"✅ Current report generated: {output_path.name}")


def print_detailed_report(current_data, baseline_data):
    """Prints a detailed per-file comparison."""
    current_files = current_data.get("files", {})
    baseline_files = baseline_data.get("files", {})

    all_filenames = sorted(set(current_files.keys()) | set(baseline_files.keys()))

    print("\n--- Detailed Per-File Coverage Comparison ---")
    print(f"{'File':<50} {'Baseline':>10} {'Current':>10} {'Diff':>10}")
    print("-" * 83)

    for filename in all_filenames:
        curr_cov = current_files.get(filename, {}).get("summary", {}).get("percent_covered")
        base_cov = baseline_files.get(filename, {}).get("summary", {}).get("percent_covered")

        curr_str = f"{curr_cov:>9.2f}%" if curr_cov is not None else f"{'N/A':>10}"
        base_str = f"{base_cov:>9.2f}%" if base_cov is not None else f"{'N/A':>10}"

        diff_str = ""
        if curr_cov is not None and base_cov is not None:
            diff = curr_cov - base_cov
            color = ""
            if diff > 0.01:
                color = "✅ "
            elif diff < -0.01:
                color = "❌ "
            diff_str = f"{color}{diff:>+9.2f}%"

        print(f"{filename:<50} {base_str} {curr_str} {diff_str}")


def main():
    parser = argparse.ArgumentParser(description="Check for coverage regression.")
    parser.add_argument("--force", action="store_true", help="Force regeneration of coverage reports")
    parser.add_argument("--detailed", action="store_true", help="Show detailed per-file comparison")

    args = parser.parse_args()

    current_report = Path("coverage.json")
    baseline_report = Path("coverage_main.json")

    # Generate Baseline
    if args.force or not baseline_report.exists():
        try:
            generate_baseline(baseline_report)
        except subprocess.CalledProcessError:
            # Error details already printed by run_command
            sys.exit(1)
        except Exception as e:
            print(f"❌ Unexpected error generating baseline: {e}")
            sys.exit(1)
    else:
        print(f"ℹ️  Using existing baseline: {baseline_report} (pass --force to regenerate)")

    # Generate Current
    if args.force or not current_report.exists():
        try:
            generate_current(current_report)
        except subprocess.CalledProcessError:
            # Error details already printed by run_command
            sys.exit(1)
        except Exception as e:
            print(f"❌ Unexpected error generating current report: {e}")
            sys.exit(1)
    else:
        print(f"ℹ️  Using existing current report: {current_report} (pass --force to regenerate)")

    current_data = get_coverage_data(current_report)
    baseline_data = get_coverage_data(baseline_report)

    # Safeguard against missing data
    if current_data is None or baseline_data is None:
        print("❌ Error: Could not read coverage values.")
        sys.exit(1)

    current_val = current_data["totals"]["percent_covered"]
    baseline_val = baseline_data["totals"]["percent_covered"]

    if args.detailed:
        print_detailed_report(current_data, baseline_data)

    print("\n--- Coverage Summary ---")
    print(f"Main Branch:    {baseline_val:.2f}%")
    print(f"Current Branch: {current_val:.2f}%")

    # Use a small epsilon (0.01) to handle floating point precision issues
    if current_val < (baseline_val - 0.01):
        diff = baseline_val - current_val
        print(f"❌ FAIL: Total coverage decreased by {diff:.2f}%")
        sys.exit(1)

    print("✅ PASS: Coverage is maintained or improved.")


if __name__ == "__main__":
    main()
