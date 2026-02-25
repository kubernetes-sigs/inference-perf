import json
import sys
from pathlib import Path

def get_coverage(report_path):
    with open(report_path) as f:
        data = json.load(f)
    return data["totals"]["percent_covered"]

def main():
    current_report = Path("coverage.json")
    # In a CI environment, you'd download this from the 'main' build artifacts
    baseline_report = Path("coverage_main.json") 

    if not current_report.exists():
        print("Error: current coverage.json not found.")
        sys.exit(1)

    current_val = get_coverage(current_report)
    
    if not baseline_report.exists():
        print(f"No baseline found. Current coverage: {current_val:.2f}%")
        # Optional: Save current as baseline for future runs
        sys.exit(0)

    baseline_val = get_coverage(baseline_report)

    print(f"Main: {baseline_val:.2f}% | Branch: {current_val:.2f}%")

    if current_val < baseline_val:
        print(f"FAIL: Coverage decreased by {baseline_val - current_val:.2f}%")
        sys.exit(1)
    
    print("PASS: Coverage is maintained or improved.")

if __name__ == "__main__":
    main()