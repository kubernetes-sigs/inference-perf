import os
import glob
import yaml
import pytest
from inference_perf.config import Config, sanitize_config


def test_examples_validation() -> None:
    """Find all .yml and .yaml files in examples/ directory and validate them."""
    examples_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples"))
    assert os.path.exists(examples_dir), f"Examples directory not found at {examples_dir}"

    # Find all .yml and .yaml files
    yml_files = glob.glob(os.path.join(examples_dir, "**", "*.yml"), recursive=True)
    yaml_files = glob.glob(os.path.join(examples_dir, "**", "*.yaml"), recursive=True)
    all_files = yml_files + yaml_files

    assert len(all_files) > 0, "No example config files found!"

    success_count = 0
    errors = []

    for filename in all_files:
        # Skip docker-compose and prometheus configs as they are not app configs
        if "docker-compose" in filename or "prometheus" in filename:
            continue

        with open(filename, "r") as f:
            try:
                cfg = yaml.safe_load(f)
            except Exception as e:
                errors.append(f"Failed to parse YAML in {filename}: {e}")
                continue

            if not isinstance(cfg, dict):
                # Not a dict config, skip
                continue

            # Check if it looks like an app config (has at least one expected top-level key)
            expected_keys = ["api", "data", "load", "metrics", "report", "storage", "server", "tokenizer", "circuit_breakers"]
            if not any(k in cfg for k in expected_keys):
                continue

            try:
                # Apply sanitization
                sanitized = sanitize_config(cfg)

                # Pop circuit_breakers if they are list of dicts, like read_config does
                cb_list = sanitized.pop("circuit_breakers", [])
                if cb_list and isinstance(cb_list[0], dict):
                    pass  # Handled separately in read_config
                elif cb_list:
                    # Put it back if it's strings (though read_config handles both, but for validation we need to match)
                    sanitized["circuit_breakers"] = cb_list

                # Validate
                Config.model_validate(sanitized)
                success_count += 1
            except Exception as e:
                errors.append(f"Validation failed for {filename}: {e}")

    if errors:
        error_msg = "\n".join(errors)
        pytest.fail(f"Validation failed for some examples:\n{error_msg}")

    print(f"Successfully validated {success_count} example configurations!")
