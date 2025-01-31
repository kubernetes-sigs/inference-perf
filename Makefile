VENV := .venv

# Format Python code with ruff format
.PHONY: format
format:
	@echo "Formatting Python files with ruff format..."
	$(VENV)/bin/ruff format

# Run ruff check to lint Python code in the whole repository
.PHONY: lint
lint:
	@echo "Linting Python files with ruff check..."
	$(VENV)/bin/ruff check

# Perform type checking
.PHONY: type-check
type-check:
	@echo "Running type checking with mypy..."
	$(VENV)/bin/mypy --strict ./inference_perf

# Check for and install dependencies
.PHONY: all-deps
all-deps:
	@echo "Creating virtual environment if it doesn't exist..."
	@if [ ! -d $(VENV) ]; then \
	    python3 -m venv $(VENV); \
	fi
	@echo "Activating virtual environment and installing dependencies..."
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e .

.PHONY: check
check: all-deps lint type-check
