VENV := .venv

# Format Python code with black
.PHONY: format
format:
	@echo "Formatting Python files with black..."
	$(VENV)/bin/black .

# Run flake8 to lint Python code in the whole repository
.PHONY: lint
lint:
	@echo "Linting Python files with flake8..."
	# stop if there are Python syntax errors or undefined names
	$(VENV)/bin/flake8 ./inference_perf --count --select=E9,F63,F7,F82 --show-source --statistics
	# exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
	$(VENV)/bin/flake8 ./inference_perf --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# Perform type checking
.PHONY: type-check
type-check:
	@echo "Running type checking with mypy..."
	$(VENV)/bin/mypy --strict ./inference_perf

# Check for and install dependencies
.PHONY: deps
deps:
	@echo "Creating virtual environment if it doesn't exist..."
	@if [ ! -d $(VENV) ]; then \
	    python3 -m venv $(VENV); \
	fi
	@echo "Activating virtual environment and installing dependencies..."
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt

.PHONY: check
check: deps lint type-check