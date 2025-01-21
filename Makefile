VENV := .venv

# Format the Python code in the whole repository
format:
	@echo "Formatting Python files with black..."
	$(VENV)/bin/black .

# Check if the code is properly formatted (without modifying files)
format-check:
	@echo "Checking Python formatting with black..."
	$(VENV)/bin/black --check .

# Perform type checking
type-check:
	@echo "Running type checking with mypy..."
	$(VENV)/bin/mypy --strict .

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
check: deps format-check type-check