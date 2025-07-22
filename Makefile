# Makefile for QKD System Failure Auto Detection Project

# Python interpreter
PYTHON = python3

# Project directories
SRC_DIR = src
TEST_DIR = tests
NOTEBOOK_DIR = notebooks
DEMO_DIR = demos

# Virtual environment
VENV = .venv
VENV_PYTHON = $(VENV)/bin/python
VENV_PIP = $(VENV)/bin/pip

.PHONY: help setup install test demo clean notebook lint format analyze

help:
	@echo "QKD System Failure Auto Detection Project"
	@echo "========================================"
	@echo ""
	@echo "Available targets:"
	@echo "  setup     - Create virtual environment and install dependencies"
	@echo "  install   - Install dependencies (assumes venv exists)"
	@echo "  test      - Run unit tests"
	@echo "  demo      - Run all demonstration scripts"
	@echo "  analyze   - Run analysis notebooks"
	@echo "  notebook  - Start Jupyter notebook server"
	@echo "  lint      - Run code linting"
	@echo "  format    - Format code with black"
	@echo "  clean     - Clean up generated files"
	@echo "  help      - Show this help message"

setup: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "Installing dependencies..."
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.txt
	@echo "Setup complete! Activate with: source $(VENV)/bin/activate"

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

test:
	@echo "Running unit tests..."
	$(PYTHON) -m pytest $(TEST_DIR)/test_detectors.py -v

# Run complete demonstration
demo: setup
	@echo "Running complete QKD failure detection demonstration..."
	python run_all_demos.py
	@echo "Demonstration completed successfully!"

# Run test suite
test: setup
	@echo "Running QKD failure detection test suite..."
	cd tests && python test_detectors.py
	@echo "Test suite completed!"

# Run pytest (if available)
pytest: setup
	@echo "Running pytest test suite..."
	@if command -v pytest >/dev/null 2>&1; then 
		pytest tests/ -v --tb=short; 
	else 
		echo "pytest not available, installing..."; 
		pip install pytest pytest-cov; 
		pytest tests/ -v --tb=short; 
	fi

# Run tests with coverage
test-coverage: setup
	@echo "Running test suite with coverage analysis..."
	@if command -v pytest >/dev/null 2>&1; then 
		pytest tests/ -v --cov=src/ --cov-report=html --cov-report=term; 
	else 
		echo "pytest not available, running basic tests..."; 
		cd tests && python test_detectors.py; 
	fi

# Generate test data
test-data: setup
	@echo "Generating test data and scenarios..."
	cd tests && python test_config.py
	@echo "Test data generation completed!"

# Run performance benchmarks
benchmark: setup
	@echo "Running performance benchmarks..."
	cd tests && python -c "from test_detectors import run_performance_benchmarks; run_performance_benchmarks()"
	@echo "Benchmarks completed!"

# Validate project
validate: setup test
	@echo "Validating complete QKD project..."
	@echo "✓ Dependencies installed"
	@echo "✓ Tests passed"
	@echo "✓ System ready for use"

# Clean generated files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf tests/data/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/

# Show project information
info:
	@echo "QKD Failure Detection System"
	@echo "============================"
	@echo "Under the guidance of Vijayalaxmi Mogiligidda"
	@echo ""
	@echo "Project Structure:"
	@echo "- src/: Core implementation modules"
	@echo "- demos/: Demonstration scripts"
	@echo "- tests/: Comprehensive test suite"
	@echo "- notebooks/: Jupyter notebooks for analysis"
	@echo "- plots/: Generated visualization outputs"
	@echo "- resources/: Documentation and references"
	@echo ""
	@echo "Available targets:"
	@echo "- make setup: Install dependencies"
	@echo "- make demo: Run complete demonstration"
	@echo "- make test: Run test suite"
	@echo "- make pytest: Run pytest with advanced features"
	@echo "- make test-coverage: Run tests with coverage analysis"
	@echo "- make test-data: Generate test data and scenarios"
	@echo "- make benchmark: Run performance benchmarks"
	@echo "- make validate: Complete project validation"
	@echo "- make clean: Clean generated files"
	@echo "- make info: Show this information"

.PHONY: setup demo test pytest test-coverage test-data benchmark validate clean info

analyze:
	@echo "Running analysis notebooks..."
	$(VENV_PYTHON) -m jupyter nbconvert --execute --to notebook $(NOTEBOOK_DIR)/qkd_analysis.ipynb
	$(VENV_PYTHON) -m jupyter nbconvert --execute --to notebook $(NOTEBOOK_DIR)/failure_patterns.ipynb

notebook:
	@echo "Starting Jupyter notebook server..."
	$(VENV_PYTHON) -m jupyter notebook $(NOTEBOOK_DIR)

lint:
	@echo "Running code linting..."
	$(VENV_PYTHON) -m flake8 $(SRC_DIR) $(TEST_DIR) --max-line-length=88

format:
	@echo "Formatting code with black..."
	$(VENV_PYTHON) -m black $(SRC_DIR) $(TEST_DIR) $(DEMO_DIR)

clean:
	@echo "Cleaning up generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete
	rm -rf $(VENV)
	rm -rf .pytest_cache
	rm -rf *.egg-info

# QKD specific targets
simulate:
	@echo "Running QKD system simulation..."
	$(PYTHON) $(SRC_DIR)/qkd_simulator.py

detect:
	@echo "Running anomaly detection..."
	$(PYTHON) $(SRC_DIR)/anomaly_detector.py

monitor:
	@echo "Starting security monitoring..."
	$(PYTHON) $(SRC_DIR)/security_monitor.py

# Documentation targets
docs:
	@echo "Generating documentation..."
	$(VENV_PYTHON) -m pdoc --html $(SRC_DIR) --output-dir docs

# Performance benchmarks
benchmark:
	@echo "Running performance benchmarks..."
	$(PYTHON) -m pytest $(TEST_DIR)/test_performance.py -v --benchmark-only
