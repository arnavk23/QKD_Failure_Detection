#!/bin/bash

# QKD Failure Detection System - Environment Setup Script

set -e  # Exit on any error

echo "🚀 Setting up QKD Failure Detection System environment..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python version
print_status "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    print_success "Python version $python_version is supported"
else
    print_error "Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

# Create virtual environment
print_status "Creating virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
print_status "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    print_success "Dependencies installed from requirements.txt"
else
    print_error "requirements.txt not found"
    exit 1
fi

# Install development dependencies
print_status "Installing development dependencies..."
pip install pytest pytest-cov black isort flake8 mypy pylint bandit safety pre-commit

# Create necessary directories
print_status "Creating project directories..."
mkdir -p data/{raw,processed,examples}
mkdir -p plots/{generated,temp,examples}
mkdir -p results/{experiments,benchmarks,reports}
mkdir -p logs
mkdir -p models
mkdir -p docs/{api,guides,images}

# Set up pre-commit hooks
print_status "Setting up pre-commit hooks..."
if command -v pre-commit &> /dev/null; then
    pre-commit install
    print_success "Pre-commit hooks installed"
else
    print_warning "pre-commit not available, skipping hooks setup"
fi

# Run initial tests
print_status "Running initial tests..."
if [ -d "tests" ]; then
    python -m pytest tests/ -v --tb=short
    print_success "Initial tests completed"
else
    print_warning "Tests directory not found, skipping test run"
fi

# Create environment file
print_status "Creating environment configuration..."
cat > .env << EOF
# QKD System Environment Configuration
QKD_ENV=development
QKD_LOG_LEVEL=DEBUG
QKD_DATA_PATH=data
QKD_RESULTS_PATH=results
QKD_CONFIG_PATH=config/default_config.yaml
PYTHONPATH=src
EOF

print_success "Environment file created"

# Display summary
echo ""
echo "==============================================="
print_success "QKD Failure Detection System setup complete!"
echo "==============================================="
echo ""
echo "📁 Project structure created"
echo "🐍 Virtual environment: .venv"
echo "📦 Dependencies installed"
echo "🧪 Tests verified"
echo "⚙️ Configuration ready"
echo ""
echo "Next steps:"
echo "1. Activate environment: source .venv/bin/activate"
echo "2. Run demos: python demos/run_all_demos.py"
echo "3. Start Jupyter: jupyter notebook notebooks/"
echo "4. Run tests: make test"
echo ""
print_success "Happy coding! 🚀"
