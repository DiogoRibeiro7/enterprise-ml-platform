#!/bin/bash

# =============================================================================
# Enterprise ML Platform - Development Environment Setup Script
# =============================================================================
# Author: Diogo Ribeiro
# Email: dfr@esmad.ipp.pt | diogo.debastos.ribeiro@gmail.com
# GitHub: https://github.com/diogoribeiro7/enterprise-ml-platform
# ORCID: https://orcid.org/0009-0001-2022-7072
# =============================================================================

set -euo pipefail

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PYTHON_MIN_VERSION="3.9"
PROJECT_NAME="enterprise-ml-platform"
VENV_NAME="venv"
REQUIREMENTS_DIR="requirements"

# =============================================================================
# LOGGING FUNCTIONS
# =============================================================================

log_header() {
    echo -e "${PURPLE}════════════════════════════════════════${NC}"
    echo -e "${PURPLE}  $1${NC}"
    echo -e "${PURPLE}════════════════════════════════════════${NC}"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

# =============================================================================
# SYSTEM VALIDATION FUNCTIONS
# =============================================================================

check_os() {
    log_step "Checking operating system compatibility..."
    
    local os_name=$(uname -s)
    case $os_name in
        Linux*)
            log_info "Detected Linux: $os_name"
            ;;
        Darwin*)
            log_info "Detected macOS: $os_name"
            ;;
        MINGW*|CYGWIN*|MSYS*)
            log_info "Detected Windows with Unix-like environment: $os_name"
            ;;
        *)
            log_error "Unsupported operating system: $os_name"
            log_error "This script supports Linux, macOS, and Windows with Unix-like environment"
            exit 1
            ;;
    esac
    
    log_success "Operating system check completed"
}

check_python() {
    log_step "Checking Python installation..."
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed or not in PATH"
        log_error "Please install Python ${PYTHON_MIN_VERSION} or later"
        log_error "Visit: https://www.python.org/downloads/"
        exit 1
    fi
    
    local python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    local python_major=$(echo "$python_version" | cut -d'.' -f1)
    local python_minor=$(echo "$python_version" | cut -d'.' -f2)
    
    log_info "Found Python version: $python_version"
    
    if [[ $python_major -lt 3 ]] || [[ $python_major -eq 3 && $python_minor -lt 9 ]]; then
        log_error "Python ${PYTHON_MIN_VERSION} or later is required"
        log_error "Current version: $python_version"
        log_error "Please upgrade your Python installation"
        exit 1
    fi
    
    log_success "Python version check passed"
}

check_system_dependencies() {
    log_step "Checking system dependencies..."
    
    local missing_deps=()
    local required_tools=("git" "curl" "wget" "make")
    
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            missing_deps+=("$tool")
            log_warning "Missing required tool: $tool"
        else
            log_info "✓ Found: $tool"
        fi
    done
    
    # Check for Docker (optional but recommended)
    if ! command -v docker &> /dev/null; then
        log_warning "Docker is not installed - some features will be limited"
        log_info "Install Docker from: https://docs.docker.com/get-docker/"
    else
        log_info "✓ Found: docker"
        if ! docker info &> /dev/null 2>&1; then
            log_warning "Docker daemon is not running"
        else
            log_info "✓ Docker daemon is running"
        fi
    fi
    
    # Check for kubectl (optional)
    if ! command -v kubectl &> /dev/null; then
        log_warning "kubectl not found - Kubernetes features will be limited"
        log_info "Install kubectl from: https://kubernetes.io/docs/tasks/tools/"
    else
        log_info "✓ Found: kubectl"
    fi
    
    # Check for Poetry (optional)
    if ! command -v poetry &> /dev/null; then
        log_info "Poetry not found - will use pip for dependency management"
        log_info "Install Poetry from: https://python-poetry.org/docs/"
    else
        log_info "✓ Found: poetry"
    fi
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_error "Missing required dependencies: ${missing_deps[*]}"
        log_error "Please install them using your system package manager:"
        echo
        case $(uname -s) in
            Linux*)
                log_info "Ubuntu/Debian: sudo apt-get install ${missing_deps[*]}"
                log_info "CentOS/RHEL: sudo yum install ${missing_deps[*]}"
                log_info "Arch Linux: sudo pacman -S ${missing_deps[*]}"
                ;;
            Darwin*)
                log_info "macOS (Homebrew): brew install ${missing_deps[*]}"
                log_info "macOS (MacPorts): sudo port install ${missing_deps[*]}"
                ;;
        esac
        exit 1
    fi
    
    log_success "All required system dependencies are available"
}

# =============================================================================
# PYTHON ENVIRONMENT SETUP
# =============================================================================

setup_virtual_environment() {
    log_step "Setting up Python virtual environment..."
    
    if [[ -d "$VENV_NAME" ]]; then
        log_warning "Virtual environment '$VENV_NAME' already exists"
        echo
        read -p "Do you want to recreate it? This will delete the existing environment. [y/N]: " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Removing existing virtual environment..."
            rm -rf "$VENV_NAME"
        else
            log_info "Using existing virtual environment"
            if [[ -f "$VENV_NAME/bin/activate" ]]; then
                source "$VENV_NAME/bin/activate"
                log_info "Activated existing virtual environment"
                return 0
            else
                log_error "Existing virtual environment appears corrupted"
                log_info "Removing and recreating..."
                rm -rf "$VENV_NAME"
            fi
        fi
    fi
    
    log_info "Creating virtual environment '$VENV_NAME'..."
    python3 -m venv "$VENV_NAME"
    
    log_info "Activating virtual environment..."
    source "$VENV_NAME/bin/activate"
    
    log_info "Upgrading pip, setuptools, and wheel..."
    pip install --upgrade pip setuptools wheel
    
    # Verify virtual environment
    local venv_python=$(which python)
    log_info "Virtual environment Python: $venv_python"
    
    log_success "Virtual environment created and activated"
}

install_python_dependencies() {
    log_step "Installing Python dependencies..."
    
    # Ensure we're in the virtual environment
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        log_info "Activating virtual environment..."
        source "$VENV_NAME/bin/activate"
    fi
    
    log_info "Virtual environment active: $VIRTUAL_ENV"
    
    # Install setuptools-scm first for version management
    log_info "Installing setuptools-scm for version management..."
    pip install setuptools-scm
    
    # Check for Poetry first
    if command -v poetry &> /dev/null && [[ -f "pyproject.toml" ]]; then
        log_info "Using Poetry for dependency management..."
        poetry install --with dev,test,docs
    
    # Use pip with pyproject.toml
    elif [[ -f "pyproject.toml" ]]; then
        log_info "Installing package in development mode using pyproject.toml..."
        pip install -e ".[dev,test,docs]"
    
    # Use pip with setup.py
    elif [[ -f "setup.py" ]]; then
        log_info "Installing package in development mode using setup.py..."
        pip install -e ".[dev,test,docs]"
    
    # Fallback to requirements files
    else
        log_warning "No pyproject.toml or setup.py found, installing from requirements files..."
        local req_files=("base.txt" "development.txt" "testing.txt")
        
        for req_file in "${req_files[@]}"; do
            local req_path="$REQUIREMENTS_DIR/$req_file"
            if [[ -f "$req_path" ]]; then
                log_info "Installing from $req_path..."
                pip install -r "$req_path"
            else
                log_warning "Requirements file not found: $req_path"
            fi
        done
    fi
    
    log_success "Python dependencies installed successfully"
}

setup_pre_commit() {
    log_step "Setting up pre-commit hooks..."
    
    if [[ -f ".pre-commit-config.yaml" ]]; then
        log_info "Installing pre-commit hooks..."
        pre-commit install
        pre-commit install --hook-type commit-msg
        
        log_info "Running pre-commit on all files (this may take a while)..."
        pre-commit run --all-files || log_warning "Some pre-commit checks failed (this is normal for first run)"
        
        log_success "Pre-commit hooks installed and configured"
    else
        log_warning "No .pre-commit-config.yaml found, skipping pre-commit setup"
        log_info "You can create one later with: pre-commit sample-config > .pre-commit-config.yaml"
    fi
}

# =============================================================================
# PROJECT STRUCTURE SETUP
# =============================================================================

create_directories() {
    log_step "Creating project directory structure..."
    
    local dirs=(
        "logs"
        "data/raw"
        "data/processed"
        "data/external" 
        "data/interim"
        "models/trained"
        "models/exported"
        "notebooks/exploratory"
        "notebooks/reports"
        "outputs/reports"
        "outputs/figures"
        "outputs/models"
        "temp"
        ".secrets"
        "tests/fixtures"
        "tests/data"
    )
    
    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log_info "Created directory: $dir"
        else
            log_info "Directory already exists: $dir"
        fi
    done
    
    # Create .gitkeep files for empty directories that should be tracked
    local keep_dirs=("logs" "data/raw" "data/processed" "data/external" "data/interim" "temp" "outputs/reports" "outputs/figures")
    for dir in "${keep_dirs[@]}"; do
        if [[ -d "$dir" && ! -f "$dir/.gitkeep" ]]; then
            touch "$dir/.gitkeep"
            log_info "Added .gitkeep to: $dir"
        fi
    done
    
    log_success "Project directory structure created"
}

setup_configuration() {
    log_step "Setting up configuration files..."
    
    local config_dir="config"
    
    # Create config directory if it doesn't exist
    if [[ ! -d "$config_dir" ]]; then
        mkdir -p "$config_dir"
        log_info "Created config directory"
    fi
    
    # Configuration file templates to copy
    local config_templates=(
        "development.yaml.example:development.yaml"
        "production.yaml.example:production.yaml"
        "testing.yaml.example:testing.yaml"
        ".env.example:.env"
        "logging.yaml.example:logging.yaml"
        "database.yaml.example:database.yaml"
    )
    
    for template in "${config_templates[@]}"; do
        local src="${template%:*}"
        local dst="${template#*:}"
        local src_path="$config_dir/$src"
        local dst_path="$config_dir/$dst"
        
        if [[ -f "$src_path" && ! -f "$dst_path" ]]; then
            cp "$src_path" "$dst_path"
            log_info "Created $dst_path from template"
        elif [[ ! -f "$src_path" ]]; then
            # Create basic configuration files if templates don't exist
            case "$dst" in
                "development.yaml")
                    create_basic_dev_config "$dst_path"
                    ;;
                ".env")
                    create_basic_env_file "$dst_path"
                    ;;
                "logging.yaml")
                    create_basic_logging_config "$dst_path"
                    ;;
            esac
        fi
    done
    
    # Create .env in project root if it doesn't exist
    if [[ ! -f ".env" ]]; then
        if [[ -f "config/.env" ]]; then
            cp "config/.env" ".env"
            log_info "Created .env file in project root"
        else
            create_basic_env_file ".env"
        fi
    fi
    
    log_success "Configuration files setup completed"
}

create_basic_dev_config() {
    local config_file="$1"
    cat > "$config_file" << 'EOF'
# Development Configuration for Enterprise ML Platform
# Author: Diogo Ribeiro

environment: development
debug: true

# Database Configuration
database:
  host: localhost
  port: 5432
  name: ml_platform_dev
  user: ml_user
  password: dev_password

# Redis Configuration  
redis:
  host: localhost
  port: 6379
  db: 0

# MLflow Configuration
mlflow:
  tracking_uri: http://localhost:5000
  experiment_name: development_experiment

# Logging
logging:
  level: DEBUG
  format: detailed

# API Configuration
api:
  host: 0.0.0.0
  port: 8000
  reload: true
EOF
    log_info "Created basic development configuration: $config_file"
}

create_basic_env_file() {
    local env_file="$1"
    cat > "$env_file" << 'EOF'
# Environment Variables for Enterprise ML Platform
# Author: Diogo Ribeiro

# Environment
ENVIRONMENT=development
DEBUG=true

# Database
DATABASE_URL=postgresql://ml_user:dev_password@localhost:5432/ml_platform_dev
REDIS_URL=redis://localhost:6379/0

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# API Keys (Add your actual keys here)
# AWS_ACCESS_KEY_ID=your_aws_access_key
# AWS_SECRET_ACCESS_KEY=your_aws_secret_key
# OPENAI_API_KEY=your_openai_api_key

# Security
SECRET_KEY=your-secret-key-change-this-in-production
JWT_SECRET=your-jwt-secret-change-this-in-production

# Logging
LOG_LEVEL=DEBUG
EOF
    log_info "Created basic environment file: $env_file"
}

create_basic_logging_config() {
    local logging_file="$1"
    cat > "$logging_file" << 'EOF'
version: 1
disable_existing_loggers: false

formatters:
  detailed:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  simple:
    format: '%(levelname)s - %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    level: DEBUG
    formatter: detailed
    stream: ext://sys.stdout
    
  file:
    class: logging.FileHandler
    level: INFO
    formatter: detailed
    filename: logs/app.log

loggers:
  enterprise_ml_platform:
    level: DEBUG
    handlers: [console, file]
    propagate: false

root:
  level: INFO
  handlers: [console]
EOF
    log_info "Created basic logging configuration: $logging_file"
}

# =============================================================================
# DEVELOPMENT SERVICES SETUP
# =============================================================================

setup_development_services() {
    log_step "Setting up development services..."
    
    if ! command -v docker &> /dev/null; then
        log_warning "Docker not available, skipping development services setup"
        log_info "Install Docker to use automated service setup"
        return 0
    fi
    
    if ! docker info &> /dev/null 2>&1; then
        log_warning "Docker daemon not running, skipping development services"
        log_info "Start Docker daemon to use automated service setup"
        return 0
    fi
    
    if [[ -f "docker-compose.yml" ]]; then
        log_info "Found docker-compose.yml, starting development services..."
        
        # Start core services
        local services=("postgres" "redis" "mlflow")
        for service in "${services[@]}"; do
            if docker-compose ps | grep -q "$service"; then
                log_info "Service $service is already running"
            else
                log_info "Starting service: $service"
                docker-compose up -d "$service" || log_warning "Failed to start $service"
            fi
        done
        
        # Wait for services to be ready
        log_info "Waiting for services to be ready..."
        sleep 15
        
        # Verify services
        if docker-compose ps | grep -q "postgres.*Up"; then
            log_success "PostgreSQL is running"
        else
            log_warning "PostgreSQL may not be running properly"
        fi
        
        if docker-compose ps | grep -q "redis.*Up"; then
            log_success "Redis is running"
        else
            log_warning "Redis may not be running properly"
        fi
        
        log_success "Development services setup completed"
        
    else
        log_warning "No docker-compose.yml found"
        log_info "Create a docker-compose.yml file to use automated service setup"
        create_basic_docker_compose
    fi
}

create_basic_docker_compose() {
    log_info "Creating basic docker-compose.yml..."
    
    cat > "docker-compose.yml" << 'EOF'
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: ml_platform_dev
      POSTGRES_USER: ml_user
      POSTGRES_PASSWORD: dev_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ml_user -d ml_platform_dev"]
      interval: 30s
      timeout: 10s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 5

  mlflow:
    image: python:3.9-slim
    command: >
      bash -c "pip install mlflow psycopg2-binary &&
               mlflow server --backend-store-uri postgresql://ml_user:dev_password@postgres:5432/ml_platform_dev --default-artifact-root ./artifacts --host 0.0.0.0 --port 5000"
    ports:
      - "5000:5000"
    depends_on:
      postgres:
        condition: service_healthy
    volumes:
      - mlflow_artifacts:/app/artifacts

volumes:
  postgres_data:
  redis_data:
  mlflow_artifacts:
EOF
    
    log_success "Created basic docker-compose.yml"
}

# =============================================================================
# INSTALLATION VERIFICATION
# =============================================================================

verify_installation() {
    log_step "Verifying installation..."
    
    # Check if we're in virtual environment
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        source "$VENV_NAME/bin/activate"
    fi
    
    # Check if the package can be imported
    log_info "Testing package import..."
    if python3 -c "import enterprise_ml_platform; print(f'Package version: {getattr(enterprise_ml_platform, \"__version__\", \"development\")}')" 2>/dev/null; then
        log_success "Package imports successfully"
    else
        log_warning "Package import failed (this might be expected in development mode)"
    fi
    
    # Check CLI availability
    log_info "Testing CLI availability..."
    if command -v mlp &> /dev/null; then
        log_success "CLI tool 'mlp' is available"
        if mlp --version 2>/dev/null; then
            log_info "CLI version check successful"
        else
            log_warning "CLI version check failed, but CLI is accessible"
        fi
    else
        log_warning "CLI tool 'mlp' not found in PATH"
        log_info "Try: source venv/bin/activate && pip install -e ."
    fi
    
    # Run basic tests if available
    if [[ -d "tests" ]] && command -v pytest &> /dev/null; then
        log_info "Running basic smoke tests..."
        if timeout 60 pytest tests/ -v --tb=short -x -q --maxfail=3 2>/dev/null; then
            log_success "Basic tests passed"
        else
            log_warning "Some tests failed or timed out, but development environment should still work"
        fi
    else
        log_info "Skipping tests (pytest not available or no tests directory)"
    fi
    
    log_success "Installation verification completed"
}

# =============================================================================
# DOCUMENTATION GENERATION
# =============================================================================

generate_dev_documentation() {
    log_step "Generating development documentation..."
    
    # Create comprehensive development guide
    cat > "DEVELOPMENT.md" << 'EOF'
# Development Guide

## Quick Start Commands

### Environment Activation
```bash
# Activate virtual environment
source venv/bin/activate  

# On Windows
venv\Scripts\activate
```

### Development Server
```bash
# Start API server in development mode
mlp server --dev

# Start with specific config
mlp server --config config/development.yaml

# Start background services
docker-compose up -d
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=enterprise_ml_platform

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Skip slow tests

# Run tests in parallel
pytest -n auto
```

### Code Quality
```bash
# Format code
black .
isort .

# Type checking
mypy src/

# Linting
flake8 src/
pylint src/

# Security scan
bandit -r src/

# Run all pre-commit hooks
pre-commit run --all-files
```

## Pipeline Commands

### Basic Pipeline Operations
```bash
# Run complete pipeline
mlp pipeline run --config config/development.yaml

# Run specific stages
mlp pipeline run --stages data_ingestion,feature_engineering

# Check pipeline status
mlp pipeline status --run-id <run-id>
```

### Model Operations
```bash
# List models
mlp model list

# Deploy model
mlp model deploy --name fraud-detector --version v1.0.0 --platform kubernetes

# Monitor model
mlp model monitor --name fraud-detector --metrics accuracy,drift
```

### Data Operations
```bash
# Validate data
mlp data validate --source s3://my-bucket/data --rules config/data_quality.yaml

# Check data drift
mlp data drift --reference data/reference.csv --current data/current.csv
```

## Development Workflow

### 1. Feature Development
1. Create feature branch: `git checkout -b feature/amazing-feature`
2. Make changes and add tests
3. Run quality checks: `pre-commit run --all-files`
4. Commit changes: `git commit -m "Add amazing feature"`
5. Push and create PR: `git push origin feature/amazing-feature`

### 2. Testing Strategy
- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows
- **Performance Tests**: Benchmark critical paths

### 3. Code Standards
- **Python**: Follow PEP 8, use type hints, docstrings required
- **Testing**: Minimum 80% code coverage
- **Documentation**: Update docs for user-facing changes
- **Security**: No secrets in code, run security scans

## Configuration

### Environment Variables
Edit `.env` file or set environment variables:

```bash
# Core settings
ENVIRONMENT=development
DEBUG=true

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/db

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# Cloud credentials (optional)
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
```

### Configuration Files
- `config/development.yaml` - Development settings
- `config/production.yaml` - Production settings
- `config/logging.yaml` - Logging configuration
- `.env` - Environment variables

## Services

### Local Development Stack
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Reset data
docker-compose down -v
```

### Service Endpoints
- **API Server**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **MLflow UI**: http://localhost:5000
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Reinstall in development mode
pip install -e .

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

#### Database Connection
```bash
# Check if PostgreSQL is running
docker-compose ps postgres

# Test connection
psql -h localhost -U ml_user -d ml_platform_dev
```

#### Virtual Environment Issues
```bash
# Recreate virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev,test,docs]"
```

#### Docker Issues
```bash
# Check Docker status
docker info

# Restart Docker services
docker-compose restart

# View service logs
docker-compose logs <service_name>
```

### Getting Help

1. Check the logs in `logs/` directory
2. Review configuration files
3. Run diagnostic commands:
   ```bash
   mlp doctor  # System health check
   mlp config validate  # Configuration validation
   ```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.

For questions or issues, contact:
- Professional: dfr@esmad.ipp.pt
- Personal: diogo.debastos.ribeiro@gmail.com
- ORCID: https://orcid.org/0009-0001-2022-7072
EOF
    
    # Create diagnostic script
    cat > "scripts/diagnose.sh" << 'EOF'
#!/bin/bash
# System diagnostic script

echo "=== Enterprise ML Platform Diagnostics ==="
echo
echo "System Information:"
echo "- OS: $(uname -s -r)"
echo "- Python: $(python3 --version)"
echo "- Virtual Env: ${VIRTUAL_ENV:-Not activated}"
echo
echo "Service Status:"
if command -v docker &> /dev/null; then
    echo "- Docker: Available"
    if docker info &> /dev/null; then
        echo "- Docker Daemon: Running"
    else
        echo "- Docker Daemon: Not running"
    fi
else
    echo "- Docker: Not available"
fi

if command -v kubectl &> /dev/null; then
    echo "- kubectl: Available"
else
    echo "- kubectl: Not available"
fi

echo
echo "Python Packages:"
pip list | grep -E "(mlflow|pandas|scikit-learn|fastapi)" || echo "Core packages not found"

echo
echo "Configuration Files:"
for file in .env config/development.yaml docker-compose.yml; do
    if [[ -f "$file" ]]; then
        echo "- $file: ✓"
    else
        echo "- $file: Missing"
    fi
done
EOF
    
    chmod +x "scripts/diagnose.sh"
    
    log_success "Development documentation generated"
}

# =============================================================================
# FINAL SETUP STEPS
# =============================================================================

display_next_steps() {
    log_header "Setup Complete! 🎉"
    
    echo -e "${GREEN}Your Enterprise ML Platform development environment is ready!${NC}"
    echo
    echo -e "${CYAN}Next Steps:${NC}"
    echo -e "  ${BLUE}1.${NC} Activate the virtual environment:"
    echo -e "     ${YELLOW}source $VENV_NAME/bin/activate${NC}"
    echo
    echo -e "  ${BLUE}2.${NC} Verify the installation:"
    echo -e "     ${YELLOW}mlp --version${NC}"
    echo -e "     ${YELLOW}./scripts/diagnose.sh${NC}"
    echo
    echo -e "  ${BLUE}3.${NC} Start development services:"
    echo -e "     ${YELLOW}docker-compose up -d${NC}"
    echo
    echo -e "  ${BLUE}4.${NC} Start the development server:"
    echo -e "     ${YELLOW}mlp server --dev${NC}"
    echo
    echo -e "  ${BLUE}5.${NC} Run tests to verify everything works:"
    echo -e "     ${YELLOW}pytest tests/${NC}"
    echo
    echo -e "${CYAN}Important Files & Directories:${NC}"
    echo -e "  • ${YELLOW}config/development.yaml${NC} - Development configuration"
    echo -e "  • ${YELLOW}.env${NC} - Environment variables"
    echo -e "  • ${YELLOW}DEVELOPMENT.md${NC} - Comprehensive development guide"
    echo -e "  • ${YELLOW}docker-compose.yml${NC} - Development services"
    echo -e "  • ${YELLOW}scripts/diagnose.sh${NC} - System diagnostic tool"
    echo
    echo -e "${CYAN}Service Endpoints:${NC}"
    echo -e "  • ${YELLOW}API Server:${NC} http://localhost:8000"
    echo -e "  • ${YELLOW}API Documentation:${NC} http://localhost:8000/docs"
    echo -e "  • ${YELLOW}MLflow UI:${NC} http://localhost:5000"
    echo -e "  • ${YELLOW}PostgreSQL:${NC} localhost:5432"
    echo -e "  • ${YELLOW}Redis:${NC} localhost:6379"
    echo
    echo -e "${CYAN}Author Information:${NC}"
    echo -e "  • ${YELLOW}Name:${NC} Diogo Ribeiro"
    echo -e "  • ${YELLOW}Professional:${NC} dfr@esmad.ipp.pt"
    echo -e "  • ${YELLOW}Personal:${NC} diogo.debastos.ribeiro@gmail.com"
    echo -e "  • ${YELLOW}ORCID:${NC} https://orcid.org/0009-0001-2022-7072"
    echo -e "  • ${YELLOW}GitHub:${NC} https://github.com/diogoribeiro7/enterprise-ml-platform"
    echo
    echo -e "${GREEN}Happy coding! 🚀${NC}"
}

# =============================================================================
# COMMAND LINE ARGUMENT HANDLING
# =============================================================================

show_help() {
    echo "Enterprise ML Platform - Development Environment Setup"
    echo "Author: Diogo Ribeiro <dfr@esmad.ipp.pt>"
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --skip-deps         Skip Python dependency installation"
    echo "  --skip-services     Skip Docker services setup"
    echo "  --skip-tests        Skip test execution during verification"
    echo "  --skip-precommit    Skip pre-commit hooks installation"
    echo "  --force-recreate    Force recreation of virtual environment"
    echo "  --minimal           Minimal installation (core dependencies only)"
    echo "  --verbose           Enable verbose output"
    echo "  --dry-run           Show what would be done without executing"
    echo "  -h, --help          Show this help message"
    echo
    echo "Examples:"
    echo "  $0                          # Full setup"
    echo "  $0 --skip-services          # Setup without Docker services"
    echo "  $0 --minimal --skip-tests   # Minimal setup for CI/CD"
    echo "  $0 --dry-run                # Preview what will be done"
    echo
    echo "For more information, visit:"
    echo "https://github.com/diogoribeiro7/enterprise-ml-platform"
}

# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

main() {
    # Parse command line arguments
    local skip_deps=false
    local skip_services=false
    local skip_tests=false
    local skip_precommit=false
    local force_recreate=false
    local minimal=false
    local verbose=false
    local dry_run=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-deps)
                skip_deps=true
                shift
                ;;
            --skip-services)
                skip_services=true
                shift
                ;;
            --skip-tests)
                skip_tests=true
                shift
                ;;
            --skip-precommit)
                skip_precommit=true
                shift
                ;;
            --force-recreate)
                force_recreate=true
                shift
                ;;
            --minimal)
                minimal=true
                shift
                ;;
            --verbose)
                verbose=true
                set -x  # Enable verbose mode
                shift
                ;;
            --dry-run)
                dry_run=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                echo
                show_help
                exit 1
                ;;
        esac
    done
    
    # Display banner
    echo -e "${PURPLE}"
    echo "════════════════════════════════════════════════════════════"
    echo "  🚀 Enterprise ML Platform - Development Environment Setup"
    echo "════════════════════════════════════════════════════════════"
    echo -e "${NC}"
    echo -e "${CYAN}Author:${NC} Diogo Ribeiro"
    echo -e "${CYAN}Email:${NC} dfr@esmad.ipp.pt | diogo.debastos.ribeiro@gmail.com"
    echo -e "${CYAN}ORCID:${NC} https://orcid.org/0009-0001-2022-7072"
    echo -e "${CYAN}GitHub:${NC} https://github.com/diogoribeiro7/enterprise-ml-platform"
    echo
    
    # Dry run mode
    if [[ "$dry_run" == true ]]; then
        log_info "DRY RUN MODE - No actual changes will be made"
        echo
        log_info "Would execute the following steps:"
        echo "  1. Check operating system compatibility"
        echo "  2. Verify Python installation (>=3.9)"
        echo "  3. Check system dependencies (git, curl, wget, make)"
        echo "  4. Setup Python virtual environment"
        [[ "$skip_deps" == false ]] && echo "  5. Install Python dependencies"
        [[ "$skip_precommit" == false ]] && echo "  6. Setup pre-commit hooks"
        echo "  7. Create project directory structure"
        echo "  8. Setup configuration files"
        [[ "$skip_services" == false ]] && echo "  9. Setup development services (Docker)"
        echo "  10. Verify installation"
        echo "  11. Generate development documentation"
        echo
        log_info "Run without --dry-run to execute these steps"
        exit 0
    fi
    
    # Check if we're in the right directory
    if [[ ! -f "pyproject.toml" && ! -f "setup.py" ]]; then
        log_error "This doesn't appear to be the project root directory"
        log_error "Please run this script from the project root where pyproject.toml or setup.py is located"
        log_info "Current directory: $(pwd)"
        exit 1
    fi
    
    # Force recreate virtual environment if requested
    if [[ "$force_recreate" == true && -d "$VENV_NAME" ]]; then
        log_info "Force recreation requested, removing existing virtual environment..."
        rm -rf "$VENV_NAME"
    fi
    
    # Execute setup steps
    local start_time=$(date +%s)
    
    # Core system checks
    check_os
    check_python  
    check_system_dependencies
    
    # Python environment setup
    setup_virtual_environment
    
    if [[ "$skip_deps" == false ]]; then
        install_python_dependencies
        
        if [[ "$skip_precommit" == false ]]; then
            setup_pre_commit
        else
            log_info "Skipping pre-commit hooks installation"
        fi
    else
        log_info "Skipping Python dependency installation"
    fi
    
    # Project structure
    create_directories
    setup_configuration
    
    # Development services
    if [[ "$skip_services" == false ]]; then
        setup_development_services
    else
        log_info "Skipping Docker services setup"
    fi
    
    # Verification and documentation
    if [[ "$skip_tests" == false ]]; then
        verify_installation
    else
        log_info "Skipping installation verification"
    fi
    
    generate_dev_documentation
    
    # Calculate setup time
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))
    
    echo
    log_success "Setup completed in ${minutes}m ${seconds}s"
    
    # Display next steps
    display_next_steps
}

# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

# Only run main if script is executed directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
