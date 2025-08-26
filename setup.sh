"""
Setup script for Enterprise ML Platform.

This setup.py provides backward compatibility for systems that don't support 
pyproject.toml yet, but the canonical configuration is in pyproject.toml.
"""

import os
import sys
from pathlib import Path

# Ensure setuptools is available
try:
    from setuptools import setup
except ImportError:
    print("Error: setuptools is required to install this package.")
    print("Please install setuptools: pip install setuptools")
    sys.exit(1)

# Check Python version
if sys.version_info < (3, 9):
    print("Error: Python 3.9 or later is required.")
    print(f"Current version: {sys.version}")
    sys.exit(1)

# Get the directory containing this file
HERE = Path(__file__).parent.absolute()

# Read README for long description
README_PATH = HERE / "README.md"
if README_PATH.exists():
    with open(README_PATH, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "A comprehensive, production-ready machine learning platform designed for enterprise environments"

# Read version from _version.py if it exists, otherwise use setuptools_scm
def get_version():
    """Get version from setuptools_scm or fallback."""
    try:
        from setuptools_scm import get_version
        return get_version(root=HERE)
    except (ImportError, LookupError):
        # Fallback version if setuptools_scm is not available or no git repo
        version_file = HERE / "src" / "enterprise_ml_platform" / "_version.py"
        if version_file.exists():
            with open(version_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("__version__"):
                        return line.split("=")[1].strip().strip('"\'')
        return "0.1.0.dev0"

# Core dependencies (subset of what's in pyproject.toml for backward compatibility)
CORE_REQUIREMENTS = [
    # Core ML libraries
    "numpy>=1.24.0,<2.0.0",
    "pandas>=2.1.0,<3.0.0",
    "scikit-learn>=1.3.0,<2.0.0",
    "xgboost>=2.0.0,<3.0.0",
    "lightgbm>=4.0.0,<5.0.0",
    
    # Data processing
    "pyarrow>=12.0.0,<14.0.0",
    "dask[complete]>=2023.9.0,<2024.0.0",
    
    # ML infrastructure
    "mlflow>=2.8.0,<3.0.0",
    "optuna>=3.4.0,<4.0.0",
    
    # Web framework
    "fastapi>=0.104.0,<1.0.0",
    "uvicorn[standard]>=0.24.0,<1.0.0",
    "pydantic>=2.5.0,<3.0.0",
    
    # Database and storage
    "sqlalchemy>=2.0.0,<3.0.0",
    "asyncpg>=0.29.0,<1.0.0",
    "redis>=5.0.0,<6.0.0",
    
    # Cloud and infrastructure
    "boto3>=1.34.0,<2.0.0",
    "kubernetes>=28.1.0,<29.0.0",
    
    # Monitoring
    "prometheus-client>=0.19.0,<1.0.0",
    "structlog>=23.2.0,<24.0.0",
    
    # Configuration and CLI
    "click>=8.1.0,<9.0.0",
    "pyyaml>=6.0.1,<7.0.0",
    "python-dotenv>=1.0.0,<2.0.0",
    
    # Utilities
    "tqdm>=4.66.0,<5.0.0",
    "requests>=2.31.0,<3.0.0",
    "tenacity>=8.2.0,<9.0.0",
]

# Development dependencies
DEV_REQUIREMENTS = [
    "pytest>=7.4.0,<8.0.0",
    "pytest-asyncio>=0.23.0,<1.0.0",
    "pytest-cov>=4.1.0,<5.0.0",
    "black>=23.12.0,<24.0.0",
    "isort>=5.13.0,<6.0.0",
    "mypy>=1.8.0,<2.0.0",
    "pre-commit>=3.6.0,<4.0.0",
]

# Test dependencies
TEST_REQUIREMENTS = [
    "pytest>=7.4.0,<8.0.0",
    "pytest-asyncio>=0.23.0,<1.0.0",
    "pytest-cov>=4.1.0,<5.0.0",
    "pytest-xdist>=3.5.0,<4.0.0",
    "pytest-mock>=3.12.0,<4.0.0",
    "factory-boy>=3.3.0,<4.0.0",
    "faker>=21.0.0,<22.0.0",
]

# Documentation dependencies
DOC_REQUIREMENTS = [
    "sphinx>=7.2.0,<8.0.0",
    "sphinx-rtd-theme>=2.0.0,<3.0.0",
    "myst-parser>=2.0.0,<3.0.0",
    "mkdocs>=1.5.0,<2.0.0",
    "mkdocs-material>=9.5.0,<10.0.0",
]

# All optional dependencies
ALL_REQUIREMENTS = CORE_REQUIREMENTS + DEV_REQUIREMENTS + TEST_REQUIREMENTS + DOC_REQUIREMENTS

def read_requirements_file(filename):
    """Read requirements from a file, handling comments and blank lines."""
    requirements_path = HERE / "requirements" / filename
    if not requirements_path.exists():
        return []
    
    requirements = []
    with open(requirements_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith("#") and not line.startswith("-"):
                requirements.append(line)
    return requirements

# Try to read from requirements files if they exist
try:
    base_reqs = read_requirements_file("base.txt")
    if base_reqs:
        CORE_REQUIREMENTS = base_reqs
    
    dev_reqs = read_requirements_file("development.txt")
    if dev_reqs:
        DEV_REQUIREMENTS = dev_reqs
        
    test_reqs = read_requirements_file("testing.txt")
    if test_reqs:
        TEST_REQUIREMENTS = test_reqs
        
except Exception as e:
    print(f"Warning: Could not read requirements files: {e}")
    print("Using fallback requirements from setup.py")

def get_packages():
    """Get packages from src directory."""
    try:
        from setuptools import find_packages
        return find_packages(where="src")
    except ImportError:
        # Fallback for older setuptools versions
        import os
        packages = []
        src_dir = HERE / "src"
        if src_dir.exists():
            for root, dirs, files in os.walk(src_dir):
                if "__init__.py" in files:
                    package = os.path.relpath(root, src_dir).replace(os.sep, ".")
                    packages.append(package)
        return packages

# Package data
PACKAGE_DATA = {
    "enterprise_ml_platform": [
        "py.typed",
        "*.yaml",
        "*.yml", 
        "*.json",
        "*.toml",
        "*.txt",
        "templates/*",
        "configs/*",
        "schemas/*",
    ]
}

# Entry points for CLI tools
ENTRY_POINTS = {
    "console_scripts": [
        "mlp=enterprise_ml_platform.cli.main:main",
        "enterprise-ml=enterprise_ml_platform.cli.main:main",
        "mlp-server=enterprise_ml_platform.api.main:start_server",
        "mlp-worker=enterprise_ml_platform.workers.main:start_worker",
        "mlp-monitor=enterprise_ml_platform.monitoring.cli:main",
    ],
    "enterprise_ml_platform.connectors": [
        "s3=enterprise_ml_platform.services.data_ingestion.connectors.s3_connector:S3DataConnector",
        "postgres=enterprise_ml_platform.services.data_ingestion.connectors.postgres_connector:PostgreSQLConnector",
        "kafka=enterprise_ml_platform.services.data_ingestion.connectors.kafka_connector:KafkaStreamConnector",
        "api=enterprise_ml_platform.services.data_ingestion.connectors.api_connector:APIConnector",
    ],
    "enterprise_ml_platform.transformers": [
        "numerical=enterprise_ml_platform.services.feature_engineering.transformers.numerical_transformer:NumericalFeatureTransformer",
        "categorical=enterprise_ml_platform.services.feature_engineering.transformers.categorical_transformer:CategoricalFeatureTransformer",
        "temporal=enterprise_ml_platform.services.feature_engineering.transformers.temporal_transformer:TemporalFeatureTransformer",
        "composite=enterprise_ml_platform.services.feature_engineering.transformers.composite_transformer:CompositeFeatureTransformer",
    ],
    "enterprise_ml_platform.trainers": [
        "xgboost=enterprise_ml_platform.services.model_training.trainers.xgboost_trainer:XGBoostTrainer",
        "lightgbm=enterprise_ml_platform.services.model_training.trainers.lightgbm_trainer:LightGBMTrainer",
        "ensemble=enterprise_ml_platform.services.model_training.trainers.ensemble_trainer:EnsembleTrainer",
        "neural=enterprise_ml_platform.services.model_training.trainers.neural_trainer:NeuralNetworkTrainer",
    ],
    "enterprise_ml_platform.deployers": [
        "kubernetes=enterprise_ml_platform.services.model_deployment.deployers.kubernetes_deployer:KubernetesDeployer",
        "sagemaker=enterprise_ml_platform.services.model_deployment.deployers.sagemaker_deployer:SageMakerDeployer",
        "gcp=enterprise_ml_platform.services.model_deployment.deployers.gcp_deployer:GCPDeployer",
        "azure=enterprise_ml_platform.services.model_deployment.deployers.azure_deployer:AzureDeployer",
    ],
}

# Classifiers for PyPI
CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Information Technology",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: System :: Distributed Computing",
    "Typing :: Typed",
]

# URLs for the project
PROJECT_URLS = {
    "Homepage": "https://github.com/diogoribeiro7/enterprise-ml-platform",
    "Documentation": "https://diogoribeiro7.github.io/enterprise-ml-platform",
    "Repository": "https://github.com/diogoribeiro7/enterprise-ml-platform.git",
    "Issues": "https://github.com/diogoribeiro7/enterprise-ml-platform/issues",
    "Changelog": "https://github.com/diogoribeiro7/enterprise-ml-platform/releases",
    "Discussions": "https://github.com/diogoribeiro7/enterprise-ml-platform/discussions",
    "CI": "https://github.com/diogoribeiro7/enterprise-ml-platform/actions",
    "Author ORCID": "https://orcid.org/0009-0001-2022-7072",
    "Professional Contact": "mailto:dfr@esmad.ipp.pt",
    "Personal Contact": "mailto:diogo.debastos.ribeiro@gmail.com",
}

if __name__ == "__main__":
    setup(
        name="enterprise-ml-platform",
        version=get_version(),
        description="A comprehensive, production-ready machine learning platform designed for enterprise environments",
        long_description=long_description,
        long_description_content_type="text/markdown",
        
        # Author and maintainer info
        author="Diogo Ribeiro",
        author_email="dfr@esmad.ipp.pt",
        maintainer="Diogo Ribeiro",
        maintainer_email="diogo.debastos.ribeiro@gmail.com",
        
        # URLs
        url="https://github.com/diogoribeiro7/enterprise-ml-platform",
        project_urls=PROJECT_URLS,
        
        # Licensing
        license="MIT",
        license_files=["LICENSE"],
        
        # Package structure
        packages=get_packages(),
        package_dir={"": "src"},
        package_data=PACKAGE_DATA,
        include_package_data=True,
        zip_safe=False,
        
        # Dependencies
        python_requires=">=3.9",
        install_requires=CORE_REQUIREMENTS,
        extras_require={
            "dev": DEV_REQUIREMENTS,
            "test": TEST_REQUIREMENTS,
            "docs": DOC_REQUIREMENTS,
            "all": ALL_REQUIREMENTS,
        },
        
        # Entry points
        entry_points=ENTRY_POINTS,
        
        # Metadata
        classifiers=CLASSIFIERS,
        keywords=[
            "machine-learning",
            "mlops", 
            "data-science",
            "ai",
            "enterprise",
            "pipeline",
            "automation",
            "kubernetes",
            "cloud",
            "monitoring"
        ],
        
        # Options
        options={
            "bdist_wheel": {
                "universal": False,
            },
            "egg_info": {
                "tag_build": None,
                "tag_date": None,
            },
        },
        
        # Setuptools-scm configuration
        use_scm_version={
            "write_to": "src/enterprise_ml_platform/_version.py",
            "version_scheme": "release-branch-semver",
            "local_scheme": "dirty-tag",
        },
        setup_requires=["setuptools-scm"],
    )
