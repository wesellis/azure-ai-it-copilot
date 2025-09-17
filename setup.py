"""
Azure AI IT Copilot - Setup Configuration
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="azure-ai-it-copilot",
    version="1.0.0",
    author="Wes Ellis",
    author_email="wes@wesellis.com",
    description="AI-powered IT operations platform for Azure infrastructure management",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wesellis/azure-ai-it-copilot",
    packages=find_packages(exclude=["tests", "tests.*", "docs", "dashboard"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: System Administrators",
        "Topic :: System :: Systems Administration",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.4",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            "black>=23.12.1",
            "ruff>=0.1.11",
            "mypy>=1.8.0",
            "pre-commit>=3.6.0",
        ],
        "ml": [
            "tensorflow>=2.15.0",
            "torch>=2.1.0",
            "transformers>=4.36.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "ai-copilot=ai_orchestrator.cli:main",
            "copilot-server=api.server:run",
        ],
    },
    include_package_data=True,
    package_data={
        "ai_orchestrator": ["*.yaml", "*.json"],
        "automation_engine": ["powershell/*.ps1", "terraform/*.tf"],
        "ml_models": ["*.pkl", "*.joblib", "*.h5"],
    },
)