# Contributing to Azure AI IT Copilot

First off, thank you for considering contributing to Azure AI IT Copilot! 🎉 It's people like you that make this platform amazing for the IT operations community.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Submitting Changes](#submitting-changes)
- [Style Guides](#style-guides)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [wes@wesellis.com](mailto:wes@wesellis.com).

## Getting Started

### 🎯 Ways to Contribute

- **🐛 Bug Reports**: Help us identify and fix issues
- **💡 Feature Requests**: Suggest new capabilities
- **📝 Documentation**: Improve our guides and examples
- **🔧 Code**: Implement features, fix bugs, improve performance
- **🧪 Testing**: Add test cases and improve coverage
- **🎨 UI/UX**: Enhance the dashboard and user experience
- **🤖 AI Agents**: Create new specialized agents
- **🔌 Integrations**: Add support for new services

### 🏁 Quick Start for Contributors

1. **Fork the repository**
2. **Clone your fork**:
   ```bash
   git clone https://github.com/yourusername/azure-ai-it-copilot.git
   cd azure-ai-it-copilot
   ```
3. **Set up development environment**:
   ```bash
   ./setup.sh
   ```
4. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## How Can I Contribute?

### 🐛 Reporting Bugs

Before creating bug reports, please check the [issue list](https://github.com/yourusername/azure-ai-it-copilot/issues) as you might find out that you don't need to create one.

**When filing a bug report, please include:**

- **Summary**: A clear and descriptive title
- **Environment**: OS, Python version, Azure region, etc.
- **Steps to Reproduce**: Detailed steps to reproduce the issue
- **Expected Behavior**: What you expected to happen
- **Actual Behavior**: What actually happened
- **Screenshots**: If applicable
- **Logs**: Relevant log output (sanitize sensitive information)

**Use the bug report template:**
```markdown
**Environment:**
- OS: [e.g., Ubuntu 22.04, Windows 11, macOS 13]
- Python Version: [e.g., 3.11.5]
- Azure AI IT Copilot Version: [e.g., 1.0.0]

**Steps to Reproduce:**
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected Behavior:**
A clear description of what you expected to happen.

**Actual Behavior:**
A clear description of what actually happened.

**Additional Context:**
Add any other context about the problem here.
```

### 💡 Suggesting Features

We love feature suggestions! Before creating enhancement suggestions, please check the [issue list](https://github.com/yourusername/azure-ai-it-copilot/issues) and [roadmap](README.md#roadmap).

**When suggesting a feature:**

- **Use a clear and descriptive title**
- **Provide a step-by-step description** of the suggested enhancement
- **Explain why this would be useful** to the community
- **Provide examples** of how the feature would be used
- **Consider the scope** - start with small, focused features

### 📝 Improving Documentation

Documentation improvements are always welcome! This includes:

- **README updates**: Keep installation and usage instructions current
- **API documentation**: Document new endpoints and parameters
- **Code comments**: Add helpful comments to complex code
- **Examples**: Create practical usage examples
- **Tutorials**: Write step-by-step guides
- **Architecture docs**: Explain system design decisions

## Development Setup

### Prerequisites

- **Python 3.11+**
- **Node.js 18+** (for dashboard)
- **Docker** (optional, for containerized development)
- **Azure CLI** (for Azure integration)
- **Git**

### Environment Setup

1. **Clone and setup**:
   ```bash
   git clone https://github.com/yourusername/azure-ai-it-copilot.git
   cd azure-ai-it-copilot
   ./setup.sh
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Install dependencies**:
   ```bash
   # Python dependencies
   pip install -r requirements.txt
   pip install -r requirements-dev.txt

   # Dashboard dependencies
   cd dashboard && npm install && cd ..
   ```

4. **Run tests**:
   ```bash
   make test
   ```

5. **Start development server**:
   ```bash
   make dev
   ```

### Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/amazing-new-feature
   ```

2. **Make your changes**:
   - Write code following our [style guide](#style-guides)
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**:
   ```bash
   make test
   make lint
   make type-check
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add amazing new feature"
   ```

5. **Push and create PR**:
   ```bash
   git push origin feature/amazing-new-feature
   ```

## Submitting Changes

### Pull Request Process

1. **Update documentation** for any public API changes
2. **Add tests** that cover your changes
3. **Ensure all tests pass** and maintain code coverage
4. **Update the changelog** if your change is user-facing
5. **Follow the PR template** when creating your pull request

### PR Guidelines

- **One feature per PR**: Keep changes focused and atomic
- **Clear descriptions**: Explain what changes you made and why
- **Reference issues**: Link to related issues with "Fixes #123"
- **Breaking changes**: Clearly mark any breaking changes
- **Screenshots**: Include before/after screenshots for UI changes

### PR Template

```markdown
## Summary
Brief description of what this PR does.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Changes Made
- List the specific changes made
- Be as detailed as necessary

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Screenshots (if applicable)
Add screenshots to help explain your changes.

## Related Issues
Closes #123
```

## Style Guides

### Python Code Style

We follow PEP 8 with some modifications:

- **Line length**: 88 characters (Black default)
- **Imports**: Use `isort` for import organization
- **Type hints**: Use type hints for all function signatures
- **Docstrings**: Use Google-style docstrings

**Tools we use:**
```bash
# Auto-formatting
black .
isort .

# Linting
ruff check .
mypy .

# All together
make lint
```

**Example function:**
```python
async def process_command(
    self,
    command: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Process a natural language command.

    Args:
        command: The natural language command to process
        context: Optional context information

    Returns:
        The command execution result

    Raises:
        ValueError: If command is invalid
    """
    if not command.strip():
        raise ValueError("Command cannot be empty")

    # Implementation here
    return {"status": "success"}
```

### JavaScript/TypeScript Style

For the React dashboard:

- **Prettier** for formatting
- **ESLint** for linting
- **TypeScript** for type safety
- **Functional components** with hooks

### Commit Message Style

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(api): add endpoint for cost optimization
fix(orchestrator): resolve memory leak in agent lifecycle
docs(readme): update installation instructions
test(agents): add unit tests for incident agent
```

## Community

### Getting Help

- **Discussions**: Use [GitHub Discussions](https://github.com/yourusername/azure-ai-it-copilot/discussions) for questions
- **Discord**: Join our [Discord server](https://discord.gg/azure-ai-copilot) for real-time chat
- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Email**: Contact [wes@wesellis.com](mailto:wes@wesellis.com) for private matters

### Recognition

Contributors are recognized in several ways:

- **Contributors list**: Added to README.md
- **Release notes**: Mentioned in release announcements
- **Special badges**: For significant contributions
- **Maintainer invitation**: For consistent, high-quality contributions

### Areas Looking for Contributors

We especially welcome contributions in these areas:

- **🤖 AI Agents**: Creating new specialized agents
- **🔌 Integrations**: Adding support for new services (AWS, GCP, etc.)
- **📊 ML Models**: Improving prediction and optimization algorithms
- **🎨 UI/UX**: Enhancing the dashboard and user experience
- **📝 Documentation**: Creating tutorials and examples
- **🧪 Testing**: Improving test coverage and quality
- **🌍 Internationalization**: Adding multi-language support

## Development Resources

### Useful Commands

```bash
# Development
make dev              # Start development servers
make test             # Run all tests
make test-watch       # Run tests in watch mode
make lint             # Run all linters
make format           # Format all code
make type-check       # Run type checking

# Docker
make docker-build     # Build Docker images
make docker-up        # Start with Docker Compose
make docker-down      # Stop Docker containers

# Documentation
make docs-serve       # Serve documentation locally
make docs-build       # Build documentation

# Release
make version          # Bump version
make changelog        # Generate changelog
```

### Project Structure

```
azure-ai-it-copilot/
├── 🧠 ai_orchestrator/      # Core AI engine
├── 🌐 api/                  # FastAPI backend
├── 💻 dashboard/            # React frontend
├── 🔌 integrations/         # External service connectors
├── 🤖 automation_engine/    # Execution layer
├── 📊 ml_models/            # Machine learning models
├── ☁️ azure_clients/        # Azure service clients
├── ⚙️ config/               # Configuration management
├── 🚀 infrastructure/       # Deployment configs
├── 📚 docs/                 # Documentation
├── 🧪 tests/                # Test suites
└── 📝 examples/             # Usage examples
```

### Learning Resources

- **FastAPI**: https://fastapi.tiangolo.com/
- **LangChain**: https://langchain.readthedocs.io/
- **Azure SDK**: https://docs.microsoft.com/en-us/azure/developer/python/
- **React**: https://reactjs.org/docs/
- **TypeScript**: https://www.typescriptlang.org/docs/

---

## 🚀 Ready to Contribute?

1. **Star the repository** ⭐
2. **Fork it** 🍴
3. **Clone your fork** 📥
4. **Create a feature branch** 🌿
5. **Make your changes** ✨
6. **Submit a pull request** 📤

Thank you for contributing to Azure AI IT Copilot! 🙏

Together, we're building the future of IT operations automation. 🚀