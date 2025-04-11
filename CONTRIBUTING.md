# Contributing to OARC

Thank you for your interest in contributing to OARC (Ollama Agent Roll Cage)! This document provides guidelines and instructions for contributing to the project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Coding Standards](#coding-standards)
- [Pull Request Process](#pull-request-process)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Issue Reporting](#issue-reporting)
- [Feature Requests](#feature-requests)
- [Community](#community)

## 📜 Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone. We expect all contributors to:

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## 🚀 Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR-USERNAME/oarc.git`
3. Add upstream remote: `git remote add upstream https://github.com/Ollama-Agent-Roll-Cage/oarc.git`
4. Create a feature branch: `git checkout -b feature/your-feature-name`

## 💻 Development Environment

OARC requires Python 3.10 or 3.11. We recommend using UV for environment management:

```bash
# Install UV package manager
pip install uv

# Create & activate virtual environment with UV
uv venv --python 3.11

# Install the package and dependencies in one step
uv run pip install -e .[dev]

# Run the setup command directly
uv run oarc setup

# On Windows, activate the virtual environment with:
.venv\Scripts\activate
# On Unix/MacOS:
# source .venv/bin/activate
```

## 🧰 Coding Standards

### Python Style Guide

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use type hints for function parameters and return values
- Keep line length to a maximum of 100 characters
- Use descriptive variable names
- Document all public functions, classes, and methods using docstrings

### Project Structure

- Place new features in the appropriate module
- Add tests for all new features
- Update documentation when adding or modifying features

## 🔄 Pull Request Process

1. Update your feature branch with latest changes from upstream:
   ```bash
   git fetch upstream
   git rebase upstream/master
   ```

2. Ensure your code passes all tests:
   ```bash
   uv run .\tests\run_all_tests.py
   ```

3. Make sure documentation is updated

4. Push your changes to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

5. Submit a pull request to the `develop` branch of the main repository

6. In your pull request description:
   - Describe what your changes do
   - Link to any related issues
   - Mention any breaking changes
   - Include screenshots for UI changes if applicable

7. Wait for review and address any feedback

## 🧪 Testing Guidelines

- Write tests for all new features and bug fixes
- Ensure existing tests pass before submitting a pull request
- Follow the existing testing patterns in the project
- Use appropriate mocking for external dependencies

Run tests using:
```bash
uv run .\tests\run_all_tests.py
```

For specific test categories:
```bash
# For speech tests
uv run .\tests\speech\tts_fast_tests.py

# Add more specific test examples as needed
```

## 📚 Documentation

- Update README.md for user-facing changes
- Document new features with examples
- Update or create architecture diagrams when necessary
- Document API changes

## 🐛 Issue Reporting

When reporting issues, please include:

1. A clear, descriptive title
2. Steps to reproduce the issue
3. Expected behavior
4. Actual behavior
5. Environment information (OS, Python version, etc.)
6. Screenshots or logs if applicable

## 💡 Feature Requests

We welcome feature requests! When suggesting a new feature:

1. Describe the problem you're trying to solve
2. Explain why this feature would benefit the project
3. Provide examples of how the feature would be used
4. If possible, suggest an implementation approach

## 👥 Community

- Join our Discord channels for discussions:
  - [Discord Server 1](https://discord.gg/vksT5csPbd)
  - [Discord Server 2](https://discord.gg/mNeQZzBHuW)
- Participate in design discussions
- Help review pull requests from other contributors
- Share your use cases and feedback

---

Thank you for contributing to OARC! Your efforts help make this project better for everyone.
