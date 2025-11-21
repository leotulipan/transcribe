# Contributing to Audio Transcribe

Thank you for your interest in contributing to Audio Transcribe! This document provides guidelines and instructions for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/audio-transcribe.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test your changes
6. Commit your changes: `git commit -m "Add your feature"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Open a Pull Request

## Development Setup

1. Install [UV](https://github.com/astral-sh/uv) if you haven't already
2. Clone the repository
3. Install dependencies: `uv sync`
4. Run the development version: `uv run transcribe.py --help`

## Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Add docstrings to functions and classes
- Keep functions focused and small
- Write meaningful commit messages

## Testing

Before submitting a PR, please:
- Test your changes with different audio formats
- Test with different APIs if your changes affect API integration
- Ensure existing functionality still works
- Add tests if you're adding new features

## Pull Request Process

1. Update the README.md if needed
2. Update TASKS.md if you're working on a tracked task
3. Ensure your code follows the project's style guidelines
4. Make sure all tests pass (if applicable)
5. Request review from maintainers

## Reporting Issues

When reporting issues, please include:
- Description of the issue
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment details (OS, Python version, etc.)
- Relevant error messages or logs

## Feature Requests

Feature requests are welcome! Please open an issue with:
- Clear description of the feature
- Use case and motivation
- Potential implementation approach (if you have ideas)

## Questions?

Feel free to open an issue for questions or discussions.

