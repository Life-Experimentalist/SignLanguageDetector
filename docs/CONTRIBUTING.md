# Contributing to Sign Language Detector

## Getting Started
1. Fork the repository
2. Create a virtual environment: `uv venv --python 3.12 .venv`
3. Install dependencies: `uv sync --python .venv\Scripts\python.exe`
4. Run the application: `uv run --python .venv\Scripts\python.exe python app.py`

## Development Workflow
1. Create a new branch for your feature
2. Make your changes
3. Write or update tests
4. Submit a pull request

## Code Style
- Follow PEP 8 guidelines
- Use type hints where possible
- Include docstrings for functions and classes
- Keep functions focused and single-purpose

## Commit Guidelines
- Use clear, descriptive commit messages
- Reference issues in commits where applicable
- Keep commits focused and atomic

## Pull Request Process
1. Update documentation as needed
2. Add tests for new features
3. Ensure all tests pass
4. Update the changelog
5. Request review from maintainers

## Testing
- Run tests before submitting: `uv run --python .venv\Scripts\python.exe pytest`
- Include both unit and integration tests
- Maintain test coverage above 80%

## Documentation
- Update relevant documentation
- Include docstrings for new functions
- Add comments for complex logic
- Update README.md if needed
