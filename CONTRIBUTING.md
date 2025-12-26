# Contributing to Automated Time Series Forecasting

Thank you for your interest in contributing!

## Development Setup

1. Fork and clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Code Style

- Follow PEP 8 guidelines
- Use Black for formatting: `black src/`
- Run Flake8 for linting: `flake8 src/`

## Testing

Run tests with pytest:
```bash
pytest tests/ -v
```

## Pull Request Process

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes with clear commit messages
3. Add tests for new functionality
4. Update documentation as needed
5. Submit a pull request

## Adding New Models

To add a new forecasting model:

1. Create a new file in `src/models/`
2. Implement a class with `fit()` and `predict()` methods
3. Add to `src/models/__init__.py`
4. Update the pipeline in `src/pipeline/forecaster.py`
5. Add tests in `tests/`
