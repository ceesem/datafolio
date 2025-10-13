# datafolio Development Guide

This guide documents the development workflow, testing strategies, and coding standards for datafolio.

## Development Environment

### Package Management: UV + Poe

This project uses **UV** for dependency management and **Poe** for task running instead of the system Python environment:

```bash
# Install dependencies (automatically creates virtual environment)
uv sync

# Run tasks via Poe (preferred over direct python commands)
poe test          # Run tests with coverage
poe doc-preview   # Preview documentation
poe bump patch    # Bump version (patch/minor/major)
poe drybump patch # Dry run version bump
```

### Key Commands

From `pyproject.toml`:

- **Testing**: `poe test` → `uv run pytest --cov=datafolio tests`
- **Documentation**: `poe doc-preview` → `uv run mkdocs serve`
- **Version Management**: `poe bump patch/minor/major`
- **Linting**: `uv run ruff check src/ tests/`

## Python Version Requirements

- **Minimum**: Python 3.10+ (leverages match statements and improved type annotations)
- **Tested**: Python 3.10, 3.11, 3.12
- **Compatibility**: Use `typing-extensions` for Python < 3.11 features

## Type Hinting Standards

### Required Practices

- **All functions/methods** must have complete type annotations
- **All class attributes** must have type hints (use `attrs` or `dataclasses` with type annotations).
- **Import patterns**:
  ```python
  from typing import Optional, Union, Literal, Any, Dict, List, Tuple
  from typing_extensions import Self  # For Python < 3.11
  ```
- **All python files**: Follow ruff guidelines for linting, imports, and code style.

### Common Patterns

## Testing Strategy

### Dual-Level Testing Approach

Always implement **both** high-level integration tests and low-level unit tests:

#### High-Level Integration Tests
Focus on real-world workflows and end-to-end functionality:

#### Low-Level Unit Tests
Focus on individual methods, edge cases, and error conditions:

### Testing Guidelines

- **Coverage Target**: Aim for >90% line coverage, >85% branch coverage
- **Test Organization**: Mirror source structure in `tests/` directory
- **Mocking Strategy**: Mock external dependencies (neuroglancer), test actual behavior for internal logic
- **Parametrization**: Use `@pytest.mark.parametrize` for testing multiple inputs
- **Fixtures**: Create reusable test data in `conftest.py`

### Running Tests

```bash
# Full test suite with coverage
poe test

# Specific test file
uv run pytest tests/EXAMPLE_TEST_FILE.py -v

# Coverage report
uv run pytest --cov=datafolio --cov-report=html tests/
```

## Code Architecture Patterns

## Common Development Workflows

### Bug Fixes

1. Write failing test first (TDD approach)
2. Describe problem fully to determine whether the behavior is a bug, intentional, or based on unspecified assumptions.
3. Implement minimal fix
4. Ensure all tests pass
5. Check type annotations are correct
6. Run full test suite: `poe test`

## Documentation Standards

- **Docstrings**: Use Google/NumPy style docstrings
- **Examples**: Include usage examples in docstrings
- **Type Information**: Document parameter and return types in docstrings
- **API Documentation**: Use mkdocs with mkdocstrings for auto-generation

```python
def add_points(
    self, 
    data: pd.DataFrame,
    point_column: Union[str, List[str]],
    segment_column: Optional[str] = None
) -> Self:
    """Add point annotations to the viewer state.
    
    Args:
        data: DataFrame containing point data
        point_column: Column name(s) for coordinates. Can be:
            - Single string for prefix (e.g., 'pos' → ['pos_x', 'pos_y', 'pos_z'])  
            - List of column names (e.g., ['x', 'y', 'z'])
        segment_column: Optional column containing segment IDs
        
    Returns:
        Self for method chaining
        
    Examples:
        >>> vs = ViewerState()
        >>> vs.add_points(df, point_column=['x', 'y', 'z'])
        >>> vs.add_points(df, point_column='position')  # Uses position_x, position_y, position_z
    """
```

## Pre-Commit Workflow

Before committing changes:

```bash
# Run full test suite
poe test

# Run linting
uv run ruff check src/ tests/

# Check type annotations
uv run mypy src/ --ignore-missing-imports  # if mypy is configured

# Ensure documentation builds
poe doc-preview
```

## Release Process

```bash
# Check what will be bumped
poe drybump patch  # or minor/major

# Bump version and create tag
poe bump patch     # or minor/major
```

This automatically:
- Updates version in `pyproject.toml` and `src/datafolio/__init__.py`
- Creates git commit and tag
- Runs pre/post commit hooks including `uv sync`