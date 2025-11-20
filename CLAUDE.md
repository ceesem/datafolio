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

**Avoiding circular imports with TYPE_CHECKING:**
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datafolio.folio import DataFolio  # Only imported for type checking

def my_method(self, folio: "DataFolio") -> None:  # Use string quotes
    ...
```

**Modern type hints (Python 3.10+):**
```python
# Prefer built-in types over typing module
def process(data: list[str]) -> dict[str, Any]:  # ✅ Python 3.10+
def process(data: List[str]) -> Dict[str, Any]:  # ❌ Old style
```

## Testing Strategy

### Dual-Level Testing Approach

Always implement **both** high-level integration tests and low-level unit tests:

#### High-Level Integration Tests
Focus on real-world workflows and end-to-end functionality:

- Test complete workflows (add → save → load → delete)
- Use real data types (DataFrames, models, etc.)
- Test with actual file I/O (no mocking of storage)
- Example: `tests/test_folio_basic.py`, `tests/test_pytorch_models.py`

#### Low-Level Unit Tests
Focus on individual methods, edge cases, and error conditions:

- Test individual handler methods in isolation
- Test edge cases and error conditions
- Mock external dependencies when appropriate
- Example: `tests/test_handlers_pytorch.py`, `tests/test_storage_backend.py`

### Testing Guidelines

- **Coverage Target**: Aim for >75% line coverage, >70% branch coverage
- **Current Status**: 79% coverage with 423 tests passing
- **100% Coverage Modules**: metadata.py, timestamps.py, categories.py, handlers/__init__.py, base/registry.py
- **Test Organization**: Mirror source structure in `tests/` directory
- **Mocking Strategy**: Mock external dependencies when needed, test actual behavior for internal logic
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

### Handler-Based Architecture

DataFolio uses a modular handler system where each data type is managed by a dedicated handler class. This architecture provides clean separation of concerns and makes the codebase easy to extend.

**Core Components:**
- **BaseHandler** (`base/handler.py`): Abstract base class defining the handler interface
- **HandlerRegistry** (`base/registry.py`): Global registry for managing and auto-detecting handlers
- **StorageBackend** (`storage/backend.py`): Abstraction layer for file I/O (local and cloud)
- **Handlers** (`handlers/`): 8 specialized handlers for different data types

**Handler Structure** (~35-80 lines each):
```python
from datafolio.base.handler import BaseHandler

class MyHandler(BaseHandler):
    @property
    def item_type(self) -> str:
        return "my_type"  # Must be in ITEM_TYPE_TO_CATEGORY mapping

    def can_handle(self, data: Any) -> bool:
        return isinstance(data, MyDataType)

    def add(self, folio, name, data, **kwargs) -> Dict[str, Any]:
        # Write data, return metadata
        ...

    def get(self, folio, name, **kwargs) -> Any:
        # Read and return data
        ...
```

**When to create a new handler:**
- Adding support for a new data type (e.g., Polars DataFrames, DuckDB tables)
- Adding a new storage format for existing types
- Adding specialized behavior that doesn't fit existing handlers

**How to add a handler:**
1. Create new file in `handlers/` directory
2. Inherit from `BaseHandler` and implement required methods
3. Add your `item_type` to `ITEM_TYPE_TO_CATEGORY` in `storage/categories.py`
4. Register handler in `handlers/__init__.py`
5. Add comprehensive unit tests (see `tests/test_handlers_pytorch.py` for example)
6. Update integration tests if needed

**Storage organization:**
- Handlers don't manually specify storage paths
- Storage location is auto-derived from `item_type` via `ITEM_TYPE_TO_CATEGORY`
- Three storage categories: TABLES, MODELS, ARTIFACTS

**See `ARCHITECTURE.md` for detailed documentation of the handler system.**

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