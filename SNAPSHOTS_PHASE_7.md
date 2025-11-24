# Snapshot Loading & Read-Only Mode Implementation Plan

## Overview

This document describes the implementation of:
1. `DataFolio.load_snapshot()` classmethod for full snapshot access
2. `read_only` parameter on DataFolio for write protection
3. Integration with existing `SnapshotView` accessor

## Design Principles

### 1. Read-Only is a General Feature

`read_only` should be a parameter on **any** DataFolio, not just snapshots:

```python
# Read-only snapshot (most common)
snapshot = DataFolio.load_snapshot('path', 'v1.0')  # read_only=True by default

# Read-only current state (for inspection)
folio = DataFolio('path', read_only=True)

# Normal mutable folio
folio = DataFolio('path')  # read_only=False (default)
```

**Rationale:** Separation of concerns. Read-only mode is about safety/permissions, not about snapshots specifically.

### 2. Snapshots Load as Read-Only by Default

When loading a snapshot, `read_only=True` by default to protect immutability:

```python
# Read-only by default (recommended)
snapshot = DataFolio.load_snapshot('path', 'v1.0')
snapshot.add_table('new', df)  # Error: folio is read-only

# Can override if needed (for experimentation)
snapshot = DataFolio.load_snapshot('path', 'v1.0', read_only=False)
snapshot.add_table('new', df)  # OK, modifies current state
```

### 3. Two Ways to Access Snapshots

Keep both patterns for different use cases:

```python
# Quick inspection (read-only view, limited methods)
snapshot_view = folio.snapshots['v1.0']
df = snapshot_view.get_table('data')
print(snapshot_view.metadata)

# Full functionality (read-only DataFolio)
snapshot_folio = DataFolio.load_snapshot('path', 'v1.0')
df = snapshot_folio.get_table('data')
print(snapshot_folio.describe())
model = snapshot_folio.get_model('classifier')
```

## API Design

### 1. Modified `DataFolio.__init__()`

```python
class DataFolio:
    def __init__(
        self,
        bundle_dir: Union[str, Path],
        name: Optional[str] = None,
        description: Optional[str] = None,
        read_only: bool = False,  # NEW PARAMETER
    ):
        """Initialize DataFolio.

        Args:
            bundle_dir: Path to bundle directory
            name: Optional bundle name (auto-generated if not provided)
            description: Optional bundle description
            read_only: If True, prevent all write operations (default: False)

        Examples:
            # Normal mutable folio
            >>> folio = DataFolio('experiments/exp1')
            >>> folio.add_table('data', df)  # OK

            # Read-only folio (inspection only)
            >>> folio = DataFolio('experiments/exp1', read_only=True)
            >>> folio.add_table('data', df)  # Error
            >>> df = folio.get_table('data')  # OK
        """
        self._bundle_dir = Path(bundle_dir)
        self._read_only = read_only
        # ... rest of initialization
```

### 2. New `load_snapshot()` Classmethod

```python
@classmethod
def load_snapshot(
    cls,
    bundle_dir: Union[str, Path],
    snapshot: str,
    read_only: bool = True,  # Read-only by default
) -> "DataFolio":
    """Load a DataFolio in snapshot state.

    Creates a DataFolio instance configured to access items and metadata
    as they existed at snapshot time. By default, the folio is read-only
    to preserve snapshot immutability.

    Args:
        bundle_dir: Path to bundle directory
        snapshot: Snapshot name to load
        read_only: If True, prevent write operations (default: True)

    Returns:
        DataFolio instance in snapshot state

    Raises:
        KeyError: If snapshot doesn't exist

    Examples:
        # Load snapshot for inspection (read-only)
        >>> paper = DataFolio.load_snapshot('research/exp', 'paper-v1')
        >>> model = paper.get_model('classifier')
        >>> print(paper.metadata['accuracy'])
        >>> paper.add_table('new', df)  # Error: read-only

        # Load snapshot for experimentation (mutable)
        >>> baseline = DataFolio.load_snapshot('research/exp', 'v1.0', read_only=False)
        >>> baseline.add_table('new_test', df)  # OK, modifies current state

        # Compare multiple snapshots
        >>> v1 = DataFolio.load_snapshot('path', 'v1.0')
        >>> v2 = DataFolio.load_snapshot('path', 'v2.0')
        >>> print(f"v1: {v1.metadata['accuracy']}, v2: {v2.metadata['accuracy']}")
    """
    # Load folio normally
    folio = cls(bundle_dir, read_only=read_only)

    # Verify snapshot exists
    if snapshot not in folio._snapshots:
        raise KeyError(f"Snapshot '{snapshot}' not found in bundle")

    # Set snapshot mode
    folio._in_snapshot_mode = True
    folio._loaded_snapshot = snapshot

    # Get snapshot metadata
    snapshot_meta = folio._snapshots[snapshot]

    # Replace current metadata with snapshot metadata
    folio._metadata = MetadataDict(snapshot_meta.get('metadata_snapshot', {}))

    # Set current item versions to snapshot versions
    snapshot_versions = snapshot_meta.get('item_versions', {})

    # For each item in snapshot, point to that version
    for item_name, version in snapshot_versions.items():
        # Find the item with this name and version
        item = folio._find_item_version(item_name, version)
        if item:
            folio._items[item_name] = item

    # Remove items not in snapshot
    current_items = list(folio._items.keys())
    for item_name in current_items:
        if item_name not in snapshot_versions:
            del folio._items[item_name]

    return folio
```

### 3. Helper Method `_find_item_version()`

```python
def _find_item_version(self, name: str, version: int) -> Optional[Dict[str, Any]]:
    """Find item by name and version.

    Args:
        name: Item name
        version: Version number

    Returns:
        Item metadata dict or None if not found
    """
    # Check current items
    if name in self._items:
        item = self._items[name]
        if item.get('version') == version:
            return item

    # Check snapshot versions
    for item in self._snapshot_versions:
        if item.get('name') == name and item.get('version') == version:
            return item

    return None
```

### 4. Read-Only Enforcement

Add checks to all write methods:

```python
def _check_read_only(self) -> None:
    """Raise error if folio is read-only.

    Raises:
        RuntimeError: If folio is in read-only mode
    """
    if self._read_only:
        msg = "Cannot modify a read-only DataFolio"
        if hasattr(self, '_in_snapshot_mode') and self._in_snapshot_mode:
            msg += f" (loaded from snapshot '{self._loaded_snapshot}')"
        msg += ". Open without read_only=True to make changes."
        raise RuntimeError(msg)

# Add to all write methods:
def add_table(self, name: str, data: pd.DataFrame, **kwargs) -> Self:
    """Add a table to the bundle."""
    self._check_read_only()  # NEW
    # ... rest of implementation

def delete(self, names: Union[str, list[str]], **kwargs) -> Self:
    """Delete items from bundle."""
    self._check_read_only()  # NEW
    # ... rest of implementation

def create_snapshot(self, name: str, **kwargs) -> None:
    """Create a snapshot."""
    self._check_read_only()  # NEW
    # ... rest of implementation

# Metadata dict also needs read-only checks
class MetadataDict(dict):
    def __setitem__(self, key, value):
        if hasattr(self, '_folio') and self._folio._read_only:
            raise RuntimeError("Cannot modify metadata of a read-only DataFolio")
        super().__setitem__(key, value)
```

### 5. Properties for Snapshot Info

```python
@property
def read_only(self) -> bool:
    """Check if folio is in read-only mode."""
    return self._read_only

@property
def in_snapshot_mode(self) -> bool:
    """Check if folio was loaded from a snapshot."""
    return getattr(self, '_in_snapshot_mode', False)

@property
def loaded_snapshot(self) -> Optional[str]:
    """Get name of loaded snapshot, or None."""
    return getattr(self, '_loaded_snapshot', None)
```

### 6. Enhanced `__repr__()` and `describe()`

```python
def __repr__(self) -> str:
    base = f"DataFolio('{self._bundle_dir}', items={len(self._items)}, snapshots={len(self._snapshots)})"

    if self._read_only:
        base += " [READ-ONLY]"

    if self.in_snapshot_mode:
        base += f" [snapshot: {self.loaded_snapshot}]"

    return base

def describe(self, **kwargs) -> Optional[str]:
    """Describe the bundle."""
    # Add header info
    lines = []

    if self.in_snapshot_mode:
        lines.append(f"Snapshot: {self.loaded_snapshot}")
        snapshot_meta = self._snapshots[self.loaded_snapshot]
        lines.append(f"Created: {snapshot_meta.get('timestamp', 'unknown')}")
        if snapshot_meta.get('description'):
            lines.append(f"Description: {snapshot_meta['description']}")
        lines.append("")

    if self._read_only:
        lines.append("[READ-ONLY MODE]")
        lines.append("")

    # ... rest of describe output
```

## Error Messages

### Clear, Actionable Error Messages

```python
# Read-only folio
>>> folio = DataFolio('path', read_only=True)
>>> folio.add_table('new', df)
RuntimeError: Cannot modify a read-only DataFolio.
Open without read_only=True to make changes.

# Read-only snapshot
>>> snapshot = DataFolio.load_snapshot('path', 'v1.0')
>>> snapshot.add_table('new', df)
RuntimeError: Cannot modify a read-only DataFolio (loaded from snapshot 'v1.0').
Open without read_only=True to make changes.

# Metadata modification
>>> snapshot.metadata['test'] = 123
RuntimeError: Cannot modify metadata of a read-only DataFolio
```

## Use Cases

### 1. Load Snapshot for Inspection

```python
# Load paper submission version
paper = DataFolio.load_snapshot('research/protein', 'neurips-2025')

# Safe inspection (read-only)
model = paper.get_model('classifier')
data = paper.get_table('train_data')
print(paper.metadata['accuracy'])
paper.describe()

# Cannot accidentally modify
paper.add_table('oops', df)  # Error!
```

### 2. Compare Multiple Snapshots

```python
# Load multiple versions
v1 = DataFolio.load_snapshot('experiments/tuning', 'lr0.001')
v2 = DataFolio.load_snapshot('experiments/tuning', 'lr0.01')
v3 = DataFolio.load_snapshot('experiments/tuning', 'lr0.1')

# Compare
results = []
for version, folio in [('v1', v1), ('v2', v2), ('v3', v3)]:
    results.append({
        'version': version,
        'accuracy': folio.metadata['accuracy'],
        'f1': folio.metadata['f1_score']
    })

best = max(results, key=lambda x: x['accuracy'])
print(f"Best: {best['version']} with {best['accuracy']}")
```

### 3. Start from Snapshot (Mutable)

```python
# Load snapshot as mutable starting point
exp = DataFolio.load_snapshot('research/baseline', 'v1.0', read_only=False)

# Can modify (updates current state, not snapshot)
exp.add_table('new_validation', val_df)
exp.metadata['variant'] = 'v1.0-with-new-val'

# Create new snapshot
exp.create_snapshot('v1.1-extended')
```

### 4. Safe Inspection of Any Folio

```python
# Open current state as read-only (for safety)
folio = DataFolio('experiments/active', read_only=True)

# Can inspect but not modify
print(folio.describe())
model = folio.get_model('classifier')
predictions = model.predict(X_test)

# Cannot accidentally break things
folio.delete('important_data')  # Error: read-only
```

## Implementation Tasks

### Task 1: Add `read_only` Parameter

- [ ] Add `read_only` parameter to `__init__` (default: False)
- [ ] Store as `self._read_only`
- [ ] Add `read_only` property

### Task 2: Read-Only Enforcement

- [ ] Implement `_check_read_only()` helper
- [ ] Add checks to all `add_*()` methods
- [ ] Add check to `delete()`
- [ ] Add check to `create_snapshot()`
- [ ] Add check to `restore_snapshot()`
- [ ] Add check to `delete_snapshot()`
- [ ] Make `MetadataDict` respect read-only mode

### Task 3: Implement `load_snapshot()`

- [ ] Implement `load_snapshot()` classmethod
- [ ] Implement `_find_item_version()` helper
- [ ] Set `_in_snapshot_mode` flag
- [ ] Set `_loaded_snapshot` name
- [ ] Replace metadata with snapshot metadata
- [ ] Point items to snapshot versions
- [ ] Remove items not in snapshot

### Task 4: Properties and Display

- [ ] Add `in_snapshot_mode` property
- [ ] Add `loaded_snapshot` property
- [ ] Update `__repr__()` to show read-only status
- [ ] Update `__repr__()` to show snapshot name
- [ ] Update `describe()` to show snapshot info
- [ ] Update `describe()` to show read-only warning

### Task 5: Testing

- [ ] Test `read_only=True` on normal folio
- [ ] Test all write operations fail in read-only mode
- [ ] Test all read operations work in read-only mode
- [ ] Test `load_snapshot()` with read-only (default)
- [ ] Test `load_snapshot()` with read_only=False
- [ ] Test loading same snapshot multiple times
- [ ] Test loading different snapshots simultaneously
- [ ] Test that snapshot versions are correct
- [ ] Test that snapshot metadata is correct
- [ ] Test error messages are clear and actionable
- [ ] Test integration with existing `folio.snapshots['v1.0']`

### Task 6: Documentation

- [ ] Update docstrings for all modified methods
- [ ] Add examples to `load_snapshot()` docstring
- [ ] Add read-only examples to main docs
- [ ] Update snapshots guide with load_snapshot examples
- [ ] Update README with load_snapshot examples

## Testing Strategy

### Unit Tests

```python
class TestReadOnlyMode:
    """Test read-only mode on any folio."""

    def test_read_only_prevents_add(self):
        folio = DataFolio(path, read_only=True)
        with pytest.raises(RuntimeError, match="read-only"):
            folio.add_table('new', df)

    def test_read_only_prevents_delete(self):
        folio = DataFolio(path, read_only=True)
        with pytest.raises(RuntimeError, match="read-only"):
            folio.delete('existing')

    def test_read_only_prevents_metadata_write(self):
        folio = DataFolio(path, read_only=True)
        with pytest.raises(RuntimeError, match="read-only"):
            folio.metadata['test'] = 123

    def test_read_only_allows_read(self):
        folio = DataFolio(path, read_only=True)
        df = folio.get_table('data')  # Should work
        assert df is not None

class TestLoadSnapshot:
    """Test loading snapshots as DataFolio instances."""

    def test_load_snapshot_basic(self):
        folio = DataFolio(path)
        folio.add_table('data', df)
        folio.metadata['accuracy'] = 0.89
        folio.create_snapshot('v1.0')

        # Load snapshot
        snapshot = DataFolio.load_snapshot(path, 'v1.0')

        assert snapshot.in_snapshot_mode
        assert snapshot.loaded_snapshot == 'v1.0'
        assert snapshot.read_only  # Default
        assert snapshot.metadata['accuracy'] == 0.89

    def test_load_snapshot_has_correct_items(self):
        folio = DataFolio(path)
        folio.add_table('data1', df1)
        folio.create_snapshot('v1.0')

        folio.add_table('data2', df2)
        folio.create_snapshot('v2.0')

        # Load v1.0 - should only have data1
        v1 = DataFolio.load_snapshot(path, 'v1.0')
        assert 'data1' in v1._items
        assert 'data2' not in v1._items

        # Load v2.0 - should have both
        v2 = DataFolio.load_snapshot(path, 'v2.0')
        assert 'data1' in v2._items
        assert 'data2' in v2._items

    def test_load_snapshot_read_only_by_default(self):
        folio = DataFolio(path)
        folio.add_table('data', df)
        folio.create_snapshot('v1.0')

        snapshot = DataFolio.load_snapshot(path, 'v1.0')

        with pytest.raises(RuntimeError, match="read-only"):
            snapshot.add_table('new', df)

    def test_load_snapshot_mutable(self):
        folio = DataFolio(path)
        folio.add_table('data', df)
        folio.create_snapshot('v1.0')

        snapshot = DataFolio.load_snapshot(path, 'v1.0', read_only=False)

        # Should allow modifications
        snapshot.add_table('new', df)  # No error
        assert 'new' in snapshot._items

    def test_load_multiple_snapshots(self):
        # Create snapshots
        folio = DataFolio(path)
        folio.metadata['accuracy'] = 0.85
        folio.create_snapshot('v1.0')

        folio.metadata['accuracy'] = 0.90
        folio.create_snapshot('v2.0')

        # Load both simultaneously
        v1 = DataFolio.load_snapshot(path, 'v1.0')
        v2 = DataFolio.load_snapshot(path, 'v2.0')

        assert v1.metadata['accuracy'] == 0.85
        assert v2.metadata['accuracy'] == 0.90

    def test_load_nonexistent_snapshot(self):
        folio = DataFolio(path)

        with pytest.raises(KeyError, match="not found"):
            DataFolio.load_snapshot(path, 'nonexistent')
```

### Integration Tests

```python
class TestSnapshotWorkflow:
    """Test complete snapshot loading workflows."""

    def test_paper_submission_workflow(self):
        # Create and snapshot
        folio = DataFolio('research/paper')
        folio.add_table('data', data)
        folio.add_model('model', model)
        folio.metadata['accuracy'] = 0.89
        folio.create_snapshot('neurips-2025')

        # Continue research
        folio.add_model('model', new_model, overwrite=True)
        folio.metadata['accuracy'] = 0.91

        # Later: Load paper version
        paper = DataFolio.load_snapshot('research/paper', 'neurips-2025')

        assert paper.metadata['accuracy'] == 0.89
        model = paper.get_model('model')  # Original version

        # Cannot modify
        with pytest.raises(RuntimeError):
            paper.add_table('new', df)

    def test_ab_testing_workflow(self):
        # Create two versions
        folio = DataFolio('models/recommender')
        folio.add_model('model', baseline)
        folio.metadata['latency'] = 50
        folio.create_snapshot('v1-baseline')

        folio.add_model('model', experimental, overwrite=True)
        folio.metadata['latency'] = 45
        folio.create_snapshot('v2-experimental')

        # Deploy both
        baseline = DataFolio.load_snapshot('models/recommender', 'v1-baseline')
        experimental = DataFolio.load_snapshot('models/recommender', 'v2-experimental')

        deploy(baseline.get_model('model'), 'prod-a')
        deploy(experimental.get_model('model'), 'prod-b')

        # Compare
        assert experimental.metadata['latency'] < baseline.metadata['latency']
```

## Migration Path

### Backward Compatibility

All existing code continues to work:

```python
# Existing code (unchanged)
folio = DataFolio('path')
folio.add_table('data', df)
snapshot_view = folio.snapshots['v1.0']  # Still works

# New code (load_snapshot)
snapshot_folio = DataFolio.load_snapshot('path', 'v1.0')
```

### Deprecation Path

`SnapshotView` remains for quick inspection:
- Keep `folio.snapshots['v1.0']` accessor
- Use for quick checks: `folio.snapshots['v1.0'].metadata`
- Use `load_snapshot()` for full functionality

No breaking changes!

## Future Enhancements

### Possible Future Features

1. **Read-only at item level**
   ```python
   folio.add_table('frozen', df, read_only=True)
   ```

2. **Permission modes**
   ```python
   folio = DataFolio('path', permissions='read-only')
   folio = DataFolio('path', permissions='read-write')
   folio = DataFolio('path', permissions='append-only')
   ```

3. **Snapshot branching**
   ```python
   branch = DataFolio.branch_from_snapshot('path', 'v1.0', 'new-branch')
   ```

4. **Snapshot diffs in describe**
   ```python
   snapshot.describe(show_diff_from='current')
   ```

## Summary

This design provides:

✅ **Flexible read-only mode** - works on any folio, not just snapshots
✅ **Safe by default** - snapshots load as read-only
✅ **Clear errors** - actionable messages with suggestions
✅ **Full functionality** - loaded snapshots work like normal folios
✅ **Backward compatible** - existing `SnapshotView` still works
✅ **Well-tested** - comprehensive test coverage

The implementation gives users two complementary ways to access snapshots:
1. **Quick inspection**: `folio.snapshots['v1.0']` (lightweight, read-only view)
2. **Full access**: `DataFolio.load_snapshot('path', 'v1.0')` (complete folio, read-only by default)
