# Python Development Rules

This document defines Python coding standards for this project. Referenced by `.cursorrules` and `CLAUDE.md`.

## Modern Python (3.12+) - STRICT

### Type Hints

```python
# Use built-in types, not typing module
list[str]           # not List[str]
dict[str, int]      # not Dict[str, int]
tuple[int, ...]     # not Tuple[int, ...]
set[str]            # not Set[str]

# Use | union syntax
str | None          # not Optional[str]
int | str           # not Union[int, str]

# Note: Any still needs typing import
from typing import Any
```

### Data Containers

```python
# Use dataclasses for internal models
from dataclasses import dataclass

@dataclass(slots=True)  # slots for performance
class TileMetadata:
    x: int
    y: int
    width: int
    height: int
    crs: str | None = None

# Use Pydantic only for:
# - API boundaries
# - Config file validation
# - External data validation
```

### Path Operations

```python
# Use pathlib, not os.path
from pathlib import Path

# Good
path = Path(base_dir) / "data" / "file.tif"
if path.exists():
    content = path.read_text()

# Bad
import os
path = os.path.join(base_dir, "data", "file.tif")
```

### String Operations

```python
# Use f-strings
message = f"Processing {filename} with CRS {crs}"

# Use removeprefix/removesuffix
name = filename.removeprefix("test_")
name = filename.removesuffix(".tif")

# Use match-case for complex conditionals
match file_type:
    case "ortho":
        process_orthophoto(path)
    case "dem" | "dsm":
        process_elevation(path)
    case _:
        raise ValueError(f"Unknown file type: {file_type}")
```

### Walrus Operator

```python
# Use := when it improves readability
if (count := len(items)) > 10:
    logger.info(f"Processing {count} items")

while (line := file.readline()):
    process_line(line)
```

## Function Complexity Limits

| Metric | Maximum |
|--------|---------|
| Cyclomatic complexity | 10 |
| Statements per function | 30 |
| Lines per function | 100 |
| Nesting depth | 4 |
| Parameters | 5 |

For functions needing more parameters, use a dataclass or TypedDict.

## Error Handling

### Fail Fast Policy

```python
# Good - specific exception, explicit failure
def load_raster(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Raster not found: {path}")

    with rasterio.open(path) as src:
        return src.read()

# Bad - silent fallback
def load_raster(path: Path) -> np.ndarray | None:
    try:
        with rasterio.open(path) as src:
            return src.read()
    except Exception:
        return None  # Never do this
```

### Exception Rules

```python
# Catch specific exceptions
try:
    result = risky_operation()
except (ValueError, TypeError) as e:
    logger.error(f"Operation failed: {e}")
    raise

# Never use bare except
try:
    result = operation()
except:  # Bad - catches SystemExit, KeyboardInterrupt
    pass

# Never catch broad Exception without re-raising
try:
    result = operation()
except Exception:  # Bad - swallows all errors
    return default_value
```

### Context Managers

```python
# Always use context managers for resources
with rasterio.open(path) as src:
    data = src.read()

with open(log_file, "w") as f:
    f.write(message)

# For multiple resources
with rasterio.open(input_path) as src, rasterio.open(output_path, "w", **profile) as dst:
    dst.write(src.read())
```

## Code Cleanup Requirements

After every change, ensure:

- [ ] No unused imports
- [ ] No unused variables
- [ ] No dead/unreachable code
- [ ] No commented-out code (unless marked TODO)
- [ ] No duplicate code or interfaces
- [ ] No temporary debugging (print, breakpoint)
- [ ] No mutable default arguments

## Testing Standards

### Test Pyramid

| Type | Coverage | Focus |
|------|----------|-------|
| Unit | 70% | Business logic, algorithms, data transformations |
| Integration | 20% | Data pipelines, model inference |
| End-to-end | 10% | Complete workflows |

### Test Rules

```python
# Use fixtures for reusable test data
@pytest.fixture
def sample_raster(tmp_path: Path) -> Path:
    path = tmp_path / "test.tif"
    create_test_raster(path)
    return path

# Mock external dependencies
def test_load_raster(sample_raster: Path, mocker):
    mock_open = mocker.patch("rasterio.open")
    # ...

# Test behaviors, not implementation
def test_tiles_cover_entire_image():
    tiles = create_tiles(image, tile_size=256)
    covered = sum(t.width * t.height for t in tiles)
    assert covered >= image.width * image.height
```

### Coverage Target

- Minimum 80% test coverage
- 100% coverage for critical paths (CRS handling, data export)

## Common Mistakes

### Python Style

| Bad | Good |
|-----|------|
| `from typing import List, Dict` | `list[str]`, `dict[str, int]` |
| `Optional[str]` | `str \| None` |
| `os.path.join(a, b)` | `Path(a) / b` |
| `except Exception:` | `except ValueError:` |
| `def f(items=[]):` | `def f(items=None):` |

### Geospatial

| Bad | Good |
|-----|------|
| Hardcode CRS | Read from source, validate |
| Ignore CRS mismatch | Reproject or raise error |
| Load full raster | Use windowed reading |
| Modify raw data | Write to processed/ |

### Deep Learning

| Bad | Good |
|-----|------|
| No train/val/test split | Proper stratified splits |
| Hardcode hyperparameters | Use Hydra configs |
| Ignore GPU memory | Use gradient checkpointing, mixed precision |
| Skip augmentation | Use Albumentations |

## Dependency Management

### Version Policy

- **Use latest compatible versions** — don't pin to old versions without reason
- **Check the web** for current versions when adding new dependencies
- **Verify compatibility** with existing stack (especially PyTorch, Lightning, GDAL)

### Adding Dependencies

```bash
# Check latest version before adding
uv add <package>  # uv automatically gets latest compatible version

# For specific version requirements, verify on PyPI first
uv add "package>=X.Y"
```

### Python Version

- Keep Python version current (currently 3.12+)
- When upgrading Python, verify all dependencies support the new version
- Lightning is often the bottleneck for Python version support

## Quality Commands

```bash
# Run before every commit
uv run task lint             # Linting, formatting, type checking
uv run task test             # Tests with coverage

# Or individually
uv run ruff check .          # Linting
uv run ruff format .         # Formatting
uv run mypy src/             # Type checking
uv run pytest tests/ -v --cov=grave_border_detection  # Tests
```
