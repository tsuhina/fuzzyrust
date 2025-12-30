# Installation

## Requirements

- Python 3.9 or higher
- pip or uv package manager

## Install from PyPI

=== "pip"

    ```bash
    pip install fuzzyrust
    ```

=== "uv"

    ```bash
    uv add fuzzyrust
    ```

## Verify Installation

```python
import fuzzyrust as fr
print(fr.__version__)
print(fr.jaro_winkler("hello", "world"))
```

## Optional Dependencies

FuzzyRust works standalone, but for DataFrame operations you'll need Polars:

```bash
pip install polars
```

## Development Installation

To install from source for development:

```bash
git clone https://github.com/tsuhina/fuzzyrust.git
cd fuzzyrust
uv sync --extra dev
uv run maturin develop --release
```

## Platform Support

FuzzyRust provides pre-built wheels for:

- **Linux**: x86_64, aarch64
- **macOS**: x86_64 (Intel), arm64 (Apple Silicon)
- **Windows**: x86_64

If no wheel is available for your platform, pip will build from source (requires Rust toolchain).
