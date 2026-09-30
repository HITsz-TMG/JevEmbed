# Contributing to JevEmbed

Thank you for helping improve JevEmbed.

## Development and validation

Install the development dependencies and run the test suite from the repository root:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests use fixtures and mock models without downloading model weights. Save generated traces under the ignored `artifacts/` directory.

Build the wheel and source distribution before publishing a release:

```bash
python -m build
```
