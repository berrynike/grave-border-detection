# Prepare Commit

Run all code quality checks, fix any issues, and ensure CI will pass.

## Steps

### 1. Format and Lint

Run `uv run task lint` which includes ruff check, ruff format, and mypy.

- If there are **formatting issues**: run `uv run ruff format .` to auto-fix
- If there are **lint errors with auto-fixes**: run `uv run ruff check --fix .`
- If there are **lint errors without auto-fixes**: fix them manually
- If there are **mypy errors**: fix the type issues manually
- Re-run `uv run task lint` until it passes cleanly

### 2. Run Tests

Run `uv run task test` to execute the test suite with coverage.

- If tests fail: fix the failing tests or the code causing failures
- Re-run until all tests pass

### 3. Pre-commit Hooks

Run `uv run pre-commit run --all-files` to catch any remaining issues.

- Fix any issues found
- Re-run until clean

### 4. Summary

After all checks pass, report:
- What issues were found and fixed
- Current test coverage
- Confirmation that all checks pass

Do NOT create a commit. Only prepare the code so it's ready to commit.
