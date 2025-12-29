---
name: python-lint-check
description: Use when writing or modifying Python code to verify linting and type correctness before committing
---

# Python Lint & Type Check

Run ruff for linting and pyright for type checking.

## Commands

```bash
# Lint with ruff
python -m ruff check .

# Auto-fix linting issues
python -m ruff check --fix .

# Type check with pyright
python -m pyright
```

## Workflow

1. After modifying Python files, run `python -m ruff check .`
2. Fix issues or run with `--fix` for auto-fixable problems
3. Run `python -m pyright` for type errors
4. Address any type errors before committing

## Common Ruff Fixes

| Code | Issue | Fix |
|------|-------|-----|
| F401 | Unused import | Remove import |
| F841 | Unused variable | Remove or prefix with `_` |
| E501 | Line too long | Break line or disable per-line |
| I001 | Import order | Run `--fix` |

## Pyright Tips

- Use `# type: ignore` sparingly for false positives
- Add type hints to function signatures to catch more errors
- Check `pyrightconfig.json` or `pyproject.toml` for config

## Script Arguments

When modifying argparse arguments in any script, update the corresponding section in README.md to keep documentation in sync.
