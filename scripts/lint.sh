#!/bin/bash
set -e

cd "$(dirname "$0")/.."

case "${1:-check}" in
    check)
        echo "Running ruff check..."
        python -m ruff check .

        echo ""
        echo "Running ruff format check..."
        python -m ruff format --check .

        echo ""
        echo "All checks passed!"
        ;;
    fix)
        echo "Running ruff check --fix..."
        python -m ruff check --fix .

        echo ""
        echo "Running ruff format..."
        python -m ruff format .

        echo ""
        echo "Done!"
        ;;
    types)
        echo "Running pyright..."
        python -m pyright
        ;;
    *)
        echo "Usage: $0 [check|fix|types]"
        echo "  check - Run linters (default)"
        echo "  fix   - Auto-fix issues and format"
        echo "  types - Run type checker"
        exit 1
        ;;
esac
