#!/usr/bin/env bash
# verify.sh
# Must be executed from final_project/ directory.
# Exit code 0 = all checks green.  Non-zero = failure, do not commit.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "============================================================"
echo "  verify.sh - ENPM 701 final_project gate checks"
echo "============================================================"

# -------------------------------------------------------------------
# Step 1: pytest
# -------------------------------------------------------------------
echo ""
echo "--- [1/3] pytest -v --tb=short ---"
python -m pytest tests/ -v --tb=short
echo "[1/3] pytest: PASSED"

# -------------------------------------------------------------------
# Step 2: syntax check all top-level .py files and the fp/ package
# -------------------------------------------------------------------
echo ""
echo "--- [2/3] py_compile syntax check ---"
for f in *.py fp/*.py; do
    python -m py_compile "$f" && echo "  OK  $f" || { echo "  FAIL $f"; exit 1; }
done
echo "[2/3] py_compile: PASSED"

# -------------------------------------------------------------------
# Step 3: smoke import (all modules must import without error)
# Step is guarded by the hw_stub in tests/__init__.py
# -------------------------------------------------------------------
echo ""
echo "--- [3/3] smoke import ---"
python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
import tests  # apply hw_stub before any hardware modules load
import main
from fp import hal, arena, localization, motion, search, vision, mapping, fsm, cli
print("  Smoke import OK:", [
    'main', 'fp.hal', 'fp.arena', 'fp.localization', 'fp.motion',
    'fp.search', 'fp.vision', 'fp.mapping', 'fp.fsm', 'fp.cli',
])
PYEOF
echo "[3/3] smoke import: PASSED"

echo ""
echo "============================================================"
echo "  All checks PASSED"
echo "============================================================"
