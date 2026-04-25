#!/usr/bin/env bash
# verify.sh - Gate script run before every commit in prod-hardening.
# Must be executed from enpm701/final_project/ directory.
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
from fp import hal, arena, localization, motion, navigation, state_machine, ekf, vision, planner, qr_detection, tuning, teleop, cli
from fp.fsm import context, states, runner
from fp.repl import commands, repl, tuning_diff
print("  Smoke import OK:", [
    'main', 'fp.hal', 'fp.arena', 'fp.localization', 'fp.motion', 'fp.navigation',
    'fp.state_machine', 'fp.ekf', 'fp.vision', 'fp.planner', 'fp.qr_detection',
    'fp.tuning', 'fp.teleop', 'fp.cli',
    'fp.fsm.context', 'fp.fsm.states', 'fp.fsm.runner',
    'fp.repl.commands', 'fp.repl.repl', 'fp.repl.tuning_diff',
])
PYEOF
echo "[3/3] smoke import: PASSED"

echo ""
echo "============================================================"
echo "  All checks PASSED"
echo "============================================================"
