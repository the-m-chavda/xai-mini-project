#!/usr/bin/env bash
# reproduce.sh — one-shot reproducibility script for the XAI mini project
# Usage: bash reproduce.sh
set -euo pipefail

PASS=0
FAIL=0
ERRORS=()

ok()   { echo "[OK]   $1"; PASS=$((PASS+1)); }
fail() { echo "[FAIL] $1"; ERRORS+=("$1"); FAIL=$((FAIL+1)); }
step() { echo; echo "=== $1 ==="; }

# ── Step 1: Python version ────────────────────────────────────────────────────
step "Checking Python version"
PY=$(python3 --version 2>&1 | awk '{print $2}')
MAJOR=$(echo "$PY" | cut -d. -f1)
MINOR=$(echo "$PY" | cut -d. -f2)
if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 10 ]; then
    ok "Python $PY (>= 3.10)"
else
    fail "Python $PY is too old — need 3.10 or newer"
    exit 1
fi

# ── Step 2: Virtual environment ───────────────────────────────────────────────
step "Setting up virtual environment"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    ok "Created .venv"
else
    ok ".venv already exists"
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# ── Step 3: Install dependencies ──────────────────────────────────────────────
step "Installing dependencies"
pip install --upgrade pip -q
pip install -r requirements.txt -q && ok "requirements.txt installed"
pip install -e . -q              && ok "Package installed (xai-mini command available)"

# ── Step 4: Install Ontolearn (optional but recommended) ─────────────────────
step "Installing Ontolearn / CELOE (optional)"
if pip install -r requirements-ontolearn.txt -q 2>/dev/null && \
   pip install --no-deps ontolearn==0.10.0 -q 2>/dev/null && \
   python3 -c "import ontolearn" 2>/dev/null; then
    ok "Ontolearn installed — CELOE will be used for explanations"
else
    echo "  [INFO] Ontolearn not available — fallback baseline explainer will run"
fi

# ── Step 5: Verify data files ────────────────────────────────────────────────
step "Checking dataset files"
for f in data/aifb/aifbfixed_complete.n3 \
          data/aifb/trainingSet.tsv \
          data/aifb/testSet.tsv \
          data/aifb/completeDataset.tsv; do
    if [ -f "$f" ]; then ok "$f"; else fail "Missing: $f"; fi
done

if [ "$FAIL" -gt 0 ]; then
    echo; echo "Aborting — missing data files."; exit 1
fi

# ── Step 6: Compile check ─────────────────────────────────────────────────────
step "Syntax check"
python3 -m compileall -q src && ok "All source files compile cleanly"

# ── Step 7: Unit tests ───────────────────────────────────────────────────────
step "Running unit tests"
pip install pytest -q
if python3 -m pytest -q; then
    ok "All tests passed"
else
    fail "Unit tests failed"
fi

# ── Step 8: Full pipeline ─────────────────────────────────────────────────────
step "Running full pipeline (analyze + train + explain)"
echo "  This takes ~1–2 minutes on CPU."
xai-mini --config configs/aifb.yaml --no-log run-all
ok "Pipeline completed — artifacts written to artifacts/aifb/"

# ── Step 9: Verify results ───────────────────────────────────────────────────
step "Verifying results"
METRICS="artifacts/aifb/metrics.json"
EXPLANATIONS="artifacts/aifb/explanation_results.json"

if [ ! -f "$METRICS" ]; then
    fail "metrics.json not found"
else
    TEST_ACC=$(python3 -c "import json; d=json.load(open('$METRICS')); print(d['test_accuracy'])")
    TEST_F1=$(python3  -c "import json; d=json.load(open('$METRICS')); print(d['test_macro_f1'])")

    # Accept test_accuracy in range [0.92, 0.97]
    PASS_ACC=$(python3 -c "print('yes' if 0.92 <= float('$TEST_ACC') <= 0.97 else 'no')")
    if [ "$PASS_ACC" = "yes" ]; then
        ok "Test accuracy = $TEST_ACC  (expected ~94.4%)"
    else
        fail "Test accuracy = $TEST_ACC  — outside expected range [0.92, 0.97]"
    fi

    # Accept test_macro_f1 in range [0.88, 0.95]
    PASS_F1=$(python3 -c "print('yes' if 0.88 <= float('$TEST_F1') <= 0.95 else 'no')")
    if [ "$PASS_F1" = "yes" ]; then
        ok "Test macro-F1 = $TEST_F1  (expected ~0.912)"
    else
        fail "Test macro-F1 = $TEST_F1  — outside expected range [0.88, 0.95]"
    fi
fi

if [ ! -f "$EXPLANATIONS" ]; then
    fail "explanation_results.json not found"
else
    NUM=$(python3 -c "import json; d=json.load(open('$EXPLANATIONS')); print(len(d['results']))")
    if [ "$NUM" -eq 4 ]; then
        ok "Explanation results: $NUM classes explained"
    else
        fail "Explanation results: expected 4 classes, got $NUM"
    fi
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo
echo "════════════════════════════════════════"
echo "  Results"
echo "════════════════════════════════════════"
echo "  Passed : $PASS"
echo "  Failed : $FAIL"
if [ "$FAIL" -gt 0 ]; then
    echo
    echo "  Failures:"
    for e in "${ERRORS[@]}"; do echo "    - $e"; done
    echo
    exit 1
else
    echo
    echo "  All checks passed."
    echo "  Key outputs:"
    echo "    artifacts/aifb/metrics.json            — accuracy & F1"
    echo "    artifacts/aifb/predictions.csv         — per-person predictions"
    echo "    artifacts/aifb/explanation_results.json — DL concepts per class"
    echo "════════════════════════════════════════"
fi
