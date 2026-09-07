#!/usr/bin/env bash
# Exercise the checkout wrapper extracted from the actual workflow, without network access.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" - "$ROOT/.github/workflows/alignment_model_accuracy.yaml" <<'PYTEST'
import pathlib
import re
import shlex
import subprocess
import sys
import tempfile
import textwrap
import time

text = pathlib.Path(sys.argv[1]).read_text()
checkout = text.split("      - name: Checkout Code\n", 1)[1].split("      - name:", 1)[0]
match = re.search(r"^            checkout_step\(\) \{\n.*?^            \}\n", checkout, re.M | re.S)
assert match, "actual checkout wrapper missing"
helper = textwrap.dedent(match.group())
subprocess.run(["bash", "-n"], input=helper, text=True, check=True)
assert "set -eo pipefail" in checkout
assert not re.search(r"^\s*(?:wget |git (?:pull|fetch|submodule) )", checkout, re.M), "unbounded network command"
assert 'test "$(git rev-parse HEAD)" = "$COMMIT_ID"' in checkout, "exact HEAD check removed"

secret = "fixture-credential-must-not-be-logged"
def run(label, limit, command):
    invocation = shlex.join(["checkout_step", label, limit, *command, secret])
    script = helper + "\nset -e\n" + invocation + "\necho reached-next-step\n"
    started = time.monotonic()
    result = subprocess.run(["bash"], input=script, text=True, capture_output=True, timeout=8)
    elapsed = time.monotonic() - started
    output = result.stdout + result.stderr
    assert secret not in output, output
    assert "checkout begin: " + label in output, output
    assert "checkout end: " + label in output, output
    return result.returncode, output, elapsed

rc, output, _ = run("success", "2s", ["bash", "-c", "exit 0"])
assert rc == 0 and "status=0" in output and "reached-next-step" in output, output
print("PASS: success logs begin/end and continues without printing arguments")

rc, output, _ = run("failure", "2s", ["bash", "-c", "exit 7"])
assert rc == 7 and "checkout failed: failure status=7" in output and "reached-next-step" not in output, output
print("PASS: command failure preserves exit status and stops checkout")

with tempfile.TemporaryDirectory(prefix="checkout-timeout-") as directory:
    marker = pathlib.Path(directory) / "marker"
    command = 'printf started > "$1"; sleep 20; printf finished > "$1"'
    rc, output, elapsed = run("timeout", "0.2s", ["bash", "-c", command, "fixture", str(marker)])
    assert marker.read_text() == "started", output
    assert rc == 124 and elapsed < 5, (rc, elapsed, output)
    assert "checkout timed out: timeout" in output and "reached-next-step" not in output, output
print("PASS: actual timeout ends a running command and prevents continuation")

rc, output, _ = run("killed", "2s", ["bash", "-c", "exit 137"])
assert rc == 137 and "checkout terminated: killed" in output and "checkout timed out:" not in output, output
print("PASS: exit 137 is not falsely identified as a proven timeout")
print("All checkout observability fixtures passed")
PYTEST
