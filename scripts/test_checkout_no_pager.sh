#!/usr/bin/env bash
# Reproduce an interactive Git pager in a real PTY, then check the workflow command.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
"${PYTHON_BIN:-python3}" - "$ROOT" <<'PYTEST'
import fcntl
import os
import pathlib
import pty
import select
import shlex
import shutil
import signal
import struct
import subprocess
import sys
import termios
import time

root = pathlib.Path(sys.argv[1])
text = (root / ".github/workflows/alignment_model_accuracy.yaml").read_text()
line = next(line.strip() for line in text.splitlines() if "checkout_step swift-history " in line)
actual = shlex.split(line)[3:]
assert actual == ["git", "--no-pager", "log", "--pretty=oneline", "-10"], actual
assert shutil.which("less"), "PTY regression fixture requires less"

def observe(command):
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 8, 60, 0, 0))
    env = dict(os.environ, TERM="xterm", GIT_PAGER="less", LESS="-FRX")
    proc = subprocess.Popen(command, cwd=root, env=env, stdin=slave, stdout=slave,
                            stderr=slave, start_new_session=True)
    os.close(slave)
    output = bytearray()
    try:
        until = time.monotonic() + 2
        while time.monotonic() < until and proc.poll() is None:
            if select.select([master], [], [], 0.05)[0]:
                try:
                    output.extend(os.read(master, 65536))
                except OSError:
                    break
        # PTY EOF can arrive just before the child is reaped.
        try:
            proc.wait(timeout=0.2)
            needed_input = False
        except subprocess.TimeoutExpired:
            needed_input = True
        if needed_input:
            os.write(master, b"q")
        result = proc.wait(timeout=3)
        assert result == 0, (result, bytes(output)[-200:])
        assert output, "expected Git history output"
        return needed_input
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        os.close(master)

assert observe(["git", "log", "--pretty=oneline", "-10"]), "old command should await pager input in PTY"
print("PASS: old git log waits for input in PTY and exits on q")
assert not observe(actual), "workflow Git history must exit without input"
print("PASS: actual workflow --no-pager command exits without input in same PTY")
PYTEST
