"""Exercise ncurses in a real PTY, including terminal cleanup and color output."""

import os
import select
import signal
import struct
import sys
import time

import pytest

from archdb.data import ROOT


@pytest.mark.skipif(os.name != "posix", reason="terminal explorer uses POSIX curses")
@pytest.mark.parametrize("quit_key,exit_code", [(b"q", 0), (b"\x03", 130)])
def test_terminal_navigation_colors_and_cleanup(quit_key, exit_code):
    import fcntl
    import pty
    import termios

    pid, descriptor = pty.fork()
    if pid == 0:
        os.chdir(ROOT)
        os.execve(
            sys.executable,
            [sys.executable, "-S", "-m", "archdb"],
            {"PATH": "/usr/bin:/bin", "TERM": "xterm-256color", "LANG": "en_US.UTF-8"},
        )
    fcntl.ioctl(descriptor, termios.TIOCSWINSZ, struct.pack("HHHH", 36, 110, 0, 0))
    output = bytearray()
    exited = False

    def wait_for(text):
        deadline = time.monotonic() + 5
        start = len(output)
        while time.monotonic() < deadline:
            if select.select([descriptor], [], [], 0.02)[0]:
                try:
                    chunk = os.read(descriptor, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                output.extend(chunk)
                if text in output[start:]:
                    return
        pytest.fail(f"terminal did not produce {text!r}; tail: {bytes(output[-500:])!r}")

    try:
        wait_for(b"s Sort:")
        # A real ncurses regression: setting a background color pair erased every
        # per-cell color on macOS, while the pure rendering tests still passed.
        assert b"38;5;81" in output
        assert b"48;5;237" in output
        os.write(descriptor, b" j 2")
        wait_for(b"Models 1")
        os.write(descriptor, b"3")
        wait_for(b"KV bytes / token")
        os.write(descriptor, b"v")
        wait_for(b"Fields")
        os.write(descriptor, b"n\n" + quit_key)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if select.select([descriptor], [], [], 0.02)[0]:
                try:
                    output.extend(os.read(descriptor, 65536))
                except OSError:
                    pass
            done, status = os.waitpid(pid, os.WNOHANG)
            if done:
                exited = True
                assert os.waitstatus_to_exitcode(status) == exit_code
                assert b"\x1b[?1049l" in output
                assert b"Traceback" not in output
                break
        assert exited, "terminal did not exit on q"
    finally:
        if not exited:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            os.waitpid(pid, 0)
        os.close(descriptor)
