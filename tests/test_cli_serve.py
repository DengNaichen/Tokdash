"""Tests for `tokdash serve` browser auto-open behavior and the --no-open flag."""
import sys

import tokdash.cli as cli


def test_no_open_flag_defaults_false():
    args = cli.build_parser("tokdash").parse_args(["serve"])
    assert args.no_open is False


def test_no_open_flag_sets_true():
    args = cli.build_parser("tokdash").parse_args(["serve", "--no-open"])
    assert args.no_open is True


def test_has_display_false_under_ssh(monkeypatch):
    monkeypatch.setenv("SSH_CONNECTION", "10.0.0.1 22 10.0.0.2 22")
    assert cli._has_display() is False


def test_has_display_linux_requires_display(monkeypatch):
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert cli._has_display() is False
    monkeypatch.setenv("DISPLAY", ":0")
    assert cli._has_display() is True


def test_has_display_non_linux_assumes_gui(monkeypatch):
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delenv("DISPLAY", raising=False)
    assert cli._has_display() is True


def test_open_browser_swallows_errors(monkeypatch):
    def boom(url):
        raise RuntimeError("no browser available")

    monkeypatch.setattr(cli.webbrowser, "open", boom)
    # Best-effort: must not propagate.
    cli._open_browser("http://localhost:55423")


def _patch_serve(monkeypatch, *, has_display):
    """Stub uvicorn + threading.Timer; return a list that records timer starts."""
    started: list[str] = []
    monkeypatch.setattr(cli.uvicorn, "run", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_has_display", lambda: has_display)

    class FakeTimer:
        def __init__(self, _interval, _func, args=(), **_kwargs):
            self._args = args
            self.daemon = False

        def start(self):
            started.append(self._args[0])

    monkeypatch.setattr(cli.threading, "Timer", FakeTimer)
    return started


def test_serve_opens_browser_when_enabled_and_display(monkeypatch):
    started = _patch_serve(monkeypatch, has_display=True)
    cli.serve("127.0.0.1", 55423, "info", open_browser=True)
    assert started == ["http://127.0.0.1:55423"]


def test_serve_skips_browser_when_disabled(monkeypatch):
    started = _patch_serve(monkeypatch, has_display=True)
    cli.serve("127.0.0.1", 55423, "info", open_browser=False)
    assert started == []


def test_serve_skips_browser_when_headless(monkeypatch):
    started = _patch_serve(monkeypatch, has_display=False)
    cli.serve("127.0.0.1", 55423, "info", open_browser=True)
    assert started == []
