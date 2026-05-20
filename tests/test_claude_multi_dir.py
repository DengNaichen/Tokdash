from pathlib import Path

import tokdash.sessions as sessions
from tokdash.pricing import PricingDatabase
from tokdash.sources.coding_tools import BaseParser, ClaudeParser, _sig_cache


def _write_claude_session(root: Path, session_id: str, message_id: str, tokens: int) -> None:
    session_dir = root / "projects" / "project"
    session_dir.mkdir(parents=True)
    session_file = session_dir / f"{session_id}.jsonl"
    session_file.write_text(
        (
            "{"
            f'"sessionId":"{session_id}",'
            '"cwd":"/work/project",'
            '"timestamp":"2026-05-19T12:00:00Z",'
            '"message":{'
            '"role":"assistant",'
            f'"id":"{message_id}",'
            '"model":"claude-sonnet-4.5",'
            f'"usage":{{"input_tokens":{tokens},"output_tokens":5}}'
            "}"
            "}\n"
        ),
        encoding="utf-8",
    )


def test_claude_parser_reads_all_claude_project_directories(monkeypatch, tmp_path):
    _write_claude_session(tmp_path / ".claude", "base-session", "msg-base", 11)
    _write_claude_session(tmp_path / ".claude-opus", "opus-session", "msg-opus", 22)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _sig_cache.clear()
    BaseParser._entry_cache.clear()

    entries = ClaudeParser(PricingDatabase()).collect(None, None)

    assert sorted(entry["input"] for entry in entries) == [11, 22]
    assert {entry["source"] for entry in entries} == {"claude"}


def test_claude_session_drilldown_reads_all_claude_project_directories(monkeypatch, tmp_path):
    _write_claude_session(tmp_path / ".claude", "base-session", "msg-base", 11)
    _write_claude_session(tmp_path / ".claude-opus", "opus-session", "msg-opus", 22)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    sessions._parse_claude_session_file.cache_clear()
    sessions._load_claude_sessions.cache_clear()

    raw = sessions._claude_sessions()

    assert sorted(raw) == ["base-session", "opus-session"]
    assert raw["base-session"]["turns"][0]["tokens_in"] == 11
    assert raw["opus-session"]["turns"][0]["tokens_in"] == 22
