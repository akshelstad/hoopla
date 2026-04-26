import sys

import pytest

import cli.keyword_search_cli as keyword_search_cli


def test_cli_exits_with_code_2_for_expected_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        keyword_search_cli,
        "search_command",
        lambda query: (_ for _ in ()).throw(keyword_search_cli.MissingIndexError("build first")),
    )
    monkeypatch.setattr(sys, "argv", ["keyword_search_cli.py", "search", "alien"])

    with pytest.raises(SystemExit) as exc_info:
        keyword_search_cli.main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "Error: build first" in captured.err


def test_cli_exits_with_code_3_for_unexpected_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        keyword_search_cli,
        "search_command",
        lambda query: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(sys, "argv", ["keyword_search_cli.py", "search", "alien"])

    with pytest.raises(SystemExit) as exc_info:
        keyword_search_cli.main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 3
    assert "Unexpected internal error: boom" in captured.err
