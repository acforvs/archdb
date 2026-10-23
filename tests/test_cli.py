import csv
import io
import json
import os
import subprocess
import sys

import pytest

from archdb.cli import main
from archdb.data import ROOT


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("archdb ")


def test_plain_output_removes_source_terminal_controls(capsys, monkeypatch):
    from archdb import query

    db = query.load()
    model = query.resolve(db, "deepseek-v3")
    control = "\x1b]52;c;clipboard\x07"
    model["kind"] = control
    model["weights"] = control
    db["fields"]["shape.layers"]["unit"] = control
    monkeypatch.setattr("archdb.browse.query.load", lambda *a, **kw: db)
    main(["show", "deepseek-v3", "--fields", "shape.layers", "--evidence"])
    output = capsys.readouterr().out
    assert "61" in output and "\x1b" not in output and "\x07" not in output


def test_errors_remove_source_terminal_controls(capsys, monkeypatch):
    from archdb import query

    db = query.load()
    for model in db["models"][:2]:
        model["name"] = "ambiguous"
        model["id"] += "\x1b[31m"
    monkeypatch.setattr("archdb.browse.query.load", lambda *a, **kw: db)
    with pytest.raises(SystemExit) as exc:
        main(["show", "ambiguous"])
    assert exc.value.code == 2
    assert "\x1b" not in capsys.readouterr().err


def test_search_json_contains_raw_values_and_status(capsys):
    main(["models", "deepseek-v3", "--fields", "params.total,params.active", "--format", "json"])
    result = json.loads(capsys.readouterr().out)
    m = next(m for m in result if m["id"] == "deepseek-v3")
    assert m["params.total"] == 671e9
    assert m["_status"]["params.total"] == "reported"


def test_show_full_evidence_in_json(capsys):
    main(["show", "deepseek-v3", "--fields", "shape.layers", "--evidence", "--format", "json"])
    m = json.loads(capsys.readouterr().out)
    evidence = m["_evidence"]["shape.layers"]
    assert evidence["value"] == 61
    assert "huggingface.co" in evidence["source"]["url"]
    assert evidence["at"] == "/num_hidden_layers"


def test_csv_preserves_values_status_and_kv_assumptions(capsys):
    main(
        [
            "show",
            "deepseek-v41",
            "--fields",
            "kv.bytes_per_token",
            "--precision",
            "sourced",
            "--format",
            "csv",
        ]
    )
    row = next(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert float(row["kv.bytes_per_token"]) == 910.625
    assert json.loads(row["_status"])["kv.bytes_per_token"] == "derived"
    assert json.loads(row["_kv"]) == {"context": 131072, "precision": "sourced"}


def test_evidence_matches_selected_precision_and_context(capsys):
    main(
        [
            "evidence",
            "deepseek-v41",
            "kv.bytes_at_context",
            "--context",
            "262144",
            "--precision",
            "sourced",
            "--format",
            "json",
        ]
    )
    evidence = json.loads(capsys.readouterr().out)
    assert evidence["value"] == 890 * 262144 + 40 * 128 * 528
    assert "sourced component" in evidence["formula"]


def test_fields_can_be_searched_within_group(capsys):
    main(["fields", "expert", "--group", "moe", "--format", "json"])
    fields = json.loads(capsys.readouterr().out)
    assert "moe.expert_dim" in [f["key"] for f in fields]
    assert all(f["key"].startswith("moe.") for f in fields)


@pytest.mark.parametrize(
    "argv",
    [
        ["show", "deepseek"],
        ["compare", "deepseek-v3"],
        ["models", "--where", "typo >= 4"],
        ["show", "deepseek-v3", "--group", "typo"],
        ["models", "--context", "0"],
        ["models", "--limit", "-1"],
        ["models", "--fields", "params.total", "--group", "attn"],
        ["show", "deepseek-v3", "--fields", ",,"],
    ],
)
def test_invalid_requests_exit_with_a_message(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_research_commands_need_no_third_party_imports():
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "archdb",
            "compare",
            "deepseek-v3",
            "kimi-k2",
            "--fields",
            "moe.expert_dim",
            "--format",
            "json",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": ""},
        capture_output=True,
        text=True,
        check=True,
    )
    assert len(json.loads(result.stdout)) == 2


def test_explorer_requires_a_terminal_instead_of_dumping_escape_sequences():
    result = subprocess.run(
        [sys.executable, "-S", "-m", "archdb", "explore"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "interactive terminal" in result.stderr
    assert "\x1b" not in result.stdout
