"""CLI skeleton: version, usage errors, and the exit-code contract."""

import subprocess
import sys

import pytest

from slotvox._version import __version__
from slotvox.cli import main
from slotvox.cli.common import EXIT_DATA, EXIT_RUNTIME, EXIT_USAGE, run_command
from slotvox.errors import AdapterError, SchemaError, ValidationError


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as info:
        main(["--version"])
    assert info.value.code == 0
    out = capsys.readouterr().out
    assert __version__ in out
    assert "slotvox" in out


def test_module_entrypoint_reports_version():
    result = subprocess.run(
        [sys.executable, "-m", "slotvox", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == f"slotvox {__version__}"


def test_no_command_prints_usage_and_returns_2(capsys):
    assert main([]) == EXIT_USAGE
    captured = capsys.readouterr()
    assert "COMMAND" in captured.err


def test_unknown_command_exits_2():
    with pytest.raises(SystemExit) as info:
        main(["definitely-not-a-command"])
    assert info.value.code == 2


def test_run_command_maps_errors(capsys):
    class Args:
        pass

    def raising(exc):
        def handler(_args):
            raise exc

        return handler

    assert run_command(raising(ValidationError("bad")), Args()) == EXIT_USAGE
    assert run_command(raising(SchemaError("bad")), Args()) == EXIT_DATA
    assert run_command(raising(AdapterError("bad")), Args()) == EXIT_RUNTIME
    assert capsys.readouterr().err.count("error: bad") == 3
