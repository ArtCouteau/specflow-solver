import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_comments.py"


def run_check(*args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def test_a_comment_fails_with_its_location(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    source.write_text('x = 1\ny = "#"  # trailing\n')
    result = run_check(str(tmp_path))
    assert result.returncode == 1
    assert f"{source}:2:" in result.stdout


def test_a_shebang_counts_as_a_comment(tmp_path: Path) -> None:
    (tmp_path / "script.py").write_text("#!/usr/bin/env python\nx = 1\n")
    assert run_check(str(tmp_path)).returncode == 1


def test_hashes_inside_strings_pass(tmp_path: Path) -> None:
    (tmp_path / "module.py").write_text('x = "# not a comment"\ny = f"{x}#"\n')
    result = run_check(str(tmp_path))
    assert result.returncode == 0, result.stdout


def test_single_files_can_be_checked(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    source.write_text("x = 1  # comment\n")
    assert run_check(str(source)).returncode == 1


def test_a_missing_path_is_an_error(tmp_path: Path) -> None:
    assert run_check(str(tmp_path / "missing")).returncode == 2


def test_repository_sources_are_comment_free() -> None:
    result = run_check()
    assert result.returncode == 0, result.stdout
