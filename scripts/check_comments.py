"""Fail on any comment token in Python sources.

Usage: ``python scripts/check_comments.py [PATH ...]``. Each path is a file or a
directory searched recursively for ``*.py`` files. Without paths, the ``src``,
``tests``, ``cases`` and ``scripts`` directories of the working directory that
exist are checked. Exits with 1 when a comment is found and 2 when a path is
missing or a file cannot be tokenized.
"""

import sys
import tokenize
from collections.abc import Iterator, Sequence
from pathlib import Path

DEFAULT_DIRECTORIES = ("src", "tests", "cases", "scripts")


def python_files(paths: Sequence[Path]) -> Iterator[Path]:
    """Yield the Python files of the given files and directories, sorted."""
    for path in paths:
        if path.is_dir():
            yield from sorted(path.rglob("*.py"))
        else:
            yield path


def comments(path: Path) -> Iterator[tuple[int, int, str]]:
    """Yield the line, column and text of each comment token of a file."""
    with tokenize.open(path) as source:
        for token in tokenize.generate_tokens(source.readline):
            if token.type == tokenize.COMMENT:
                yield token.start[0], token.start[1], token.string


def main(arguments: Sequence[str]) -> int:
    """Report every comment under the given paths and return the exit code."""
    if arguments:
        paths = [Path(argument) for argument in arguments]
    else:
        paths = [Path(name) for name in DEFAULT_DIRECTORIES if Path(name).is_dir()]
    missing = [path for path in paths if not path.exists()]
    for path in missing:
        print(f"{path}: no such file or directory")
    if missing:
        return 2
    found = False
    for path in python_files(paths):
        try:
            for line, column, text in comments(path):
                print(f"{path}:{line}:{column + 1}: comment {text!r}")
                found = True
        except (SyntaxError, tokenize.TokenError) as error:
            print(f"{path}: cannot tokenize: {error}")
            return 2
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
