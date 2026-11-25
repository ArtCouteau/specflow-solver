import importlib
import shutil
import subprocess
import zipfile
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_specflow_is_importable() -> None:
    assert importlib.import_module("specflow").__name__ == "specflow"


def test_license_is_declared_as_lgpl_3_or_later() -> None:
    assert metadata.metadata("specflow-solver")["License-Expression"] == (
        "LGPL-3.0-or-later"
    )


def test_wheel_ships_both_license_texts(tmp_path: Path) -> None:
    uv = shutil.which("uv")
    assert uv is not None
    subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(tmp_path), str(ROOT)],
        check=True,
        capture_output=True,
    )
    (wheel,) = tmp_path.glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        names = {
            Path(info.filename).name
            for info in archive.infolist()
            if Path(info.filename).parent.name == "licenses" and not info.is_dir()
        }
    assert names == {"LICENSE", "COPYING.LESSER"}
