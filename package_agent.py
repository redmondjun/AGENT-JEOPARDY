"""Build and validate the deployable agent.zip from an explicit allowlist."""
from __future__ import annotations

import ast
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "agent.zip"
PACKAGE_FILES = (
    "main.py",
    "jeopardy.py",
    "solver.py",
    "tools.py",
    "requirements.txt",
    "calibration.json",
)
MAX_COMPRESSED = 20 * 1024 * 1024
MAX_UNCOMPRESSED = 200 * 1024 * 1024


def imported_local_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return {
        name
        for name in names
        if (ROOT / f"{name}.py").exists()
    }


def validate_sources() -> None:
    missing = [name for name in PACKAGE_FILES if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit(f"missing package files: {missing}")
    packaged_modules = {Path(name).stem for name in PACKAGE_FILES}
    required: set[str] = set()
    for name in PACKAGE_FILES:
        path = ROOT / name
        if path.suffix == ".py":
            required.update(imported_local_modules(path))
    omitted = sorted(required - packaged_modules)
    if omitted:
        raise SystemExit(f"local imports missing from package allowlist: {omitted}")


def build() -> Path:
    validate_sources()
    with zipfile.ZipFile(
        OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in PACKAGE_FILES:
            archive.write(ROOT / name, arcname=name)
    with zipfile.ZipFile(OUTPUT) as archive:
        names = archive.namelist()
        if "main.py" not in names or any(name.startswith("/") for name in names):
            raise SystemExit("main.py must be at the ZIP root")
        uncompressed = sum(info.file_size for info in archive.infolist())
    if OUTPUT.stat().st_size > MAX_COMPRESSED:
        raise SystemExit("agent.zip exceeds the 20 MB compressed limit")
    if uncompressed > MAX_UNCOMPRESSED:
        raise SystemExit("agent.zip exceeds the 200 MB uncompressed limit")
    print(
        f"built {OUTPUT} with {len(PACKAGE_FILES)} files "
        f"({OUTPUT.stat().st_size} compressed bytes, {uncompressed} uncompressed)"
    )
    return OUTPUT


if __name__ == "__main__":
    build()
