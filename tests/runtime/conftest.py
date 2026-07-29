from __future__ import annotations

import sys
from pathlib import Path

# Make `tools` and `contracts`-or-stub importable when pytest is run from the
# repo root (rootdir-relative imports aren't automatic without a src layout).
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    d = tmp_path / "task_workdir"
    d.mkdir()
    return d
