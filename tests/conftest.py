import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.db import ProphitDB  # noqa: E402

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _no_network(socket_disabled):
    """Hard guarantee: no test may open a real socket."""
    yield


@pytest.fixture
def db(tmp_path):
    database = ProphitDB(tmp_path / "prophit_test.db")
    yield database
    database.close()


@pytest.fixture
def load_fixture():
    def _load(name: str):
        return json.loads((FIXTURES_DIR / name).read_text())

    return _load
