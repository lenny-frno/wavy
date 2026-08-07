import pytest
from pathlib import Path


def pytest_collection_modifyitems(config, items):
    """Skip tests marked need_credentials unless explicitly selected with -m need_credentials."""
    if "need_credentials" not in config.option.markexpr:
        skip_marker = pytest.mark.skip(
            reason="requires credentials; run with -m need_credentials to include"
        )
        for item in items:
            if item.get_closest_marker("need_credentials"):
                item.add_marker(skip_marker)


@pytest.fixture
def test_data():
    return Path(__file__).parent.resolve().joinpath("data")
