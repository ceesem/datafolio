import pytest


@pytest.fixture(autouse=True)
def _isolated_folio_registry(tmp_path_factory, monkeypatch):
    """Keep every test away from the real ~/.datafolio registry."""
    home = tmp_path_factory.mktemp("datafolio_home")
    monkeypatch.setenv("DATAFOLIO_HOME", str(home))
    return home
