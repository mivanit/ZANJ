import pytest


def test_assert_no_torch():
    with pytest.raises(ImportError):
        pass
