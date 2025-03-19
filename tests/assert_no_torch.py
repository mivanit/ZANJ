import pytest


def test_assert_no_torch():
    with pytest.raises(ImportError):
        import torch  # type: ignore[import-not-found]

        print(torch.rand(10))
