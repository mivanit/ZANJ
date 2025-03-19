"""
.. include:: ../README.md
"""

from __future__ import annotations

from zanj.loading import register_loader_handler
from zanj.zanj import ZANJ

__all__ = [
    "register_loader_handler",
    "ZANJ",
    # modules
    "externals",
    "loading",
    "serializing",
    "torchutil",
    "zanj",
]
