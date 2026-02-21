"""Constants and re-exports from muutils with version compatibility."""

from __future__ import annotations

# Items that exist in muutils.json_serialize.util across all versions
from muutils.json_serialize.util import (
    JSONdict,
    JSONitem,
    MonoTuple,
    safe_getsource,
    string_as_lines,
)

# _FORMAT_KEY and _REF_KEY moved from .util to .types in muutils >= 0.9
try:
    from muutils.json_serialize.types import _FORMAT_KEY, _REF_KEY  # type: ignore[import-not-found]
except ImportError:
    from muutils.json_serialize.util import _FORMAT_KEY, _REF_KEY  # type: ignore[import-not-found]

__all__ = [
    "JSONdict",
    "JSONitem",
    "MonoTuple",
    "_FORMAT_KEY",
    "_REF_KEY",
    "safe_getsource",
    "string_as_lines",
]
