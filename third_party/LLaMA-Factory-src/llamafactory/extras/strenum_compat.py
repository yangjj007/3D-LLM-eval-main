# Copyright 2025 the LlamaFactory team.
# Python 3.10: ``enum.StrEnum`` exists only from 3.11+; re-export a compatible shim.
from enum import unique

__all__ = ("StrEnum", "unique")

try:  # Python 3.11+
    from enum import StrEnum
except ImportError:  # Python 3.10 and below
    from enum import Enum

    class StrEnum(str, Enum):
        pass
