# Python 3.10: ``NotRequired`` and ``Self`` live in ``typing`` only from 3.11+.
import sys

if sys.version_info >= (3, 11):
    from typing import NotRequired, Self
else:
    from typing_extensions import NotRequired, Self

__all__ = ("NotRequired", "Self")
