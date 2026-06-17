"""
Abstract base class for all evaluation datasets.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset


class EvalDataset(Dataset, ABC):
    """
    Base class for evaluation datasets.

    Subclasses must implement __len__, __getitem__, and collate_fn.
    All datasets are configured via a dict parsed from YAML.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.data_path: Optional[str] = config.get("data_path") or None
        self.max_samples: Optional[int] = config.get("max_samples", None)
        self.samples: List[Dict[str, Any]] = []
        self._load_data()

    @abstractmethod
    def _load_data(self) -> None:
        """Load and parse the raw data file into self.samples."""
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ...

    def __len__(self) -> int:
        return len(self.samples)

    def collate_fn(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Default collate: return list of dicts (no tensor stacking)."""
        return batch
