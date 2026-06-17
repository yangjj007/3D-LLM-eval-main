"""
Abstract base class for inference engines.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from ..model_loader import ModelBundle


class InferenceEngine(ABC):
    """
    Base class for task-specific inference engines.

    Each engine takes a ModelBundle and config, then processes evaluation samples.
    """

    def __init__(self, model_bundle: ModelBundle, config: Dict[str, Any]) -> None:
        self.models = model_bundle
        self.config = config

    @abstractmethod
    def run(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run inference on a list of samples.

        Args:
            samples: List of sample dicts from the dataset.

        Returns:
            List of result dicts, each containing at minimum 'sample_id'
            and task-specific prediction fields.
        """
        ...
