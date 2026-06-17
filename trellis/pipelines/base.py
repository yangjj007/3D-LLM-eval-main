from typing import *
import torch
import torch.nn as nn
from .. import models


def _pipeline_model_ref_is_repo_absolute(v: str) -> bool:
    """
    ``pipeline.json`` entries are either repo-relative (``ckpts/...`` under the pipeline repo)
    or a full HF model path (``org/repo/ckpts/...``). Prefixing the pipeline repo to the latter
    yields invalid URLs (404 on Hub).
    """
    v = v.replace("\\", "/").strip()
    parts = v.split("/")
    if len(parts) < 3:
        return False
    if parts[0] == "ckpts":
        return False
    return True


class Pipeline:
    """
    A base class for pipelines.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
    ):
        if models is None:
            return
        self.models = models
        for model in self.models.values():
            model.eval()

    @staticmethod
    def from_pretrained(path: str) -> "Pipeline":
        """
        Load a pretrained model.
        """
        import os
        import json
        is_local = os.path.exists(f"{path}/pipeline.json")

        if is_local:
            config_file = f"{path}/pipeline.json"
        else:
            from huggingface_hub import hf_hub_download
            config_file = hf_hub_download(path, "pipeline.json")

        with open(config_file, 'r') as f:
            args = json.load(f)['args']

        _models = {}
        path = path.replace("\\", "/").rstrip("/")
        for k, v in args['models'].items():
            v = v.replace("\\", "/").strip()
            if _pipeline_model_ref_is_repo_absolute(v):
                _models[k] = models.from_pretrained(v)
            else:
                _models[k] = models.from_pretrained(f"{path}/{v}")

        new_pipeline = Pipeline(_models)
        new_pipeline._pretrained_args = args
        return new_pipeline

    @property
    def device(self) -> torch.device:
        for model in self.models.values():
            if hasattr(model, 'device'):
                return model.device
        for model in self.models.values():
            if hasattr(model, 'parameters'):
                return next(model.parameters()).device
        raise RuntimeError("No device found.")

    def to(self, device: torch.device) -> None:
        for model in self.models.values():
            model.to(device)

    def cuda(self) -> None:
        self.to(torch.device("cuda"))

    def cpu(self) -> None:
        self.to(torch.device("cpu"))
