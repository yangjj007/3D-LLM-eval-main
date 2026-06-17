from .base_dataset import EvalDataset
from .understanding_dataset import UnderstandingDataset
from .generation_dataset import GenerationDataset
from .vqvae_dataset import VQVAEDataset

DATASET_REGISTRY = {
    "understanding": UnderstandingDataset,
    "generation": GenerationDataset,
    "vqvae_recon": VQVAEDataset,
    "sparse_mesh": GenerationDataset,
}
