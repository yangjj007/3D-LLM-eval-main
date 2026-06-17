from .base_engine import InferenceEngine
from .understanding_engine import UnderstandingEngine
from .generation_engine import GenerationEngine
from .vqvae_engine import VQVAEEngine

ENGINE_REGISTRY = {
    "understanding": UnderstandingEngine,
    "generation": GenerationEngine,
    "vqvae_recon": VQVAEEngine,
}
