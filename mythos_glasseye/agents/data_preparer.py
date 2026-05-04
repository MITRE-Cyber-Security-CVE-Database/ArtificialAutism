import logging
from pathlib import Path
from typing import Optional

from .base import BaseAgent, Result

logger = logging.getLogger(__name__)

class DataPreparerAgent(BaseAgent):
    """Agent that prepares datasets for training.

    Supports streaming from HuggingFace (FineWeb-Edu) or local paths.
    """

    class Config(BaseAgent.Config):
        source: str = "HuggingFaceFW/fineweb-edu"
        subset: str = "sample-10BT"
        split: str = "train"
        cache_dir: Optional[str] = "data/cache"

    def run(self, **kwargs) -> Result:
        cfg = self.validate(**kwargs)
        
        # In a real scenario, this might download shards or tokenize.
        # For now, we verify if the source is accessible (e.g., via datasets)
        # or just prepare the metadata.
        
        try:
            from datasets import load_dataset_builder
            builder = load_dataset_builder(cfg.source, name=cfg.subset)
            info = builder.info
            
            logger.info(f"Dataset {cfg.source} ({cfg.subset}) found. Description: {info.description[:100]}...")
            
            return Result(
                success=True,
                metadata={
                    "source": cfg.source,
                    "subset": cfg.subset,
                    "split": cfg.split,
                    "features": str(info.features)
                }
            )
        except Exception as e:
            logger.error(f"Failed to prepare dataset: {e}")
            return Result(success=False, metadata={"error": str(e)})
