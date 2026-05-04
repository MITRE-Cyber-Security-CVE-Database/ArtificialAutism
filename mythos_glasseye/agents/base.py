import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

class Result(BaseModel):
    success: bool
    output_path: Optional[Path] = None
    metadata: Dict[str, Any] = {}

class BaseAgent:
    """Abstract base class for all agents."""

    class Config(BaseModel):
        """Placeholder – each subclass defines its own fields."""
        pass

    def __init__(self, **init_kwargs):
        self.init_kwargs = init_kwargs
        logger.info(f"{self.__class__.__name__} initialized with {init_kwargs}")

    def validate(self, **kwargs) -> BaseModel:
        try:
            cfg = self.Config(**kwargs)
            return cfg
        except ValidationError as exc:
            logger.error(f"Configuration validation error for {self.__class__.__name__}: {exc}")
            raise

    def run(self, **kwargs) -> Result:
        raise NotImplementedError("Sub‑classes must implement the run method")
