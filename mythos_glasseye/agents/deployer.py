import logging
import subprocess
from pathlib import Path
from typing import Optional, Any

from .base import BaseAgent, Result

logger = logging.getLogger(__name__)

class DeployerAgent(BaseAgent):
    """Agent that deploys the model to a serving environment.

    Handles Docker builds or simple process management.
    """

    class Config(BaseAgent.Config):
        model_path: Any = "artifacts/openmythos_0ai_cpu_dev.pt"
        port: int = 5000
        mode: str = "local" # or "docker"
        verify_integrity: bool = True

    def run(self, **kwargs) -> Result:
        cfg = self.validate(**kwargs)
        
        if cfg.mode == "local":
            logger.info(f"Deploying model {cfg.model_path} on port {cfg.port} (local mode)...")
            
            model_path = Path(str(cfg.model_path))
            if not model_path.exists():
                return Result(success=False, metadata={"error": f"Model not found at {model_path}"})
                
            metadata = {
                "status": "ready_to_serve",
                "command": f"python3 -m open_mythos.cli serve --model {model_path} --port {cfg.port}"
            }
            
            if cfg.verify_integrity:
                logger.info(f"Verifying integrity of {model_path}...")
                try:
                    import torch
                    # Map to CPU to avoid CUDA errors on load
                    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
                    metadata["verified"] = True
                    # If it's a dict containing a 'model' key (like our training loop outputs)
                    if isinstance(checkpoint, dict):
                        metadata["checkpoint_step"] = checkpoint.get("step", "unknown")
                        metadata["checkpoint_keys"] = list(checkpoint.keys())
                except Exception as e:
                    logger.error(f"Failed to verify model integrity: {e}")
                    return Result(success=False, metadata={"error": f"Integrity check failed: {e}"})
            
            return Result(success=True, metadata=metadata)
            
        elif cfg.mode == "docker":
            logger.info("Docker deployment requested (not fully implemented in MVP).")
            return Result(success=False, metadata={"error": "Docker mode not yet implemented."})
        
        return Result(success=False, metadata={"error": f"Unknown mode: {cfg.mode}"})
