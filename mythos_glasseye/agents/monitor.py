import logging
import time
import re
from pathlib import Path
from typing import Dict, Any, Optional

from .base import BaseAgent, Result

logger = logging.getLogger(__name__)

class MonitorAgent(BaseAgent):
    """Agent that monitors training logs and extracts metrics."""

    class Config(BaseAgent.Config):
        log_path: str = "training.log"
        interval: int = 5
        parse_metrics: bool = True

    def run(self, **kwargs) -> Result:
        cfg = self.validate(**kwargs)
        
        log_file = Path(cfg.log_path)
        if not log_file.exists():
            return Result(success=False, metadata={"error": f"Log file not found: {cfg.log_path}"})
            
        logger.info(f"Monitoring log: {cfg.log_path}")
        
        try:
            with open(log_file, "r") as f:
                lines = f.readlines()
                last_lines = lines[-20:] # Read a bit more to ensure we catch a metric line
            
            metadata: Dict[str, Any] = {
                "last_update": time.ctime(),
                "recent_logs": last_lines[-5:] # Keep last 5 lines for context
            }
            
            if cfg.parse_metrics and lines:
                # Regex to match: step   1000/3000 | loss 4.5678 | gnorm 1.23 | lr 3.00e-04
                metrics_pattern = re.compile(
                    r"step\s+(?P<step>\d+)/\d+\s+\|\s+loss\s+(?P<loss>[\d.]+)\s+\|\s+gnorm\s+(?P<gnorm>[\d.]+)\s+\|\s+lr\s+(?P<lr>[\de.-]+)"
                )
                
                # Search backwards for the latest metric line
                for line in reversed(last_lines):
                    match = metrics_pattern.search(line)
                    if match:
                        metrics = match.groupdict()
                        metadata["metrics"] = {
                            "step": int(metrics["step"]),
                            "loss": float(metrics["loss"]),
                            "gnorm": float(metrics["gnorm"]),
                            "lr": float(metrics["lr"])
                        }
                        
                        # Add a health warning if loss is NaN or too high
                        if metadata["metrics"]["loss"] > 15.0 or metadata["metrics"]["loss"] != metadata["metrics"]["loss"]: # NaN check
                            metadata["health_warning"] = "Loss is abnormally high or NaN!"
                            
                        break
                
                if "metrics" not in metadata:
                    metadata["status"] = "No recent metrics found in logs."
            
            return Result(success=True, metadata=metadata)
            
        except Exception as e:
            return Result(success=False, metadata={"error": str(e)})
