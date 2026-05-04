import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from .base import BaseAgent, Result

logger = logging.getLogger(__name__)

class TrainerAgent(BaseAgent):
    """Agent that handles model training/fine-tuning.

    Wraps execution of training scripts (e.g., training/3b_fine_web_edu.py).
    """

    class Config(BaseAgent.Config):
        script_path: str = "training/3b_fine_web_edu.py"
        model_name: str = "glasseye-3b"
        max_steps: int = 1000
        batch_size: int = 4
        use_ddp: bool = False
        num_gpus: int = 1

    def run(self, **kwargs) -> Result:
        cfg = self.validate(**kwargs)
        
        venv_python = Path.cwd() / "venv" / "bin" / "python"
        if not venv_python.exists():
            venv_python = "python3"

        # Determine the base command
        if cfg.use_ddp:
            cmd = [
                str(venv_python), "-m", "torch.distributed.run",
                f"--nproc_per_node={cfg.num_gpus}",
                cfg.script_path
            ]
        else:
            cmd = [str(venv_python), cfg.script_path]

        # In a real implementation, we would pass hyperparameters via env or CLI args
        # For the current 3b_fine_web_edu.py, hyperparams are hardcoded constants.
        # We could patch them or use environment variables if the script supports them.
        
        env = {
            **os.environ,
            "TRAIN_MAX_STEPS": str(cfg.max_steps),
            "TRAIN_BATCH_SIZE": str(cfg.batch_size),
            "PYTHONPATH": str(Path.cwd()),
        }

        logger.info(f"Starting training with command: {' '.join(cmd)}")
        log_file_path = Path.cwd() / "training.log"
        
        try:
            # Open the log file in append mode
            log_file = open(log_file_path, "a")
            
            # Start the process in the background
            process = subprocess.Popen(
                cmd,
                cwd=Path.cwd(),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Write PID to a file for tracking
            pid_file = Path.cwd() / "training.pid"
            with open(pid_file, "w") as f:
                f.write(str(process.pid))
                
            return Result(
                success=True, 
                metadata={
                    "status": "running_detached", 
                    "pid": process.pid,
                    "log_file": str(log_file_path)
                }
            )
            
        except Exception as e:
            logger.error(f"Training failed to start: {e}")
            return Result(success=False, metadata={"error": str(e)})
