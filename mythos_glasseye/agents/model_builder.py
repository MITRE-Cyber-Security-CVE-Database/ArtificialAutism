import os
from pathlib import Path
from typing import Dict, Any
import subprocess

from .base import BaseAgent, Result

class ModelBuilderAgent(BaseAgent):
    """Agent that builds an OpenMythos checkpoint."""

    class Config(BaseAgent.Config):
        model_name: str = "glasseye-tiny"
        config: Dict[str, Any] = {}
        device: str = "cpu"

    def run(self, **kwargs) -> Result:
        cfg = self.validate(**kwargs)
        # Use absolute path to the venv python if possible, or assume it's in the current dir
        venv_python = Path.cwd() / "venv" / "bin" / "python"
        if not venv_python.exists():
            venv_python = "python3" # Fallback
            
        cmd = [str(venv_python), "scripts/build_openmythos_model.py"]
        env = {**os.environ, "OPENMYTHOS_DEVICE": cfg.device, "PYTHONPATH": str(Path.cwd())}
        
        try:
            import json
            result = subprocess.run(cmd, cwd=Path.cwd(), env=env, capture_output=True, text=True, check=True)
            artifact_path = Path("artifacts/openmythos_0ai_cpu_dev.pt")
            
            metadata = {"model_name": cfg.model_name}
            # Attempt to parse the last few lines for JSON output from the script
            for line in reversed(result.stdout.splitlines()[-10:]):
                if line.strip().startswith("{"):
                    try:
                        parsed_json = json.loads(line)
                        metadata.update(parsed_json)
                        break
                    except json.JSONDecodeError:
                        pass
                        
            return Result(success=True, output_path=artifact_path, metadata=metadata)
        except subprocess.CalledProcessError as e:
            return Result(success=False, metadata={"error": str(e), "stderr": e.stderr})
