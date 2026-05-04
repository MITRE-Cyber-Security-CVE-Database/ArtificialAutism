import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List

from .registry import AgentRegistry

logger = logging.getLogger(__name__)

class PipelineRunner:
    """Executes a sequence of agent steps defined in a YAML file."""

    def __init__(self, registry: AgentRegistry):
        self.registry = registry

    def run_pipeline(self, yaml_path: str) -> List[Dict[str, Any]]:
        with open(yaml_path, "r") as f:
            pipeline_cfg = yaml.safe_load(f)
            
        steps = pipeline_cfg.get("steps", [])
        results = []
        context = {} # Shared context between steps
        
        for step in steps:
            name = step.get("name")
            agent_name = step.get("agent")
            args = step.get("args", {})
            
            # Resolve dynamic arguments from context (e.g., outputs from previous steps)
            # This is a simple implementation where {{step_name.field}} can be resolved.
            resolved_args = self._resolve_args(args, context)
            
            logger.info(f"Executing step '{name}' using agent '{agent_name}'")
            
            try:
                agent = self.registry.get(agent_name)
                result = agent.run(**resolved_args)
                
                results.append({
                    "step": name,
                    "success": result.success,
                    "output_path": str(result.output_path) if result.output_path else None,
                    "metadata": result.metadata
                })
                
                # Update context
                context[name] = {
                    "success": result.success,
                    "output_path": result.output_path,
                    **result.metadata
                }
                
                if not result.success:
                    logger.error(f"Step '{name}' failed. Aborting pipeline.")
                    break
                    
            except Exception as e:
                logger.error(f"Error in step '{name}': {e}")
                break
                
        return results

    def _resolve_args(self, args: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        # Simple resolution for now
        resolved = {}
        for k, v in args.items():
            if isinstance(v, str) and v.startswith("{{") and v.endswith("}}"):
                path = v[2:-2].strip()
                # Resolve path like "build_checkpoint.output_path"
                parts = path.split(".")
                val = context
                for part in parts:
                    if isinstance(val, dict):
                        val = val.get(part)
                    else:
                        val = getattr(val, part, None)
                resolved[k] = val
            else:
                resolved[k] = v
        return resolved
