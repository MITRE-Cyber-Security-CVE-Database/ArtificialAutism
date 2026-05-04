import sys
import os
import logging
from pathlib import Path

# Add current directory to sys.path
sys.path.append(os.getcwd())

from mythos_glasseye.registry import AgentRegistry
from mythos_glasseye.pipeline_runner import PipelineRunner
from mythos_glasseye.loops import LoopOrchestrator

# Configure logging to see the loop activity
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)

def main():
    registry = AgentRegistry()
    runner = PipelineRunner(registry)
    orchestrator = LoopOrchestrator(runner)
    
    # Define the "needs" to be served by loops
    loops_to_run = [
        {
            "name": "Maintenance",
            "yaml_path": "pipelines/maintenance_loop.yaml",
            "interval": 10 # Short for demo
        }
    ]
    
    # If optimization loop requested via CLI
    if "--optimize" in sys.argv:
        loops_to_run.append({
            "name": "Optimization",
            "yaml_path": "pipelines/optimization_loop.yaml",
            "interval": 30
        })
    
    print("Mythos Glasseye Loop System Starting...")
    print(f"Active loops: {[l['name'] for l in loops_to_run]}")
    
    try:
        # For this MVP, we run the first one in the list as a master loop
        # In multi-loop mode, we cycle through them
        orchestrator.start_loop(
            name=loops_to_run[0]["name"],
            yaml_path=loops_to_run[0]["yaml_path"],
            interval=loops_to_run[0]["interval"]
        )
    except KeyboardInterrupt:
        print("\nShutdown complete.")

if __name__ == "__main__":
    main()
