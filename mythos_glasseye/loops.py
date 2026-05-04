import time
import logging
import signal
from typing import Dict, Any, List

from .pipeline_runner import PipelineRunner
from .registry import AgentRegistry

logger = logging.getLogger(__name__)

class LoopOrchestrator:
    """Manages continuous execution of pipelines (loops)."""

    def __init__(self, runner: PipelineRunner):
        self.runner = runner
        self.active_loops = {}
        self.should_stop = False
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_exit)
        signal.signal(signal.SIGTERM, self._handle_exit)

    def _handle_exit(self, signum, frame):
        logger.info("Shutdown signal received. Stopping loops...")
        self.should_stop = True

    def start_loop(self, name: str, yaml_path: str, interval: int = 60, iterations: int = -1):
        """Starts a pipeline in a continuous loop.
        
        Args:
            name: Human-readable name for the loop.
            yaml_path: Path to the pipeline YAML.
            interval: Seconds to wait between iterations.
            iterations: Total iterations (-1 for infinite).
        """
        logger.info(f"Starting loop '{name}' with interval {interval}s")
        count = 0
        
        while not self.should_stop:
            logger.info(f"--- Loop '{name}' iteration {count + 1} ---")
            results = self.runner.run_pipeline(yaml_path)
            
            # Log summary of results
            success = all(r.get("success", False) for r in results)
            if success:
                logger.info(f"Loop '{name}' iteration {count + 1} completed successfully.")
            else:
                logger.warning(f"Loop '{name}' iteration {count + 1} had failures.")
            
            count += 1
            if iterations > 0 and count >= iterations:
                logger.info(f"Loop '{name}' reached iteration limit ({iterations}).")
                break
                
            if self.should_stop:
                break
                
            logger.info(f"Waiting {interval}s for next iteration...")
            time.sleep(interval)

    def run_multi_loop(self, loops_cfg: List[Dict[str, Any]]):
        """Runs multiple loops (sequential execution in a single thread for MVP)."""
        # Note: In a production version, these would run in separate threads/processes.
        # For now, we cycle through them.
        while not self.should_stop:
            for loop in loops_cfg:
                self.start_loop(
                    name=loop["name"],
                    yaml_path=loop["yaml_path"],
                    interval=0, # We handle the interval globally or per iteration
                    iterations=1
                )
                if self.should_stop:
                    break
            
            if self.should_stop:
                break
            
            # Global wait between full cycles if desired
            time.sleep(1)
