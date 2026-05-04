import sys
import os
import json
from pathlib import Path

# Add current directory to sys.path
sys.path.append(os.getcwd())

from mythos_glasseye.registry import AgentRegistry
from mythos_glasseye.pipeline_runner import PipelineRunner

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_pipeline.py <pipeline_yaml>")
        sys.exit(1)
        
    yaml_path = sys.argv[1]
    registry = AgentRegistry()
    runner = PipelineRunner(registry)
    
    print(f"Running pipeline: {yaml_path}")
    results = runner.run_pipeline(yaml_path)
    
    print("\nPipeline Results:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
