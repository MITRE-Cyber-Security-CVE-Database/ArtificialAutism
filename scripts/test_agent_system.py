import sys
import os
from pathlib import Path

# Add current directory to sys.path
sys.path.append(os.getcwd())

from mythos_glasseye.registry import AgentRegistry

def main():
    print(f"Working directory: {os.getcwd()}")
    registry = AgentRegistry()
    agents = registry.list_agents()
    print(f"Discovered agents: {list(agents.keys())}")
    
    expected_agents = ['model_builder', 'data_preparer', 'trainer', 'deployer', 'monitor', 'security_auditor']
    for agent_name in expected_agents:
        if agent_name in agents:
            print(f"Success: {agent_name} agent discovered.")
            agent = registry.get(agent_name)
            print(f"Instantiated: {type(agent).__name__}")
        else:
            print(f"Error: {agent_name} agent NOT found.")
            sys.exit(1)


if __name__ == "__main__":
    main()
