import importlib
import pkgutil
from typing import Dict, Type
import logging

from .agents.base import BaseAgent

logger = logging.getLogger(__name__)

class AgentRegistry:
    """Discover and provide access to agent classes."""

    def __init__(self, package: str = "mythos_glasseye.agents"):
        self.package = package
        self._registry: Dict[str, Type[BaseAgent]] = {}
        self._discover_agents()

    def _discover_agents(self) -> None:
        try:
            pkg = importlib.import_module(self.package)
        except ImportError as e:
            logger.error(f"Could not import agent package {self.package}: {e}")
            return

        for _, module_name, is_pkg in pkgutil.iter_modules(pkg.__path__):
            if is_pkg:
                continue
            try:
                module = importlib.import_module(f"{self.package}.{module_name}")
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and issubclass(attr, BaseAgent) and attr is not BaseAgent:
                        self._registry[module_name] = attr
                        logger.info(f"Registered agent: {module_name}")
                        break
            except Exception as e:
                logger.warning(f"Failed to load agent from {module_name}: {e}")

    def get(self, name: str) -> BaseAgent:
        agent_cls = self._registry.get(name)
        if agent_cls is None:
            raise KeyError(f"Agent '{name}' not found in registry.")
        return agent_cls()

    def list_agents(self) -> Dict[str, Type[BaseAgent]]:
        return self._registry.copy()
