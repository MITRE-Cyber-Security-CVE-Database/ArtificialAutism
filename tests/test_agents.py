"""
Tests for Mythos Glasseye agents
"""

import pytest
from pathlib import Path

from mythos_glasseye.registry import AgentRegistry
from mythos_glasseye.agents.base import BaseAgent, Result


class TestAgentRegistry:
    """Test agent registry functionality"""
    
    def test_registry_initialization(self):
        """Test that registry initializes and discovers agents"""
        registry = AgentRegistry()
        agents = registry.list_agents()
        
        assert len(agents) > 0, "Should discover at least one agent"
        assert "model_builder" in agents, "Should find model_builder agent"
    
    def test_get_agent(self):
        """Test getting an agent from registry"""
        registry = AgentRegistry()
        agent = registry.get("model_builder")
        
        assert agent is not None, "Should return agent instance"
        assert isinstance(agent, BaseAgent), "Should be BaseAgent subclass"
    
    def test_get_nonexistent_agent(self):
        """Test getting non-existent agent raises error"""
        registry = AgentRegistry()
        
        with pytest.raises(KeyError):
            registry.get("nonexistent_agent")


class TestBaseAgent:
    """Test base agent functionality"""
    
    def test_result_creation(self):
        """Test Result creation"""
        result = Result(
            success=True,
            output_path=Path("test.pt"),
            metadata={"key": "value"}
        )
        
        assert result.success is True
        assert result.output_path == Path("test.pt")
        assert result.metadata["key"] == "value"
    
    def test_result_failure(self):
        """Test Result with failure"""
        result = Result(success=False)
        assert result.success is False


class TestModelBuilderAgent:
    """Test model builder agent"""
    
    def test_model_builder_instantiation(self):
        """Test that model builder can be instantiated"""
        registry = AgentRegistry()
        builder = registry.get("model_builder")
        
        assert builder is not None
        assert hasattr(builder, "run")


class TestDataPreparerAgent:
    """Test data preparer agent"""
    
    def test_data_preparer_instantiation(self):
        """Test that data preparer can be instantiated"""
        registry = AgentRegistry()
        preparer = registry.get("data_preparer")
        
        assert preparer is not None
        assert hasattr(preparer, "run")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
