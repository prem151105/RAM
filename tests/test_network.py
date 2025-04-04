"""
Tests for network functionality.
"""

import pytest
import numpy as np
from src.network.network import AgentNetwork
from src.agent import Agent

def test_network_initialization():
    """Test network initialization with default parameters."""
    network = AgentNetwork()
    
    assert network.num_agents == 5
    assert network.capability_dim == 10
    assert len(network.agents) == 5
    assert network.weight_matrix.shape == (5, 5)
    
def test_agent_capabilities():
    """Test agent capability initialization and normalization."""
    network = AgentNetwork(num_agents=3, capability_dim=5)
    
    for agent in network.agents.values():
        capabilities = agent.capabilities
        assert capabilities.shape == (5,)
        assert np.isclose(np.sum(capabilities), 1.0)
        assert np.all(capabilities >= 0)
        
def test_task_processing():
    """Test basic task processing through the network."""
    network = AgentNetwork(num_agents=3, capability_dim=5)
    
    task = {
        "observation": "How do I reset my password?",
        "context": {"user_id": "123"},
        "priority": 1.0
    }
    
    result = network.process_task(task)
    
    assert "status" in result
    assert "routing_path" in result
    assert len(result["routing_path"]) > 0
    
def test_weight_matrix_updates():
    """Test weight matrix updates during task processing."""
    network = AgentNetwork(num_agents=2, capability_dim=5)
    initial_weights = network.weight_matrix.copy()
    
    task = {
        "observation": "How do I reset my password?",
        "context": {"user_id": "123"},
        "priority": 1.0
    }
    
    network.process_task(task)
    
    # Check that weights have been updated
    assert not np.array_equal(network.weight_matrix, initial_weights)
    
def test_agent_selection():
    """Test agent selection based on capabilities."""
    network = AgentNetwork(num_agents=3, capability_dim=5)
    
    # Create task with specific capability requirements
    task = {
        "observation": "How do I reset my password?",
        "capabilities": np.array([0.8, 0.2, 0.0, 0.0, 0.0])
    }
    
    initial_agent = network._select_initial_agent(task)
    assert initial_agent is not None
    
    # Test next agent selection
    next_agent = network._select_next_agent(
        task,
        initial_agent,
        visited_agents={initial_agent}
    )
    assert next_agent is not None
    assert next_agent != initial_agent
    
def test_network_state():
    """Test network state retrieval."""
    network = AgentNetwork(num_agents=3, capability_dim=5)
    state = network._get_network_state()
    
    assert len(state) == 3
    for agent_id, info in state.items():
        assert "capabilities" in info
        assert "performance" in info
        assert info["capabilities"].shape == (5,)

class TestNetwork:
    """Tests network functionality and weight updates."""
    pass 