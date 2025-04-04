"""
Network management for the AgentNet Customer Support System.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from loguru import logger

from ..agent import Agent

class AgentNetwork:
    """Main class managing the multi-agent network and task routing."""
    
    def __init__(
        self,
        num_agents: int = 5,
        capability_dim: int = 10,
        memory_size: int = 1000,
        llm_model: str = "ollama/llama2"
    ):
        """Initialize the agent network.
        
        Args:
            num_agents: Number of agents in the network
            capability_dim: Dimension of capability vectors
            memory_size: Memory size for each agent
            llm_model: Name of the LLM model to use
        """
        self.num_agents = num_agents
        self.capability_dim = capability_dim
        self.llm_model = llm_model
        
        # Initialize agents with random capabilities
        self.agents = {}
        self.weight_matrix = np.zeros((num_agents, num_agents))
        
        for i in range(num_agents):
            # Initialize random capabilities
            capabilities = np.random.random(capability_dim)
            capabilities = capabilities / np.sum(capabilities)  # Normalize
            
            # Create agent
            self.agents[f"agent_{i}"] = Agent(
                agent_id=f"agent_{i}",
                initial_capabilities=capabilities,
                memory_size=memory_size,
                llm_model=llm_model
            )
            
    def process_task(
        self,
        task: Dict,
        alpha: float = 0.8,
        max_hops: int = 5
    ) -> Dict:
        """Process a customer support task through the agent network.
        
        Args:
            task: Task dictionary with observation and context
            alpha: Weight matrix decay factor
            max_hops: Maximum number of agent hops allowed
            
        Returns:
            Dict containing task result and routing path
        """
        # Select initial agent based on task requirements
        current_agent_id = self._select_initial_agent(task)
        visited_agents = set()
        routing_path = []
        
        for hop in range(max_hops):
            if current_agent_id in visited_agents:
                logger.warning(f"Cycle detected in routing path at hop {hop}")
                break
                
            visited_agents.add(current_agent_id)
            routing_path.append(current_agent_id)
            
            # Get current agent and network state
            current_agent = self.agents[current_agent_id]
            network_state = self._get_network_state()
            
            # Process task with current agent
            action_type, result = current_agent.process_task(
                task=task,
                network_state=network_state,
                beta=alpha  # Use same decay factor for capabilities
            )
            
            if action_type == "execute":
                # Task completed
                return {
                    "status": "completed",
                    "result": result,
                    "routing_path": routing_path
                }
                
            elif action_type == "split":
                # Handle subtasks
                completed_subtasks = result["completed_subtasks"]
                remaining_subtasks = result["remaining_subtasks"]
                
                if not remaining_subtasks:
                    # All subtasks completed
                    return {
                        "status": "completed",
                        "result": completed_subtasks,
                        "routing_path": routing_path
                    }
                    
                # Continue with first remaining subtask
                subtask_id = next(iter(remaining_subtasks))
                task = remaining_subtasks[subtask_id]
                current_agent_id = self._select_next_agent(
                    task,
                    current_agent_id,
                    visited_agents
                )
                
            else:  # forward
                # Update task context with forwarding info
                task["context"] = task.get("context", {})
                task["context"]["forwarded_from"] = current_agent_id
                
                # Move to next agent
                current_agent_id = result  # result contains target_agent_id
                
            # Update weight matrix
            self._update_weights(routing_path, alpha)
            
        # Max hops reached
        return {
            "status": "max_hops_reached",
            "routing_path": routing_path,
            "last_agent": current_agent_id
        }
        
    def _select_initial_agent(self, task: Dict) -> str:
        """Select initial agent based on task requirements.
        
        Args:
            task: Task dictionary
            
        Returns:
            ID of selected agent
        """
        # Extract task requirements
        if "capabilities" in task:
            requirements = task["capabilities"]
        else:
            # Use uniform requirements if not specified
            requirements = np.ones(self.capability_dim) / self.capability_dim
            
        # Find best matching agent
        best_match = None
        best_score = -1
        
        for agent_id, agent in self.agents.items():
            score = np.dot(agent.capabilities, requirements)
            if score > best_score:
                best_score = score
                best_match = agent_id
                
        return best_match
        
    def _select_next_agent(
        self,
        task: Dict,
        current_agent_id: str,
        visited_agents: set
    ) -> str:
        """Select next agent for task routing.
        
        Args:
            task: Task dictionary
            current_agent_id: Current agent's ID
            visited_agents: Set of already visited agents
            
        Returns:
            ID of selected agent
        """
        # Get available agents (not visited)
        available_agents = set(self.agents.keys()) - visited_agents
        
        if not available_agents:
            # If all agents visited, allow revisiting least recently visited
            return min(
                visited_agents,
                key=lambda x: list(visited_agents).index(x)
            )
            
        # Use weight matrix and capabilities to select next agent
        current_idx = int(current_agent_id.split("_")[1])
        weights = self.weight_matrix[current_idx]
        
        best_agent = None
        best_score = -1
        
        for agent_id in available_agents:
            agent_idx = int(agent_id.split("_")[1])
            # Combine weight and capability match
            weight_score = weights[agent_idx]
            capability_score = np.dot(
                self.agents[agent_id].capabilities,
                task.get("capabilities", np.ones(self.capability_dim))
            )
            score = 0.5 * weight_score + 0.5 * capability_score
            
            if score > best_score:
                best_score = score
                best_agent = agent_id
                
        return best_agent
        
    def _update_weights(
        self,
        routing_path: List[str],
        alpha: float
    ) -> None:
        """Update weight matrix based on routing path.
        
        Args:
            routing_path: List of agent IDs in routing order
            alpha: Decay factor
        """
        for i in range(len(routing_path) - 1):
            from_idx = int(routing_path[i].split("_")[1])
            to_idx = int(routing_path[i + 1].split("_")[1])
            
            # Update weight using decay factor
            self.weight_matrix[from_idx, to_idx] = (
                alpha * self.weight_matrix[from_idx, to_idx] +
                (1 - alpha) * 1.0  # Successful routing weight
            )
            
    def _get_network_state(self) -> Dict:
        """Get current state of all agents in the network.
        
        Returns:
            Dict containing agent states
        """
        return {
            agent_id: {
                "capabilities": agent.capabilities,
                "performance": np.mean(self.weight_matrix[
                    int(agent_id.split("_")[1])
                ])
            }
            for agent_id, agent in self.agents.items()
        } 