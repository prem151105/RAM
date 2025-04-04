"""
Core agent class for the AgentNet Customer Support System.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import json
import time
from datetime import datetime
from loguru import logger

from .memory import Memory, EpisodicMemory, SemanticMemory
from .router import Router, WeightedRouter, HierarchicalRouter
from .executor import Executor, SpecializedExecutor

class Agent:
    """Advanced agent class integrating router, executor, and memory components."""
    def __init__(
        self,
        agent_id: str,
        initial_capabilities: np.ndarray,
        specialization: str = "general",
        memory_size: int = 1000,
        llm_model: str = "ollama/llama2",
        evolution_rate: float = 0.05,
        confidence_threshold: float = 0.7
    ):
        """Initialize an agent with router and executor components.
        
        Args:
            agent_id: Unique identifier for the agent
            initial_capabilities: Initial capability vector
            specialization: Agent's area of specialization
            memory_size: Maximum size of memory modules
            llm_model: Name of the LLM model to use
            evolution_rate: Rate at which capabilities evolve
            confidence_threshold: Threshold for task execution confidence
        """
        self.agent_id = agent_id
        self.capabilities = initial_capabilities
        self.specialization = specialization
        self.llm_model = llm_model
        self.evolution_rate = evolution_rate
        self.confidence_threshold = confidence_threshold
        self.creation_timestamp = datetime.now().isoformat()
        self.tasks_processed = 0
        
        # Initialize episodic and semantic memories
        self.episodic_memory = EpisodicMemory(
            max_size=memory_size,
            embedding_dim=len(initial_capabilities)
        )
        
        self.semantic_memory = SemanticMemory(
            max_size=memory_size * 2,
            embedding_dim=len(initial_capabilities)
        )
        
        # Initialize router with different strategies based on specialization
        if specialization.lower() in ["routing", "classification"]:
            # Use hierarchical router for routing specialists
            self.router = HierarchicalRouter(
                memory=self.semantic_memory,
                llm_model=llm_model,
                capability_dim=len(initial_capabilities)
            )
        else:
            # Use weighted router for other specialists
            self.router = WeightedRouter(
                memory=self.semantic_memory,
                llm_model=llm_model,
                capability_dim=len(initial_capabilities)
            )
        
        # Initialize executor with specialization
        self.executor = SpecializedExecutor(
            specialization=specialization,
            memory=self.episodic_memory,
            llm_model=llm_model,
            capability_dim=len(initial_capabilities)
        )
        
        # Performance metrics
        self.performance_metrics = {
            "successful_tasks": 0,
            "failed_tasks": 0,
            "avg_confidence": 0.0,
            "avg_processing_time": 0.0,
            "routing_accuracy": 0.0
        }
        
    def process_task(
        self,
        task: Dict,
        network_state: Dict,
        beta: float = 0.8
    ) -> Tuple[str, Dict]:
        """Process an incoming task using the ReAct framework.
        
        Args:
            task: Task dictionary containing observation, context and priority
            network_state: Current state of the agent network
            beta: Decay factor for capability updates
            
        Returns:
            action_type: Type of action taken (forward/split/execute)
            result: Result of the action
        """
        # Track task processing time
        start_time = time.time()
        
        # Increment tasks processed counter
        self.tasks_processed += 1
        
        # Get task components
        observation = task["observation"]
        context = task.get("context", {})
        priority = task.get("priority", 1.0)
        
        # Add task to episodic memory for future reference
        task_embedding = self._generate_embedding(observation)
        self.episodic_memory.add(
            key=f"task_{int(time.time())}",
            value={
                "observation": observation,
                "context": context,
                "priority": priority,
                "timestamp": datetime.now().isoformat()
            },
            embedding=task_embedding
        )
        
        # Router reasoning and action selection
        routing_decision = self.router.reason_and_act(
            observation=observation,
            context=context,
            network_state=network_state,
            agent_capabilities=self.capabilities
        )
        
        action_type = routing_decision["action_type"]
        logger.debug(f"Agent {self.agent_id} decided action: {action_type}")
        
        if action_type == "execute":
            # Check confidence before execution
            confidence = routing_decision.get("confidence", 0.0)
            
            if confidence >= self.confidence_threshold:
                # Execute task using executor
                result = self.executor.execute_task(
                    observation=observation,
                    context=context
                )
                
                # Store result in semantic memory
                result_embedding = self._generate_embedding(json.dumps(result))
                self.semantic_memory.add(
                    key=f"result_{int(time.time())}",
                    value={
                        "task": observation,
                        "result": result,
                        "timestamp": datetime.now().isoformat()
                    },
                    embedding=result_embedding
                )
                
                # Update capabilities based on execution
                self._update_capabilities(task, result, beta)
                
                # Update performance metrics
                if result.get("success", False):
                    self.performance_metrics["successful_tasks"] += 1
                else:
                    self.performance_metrics["failed_tasks"] += 1
                    
                # Update average confidence
                total_tasks = self.performance_metrics["successful_tasks"] + self.performance_metrics["failed_tasks"]
                self.performance_metrics["avg_confidence"] = (
                    (self.performance_metrics["avg_confidence"] * (total_tasks - 1) + confidence) / total_tasks
                )
                
                return action_type, result
            else:
                # Confidence too low, forward to better agent
                logger.debug(f"Agent {self.agent_id} confidence {confidence} below threshold {self.confidence_threshold}")
                # Find most suitable agent from network state
                suitable_agents = self._find_suitable_agents(network_state, task)
                if suitable_agents:
                    action_type = "forward"
                    return action_type, suitable_agents[0]
                else:
                    # No better agent found, execute anyway with warning
                    logger.warning(f"No better agent found, agent {self.agent_id} executing with low confidence {confidence}")
                    result = self.executor.execute_task(
                        observation=observation,
                        context=context
                    )
                    return "execute", result
            
        elif action_type == "split":
            # Split task into subtasks
            subtasks = routing_decision["subtasks"]
            results = {}
            
            # Execute subtasks that match current capabilities
            for subtask_id, subtask in subtasks.items():
                if self._can_handle_task(subtask):
                    result = self.executor.execute_task(
                        observation=subtask["observation"],
                        context=subtask.get("context", {})
                    )
                    results[subtask_id] = result
                    
            return action_type, {
                "completed_subtasks": results,
                "remaining_subtasks": {
                    k: v for k, v in subtasks.items() 
                    if k not in results
                }
            }
            
        else:  # forward
            # Update performance metrics for routing
            self.performance_metrics["routing_accuracy"] = (
                self.performance_metrics["routing_accuracy"] * 0.9 + 
                routing_decision.get("routing_confidence", 0.8) * 0.1
            )
            
            return action_type, routing_decision["target_agent"]
        
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding vector for text."""
        # Simple hashing-based embedding for simulation
        # In a real implementation, this would use an actual embedding model
        hash_val = hash(text)
        rng = np.random.RandomState(hash_val)
        embedding = rng.random(len(self.capabilities))
        return embedding / np.linalg.norm(embedding)
        
    def _can_handle_task(self, task: Dict) -> bool:
        """Check if agent can handle a task based on capabilities."""
        task_requirements = task.get("capabilities", None)
        if task_requirements is None:
            return True
            
        capability_match = np.dot(self.capabilities, task_requirements)
        
        # Also check past similar tasks in memory
        task_text = task.get("observation", "")
        task_embedding = self._generate_embedding(task_text)
        
        similar_tasks = self.episodic_memory.search(
            embedding=task_embedding,
            top_k=5
        )
        
        # If we have similar successful tasks in memory, increase confidence
        memory_confidence = 0.0
        if similar_tasks:
            # Calculate average similarity of top matches
            similarity_sum = sum(s for _, s in similar_tasks)
            memory_confidence = similarity_sum / len(similar_tasks) * 0.5
            
        # Combined confidence from capabilities and memory
        total_confidence = capability_match * 0.7 + memory_confidence * 0.3
        
        return total_confidence > self.confidence_threshold
        
    def _update_capabilities(
        self,
        task: Dict,
        result: Dict,
        beta: float
    ) -> None:
        """Update agent capabilities based on task execution.
        
        Args:
            task: Original task
            result: Execution result
            beta: Decay factor
        """
        # Extract capability delta from task execution
        delta_capabilities = self._compute_capability_delta(task, result)
        
        # Update capabilities using decay factor
        self.capabilities = (
            beta * self.capabilities + 
            (1 - beta) * delta_capabilities
        )
        
        # Normalize capabilities to sum to 1
        if np.sum(self.capabilities) > 0:
            self.capabilities = self.capabilities / np.sum(self.capabilities)
        
        # Store updated capabilities in semantic memory
        self.semantic_memory.add(
            key=f"capabilities_{int(time.time())}",
            value={
                "capabilities": self.capabilities.tolist(),
                "timestamp": datetime.now().isoformat()
            },
            embedding=self.capabilities
        )
        
    def _compute_capability_delta(
        self,
        task: Dict,
        result: Dict
    ) -> np.ndarray:
        """Compute capability changes from task execution."""
        # Extract success information
        success = result.get("success", False)
        confidence = result.get("confidence", 0.5)
        
        # Get task requirements or use current capabilities as baseline
        task_requirements = task.get("capabilities", self.capabilities)
        
        if success:
            # Successful execution strengthens capabilities in used areas
            if confidence > 0.8:
                # High confidence success strengthens capabilities more
                return task_requirements * (1.0 + self.evolution_rate)
            else:
                # Normal success with moderate strengthening
                return task_requirements * (1.0 + self.evolution_rate * 0.5)
        else:
            # Failed execution leads to smaller adjustment
            return task_requirements * (1.0 - self.evolution_rate * 0.25)
            
    def _find_suitable_agents(self, network_state: Dict, task: Dict) -> List[str]:
        """Find suitable agents in the network for a given task."""
        suitable_agents = []
        
        # Get task requirements
        task_text = task.get("observation", "")
        task_embedding = self._generate_embedding(task_text)
        
        # Score each agent in the network
        agent_scores = []
        for agent_id, agent_info in network_state.items():
            # Skip self
            if agent_id == self.agent_id:
                continue
                
            # Score based on capabilities match
            capabilities = np.array(agent_info.get("capabilities", [0] * len(self.capabilities)))
            capability_score = np.dot(capabilities, task_embedding)
            
            # Consider agent performance
            performance = agent_info.get("performance", 0.5)
            
            # Combined score
            score = capability_score * 0.7 + performance * 0.3
            
            agent_scores.append((agent_id, score))
            
        # Sort by score descending
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top agent IDs
        return [agent_id for agent_id, _ in agent_scores[:3]]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get agent performance statistics."""
        stats = self.performance_metrics.copy()
        
        # Add derived metrics
        total_tasks = stats["successful_tasks"] + stats["failed_tasks"]
        if total_tasks > 0:
            stats["success_rate"] = stats["successful_tasks"] / total_tasks
        else:
            stats["success_rate"] = 0.0
            
        stats["total_tasks"] = total_tasks
        stats["specialization"] = self.specialization
        stats["capabilities"] = self.capabilities.tolist()
        stats["memory_usage"] = {
            "episodic": self.episodic_memory.get_usage(),
            "semantic": self.semantic_memory.get_usage()
        }
        
        return stats 