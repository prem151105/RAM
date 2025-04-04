"""
Router component for the AgentNet Customer Support System.
"""

from typing import Dict, List, Optional
import numpy as np
import json
from loguru import logger

from .memory import Memory
from ..utils.llm_interface import OllamaInterface

class Router:
    """Router component responsible for task routing decisions."""
    
    def __init__(
        self,
        memory_size: int = 1000,
        llm_model: str = "ollama/llama2"
    ):
        """Initialize router with memory module.
        
        Args:
            memory_size: Maximum size of memory module
            llm_model: Name of the LLM model to use
        """
        self.memory = Memory(max_size=memory_size)
        
        # Initialize LLM interface
        model_name = llm_model
        if "/" in llm_model:
            model_name = llm_model.split("/")[-1]
            
        self.llm = OllamaInterface(model_name=model_name)
        
    def reason_and_act(
        self,
        observation: str,
        context: Dict,
        network_state: Dict,
        agent_capabilities: np.ndarray
    ) -> Dict:
        """Implement the ReAct framework for routing decisions.
        
        Args:
            observation: Task description/query
            context: Current context of the task
            network_state: State of the agent network
            agent_capabilities: Current agent's capability vector
            
        Returns:
            Dict containing routing decision and related information
        """
        # Retrieve relevant experiences from memory
        relevant_fragments = self.memory.retrieve(
            query=observation,
            context=context,
            k=5
        )
        
        # Generate routing prompt
        prompt = self._generate_routing_prompt(
            observation=observation,
            context=context,
            fragments=relevant_fragments,
            network_state=network_state,
            agent_capabilities=agent_capabilities
        )
        
        # Get routing decision from LLM
        routing_decision = self._get_llm_routing_decision(prompt)
        
        # Store the routing experience
        self.memory.store({
            "observation": observation,
            "context": context,
            "decision": routing_decision
        })
        
        return routing_decision
        
    def _generate_routing_prompt(
        self,
        observation: str,
        context: Dict,
        fragments: List[Dict],
        network_state: Dict,
        agent_capabilities: np.ndarray
    ) -> str:
        """Generate prompt for routing decision.
        
        Args:
            observation: Task description
            context: Task context
            fragments: Retrieved memory fragments
            network_state: Network state
            agent_capabilities: Agent capabilities
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""Given the following customer support task:
Task: {observation}

Current context:
{self._format_context(context)}

Your capabilities:
{self._format_capabilities(agent_capabilities)}

Available agents in network:
{self._format_network_state(network_state)}

Similar past experiences:
{self._format_fragments(fragments)}

Determine the best routing action:
1. Execute the task if it matches your capabilities
2. Split the task if it can be broken down into simpler subtasks
3. Forward the task to a more suitable agent

Provide your decision in the following format:
{{
    "action_type": "execute|split|forward",
    "reasoning": "Your step-by-step reasoning",
    "subtasks": {{}} (if splitting),
    "target_agent": "" (if forwarding)
}}
"""
        return prompt
        
    def _get_llm_routing_decision(self, prompt: str) -> Dict:
        """Get routing decision from LLM model.
        
        Args:
            prompt: Formatted prompt string
            
        Returns:
            Parsed routing decision
        """
        system_prompt = """You are a router agent in a customer support system. 
Your task is to analyze customer queries and decide whether to:
1. Execute the task directly
2. Split the task into subtasks
3. Forward the task to another agent

Always respond with a valid JSON object."""
        
        try:
            # Get LLM response as JSON
            result = self.llm.generate_json(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3  # Lower temperature for more deterministic routing
            )
            
            # Validate response format
            if not result or "action_type" not in result:
                logger.warning("Invalid routing decision format. Using default.")
                return self._get_default_decision()
                
            return result
            
        except Exception as e:
            logger.error(f"Error getting routing decision: {str(e)}")
            return self._get_default_decision()
            
    def _get_default_decision(self) -> Dict:
        """Get default routing decision."""
        return {
            "action_type": "execute",
            "reasoning": "Fallback decision due to error in LLM response",
            "subtasks": {},
            "target_agent": None
        }
        
    def _format_context(self, context: Dict) -> str:
        """Format task context for prompt."""
        return "\n".join([
            f"- {key}: {value}"
            for key, value in context.items()
        ])
        
    def _format_capabilities(self, capabilities: np.ndarray) -> str:
        """Format agent capabilities for prompt."""
        return f"Capability vector: {capabilities.tolist()}"
        
    def _format_network_state(self, network_state: Dict) -> str:
        """Format network state for prompt."""
        return "\n".join([
            f"Agent {agent_id}:"
            f"\n  Capabilities: {info['capabilities'].tolist()}"
            f"\n  Performance: {info.get('performance', 'N/A')}"
            for agent_id, info in network_state.items()
        ])
        
    def _format_fragments(self, fragments: List[Dict]) -> str:
        """Format memory fragments for prompt."""
        if not fragments:
            return "No similar past experiences found."
            
        return "\n\n".join([
            f"Fragment {i+1}:"
            f"\nTask: {f['observation']}"
            f"\nDecision: {f.get('decision', {})}"
            for i, f in enumerate(fragments)
        ]) 