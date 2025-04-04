"""
Executor component for the AgentNet Customer Support System.
"""

from typing import Dict, List, Optional
import json
from loguru import logger

from .memory import Memory
from ..utils.llm_interface import OllamaInterface

class Executor:
    """Executor component responsible for task execution."""
    
    def __init__(
        self,
        memory_size: int = 1000,
        llm_model: str = "ollama/llama2"
    ):
        """Initialize executor with memory module.
        
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
        
    def execute_task(
        self,
        observation: str,
        context: Dict
    ) -> Dict:
        """Execute a customer support task using the ReAct framework.
        
        Args:
            observation: Task description/query
            context: Current context of the task
            
        Returns:
            Dict containing execution result
        """
        # Retrieve relevant experiences
        relevant_fragments = self.memory.retrieve(
            query=observation,
            context=context,
            k=5
        )
        
        # Generate execution prompt
        prompt = self._generate_execution_prompt(
            observation=observation,
            context=context,
            fragments=relevant_fragments
        )
        
        # Get execution plan and result from LLM
        result = self._get_llm_execution_result(prompt)
        
        # Store the execution experience
        self.memory.store({
            "observation": observation,
            "context": context,
            "result": result
        })
        
        return result
        
    def _generate_execution_prompt(
        self,
        observation: str,
        context: Dict,
        fragments: List[Dict]
    ) -> str:
        """Generate prompt for task execution.
        
        Args:
            observation: Task description
            context: Task context
            fragments: Retrieved memory fragments
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""Given the following customer support task:
Task: {observation}

Current context:
{self._format_context(context)}

Similar past experiences:
{self._format_fragments(fragments)}

Follow these steps to execute the task:
1. Analyze the customer query and identify the key issues
2. Review similar past experiences for relevant solutions
3. Generate a clear and helpful response
4. Include any necessary follow-up actions or escalation paths

Provide your response in the following format:
{{
    "analysis": "Your analysis of the customer query",
    "solution": "Your proposed solution",
    "response": "Customer-facing response text",
    "follow_up_actions": ["List of follow-up actions if any"],
    "success": true/false,
    "confidence": 0.0-1.0
}}
"""
        return prompt
        
    def _get_llm_execution_result(self, prompt: str) -> Dict:
        """Get execution result from LLM model.
        
        Args:
            prompt: Formatted prompt string
            
        Returns:
            Parsed execution result
        """
        system_prompt = """You are a customer support agent. 
Your task is to analyze customer queries, provide clear solutions, and generate helpful responses.
Always structure your response as a valid JSON object with the requested fields."""
        
        try:
            # Get LLM response as JSON
            result = self.llm.generate_json(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.7  # Higher temperature for more creative responses
            )
            
            # Validate response format
            if not result or "response" not in result:
                logger.warning("Invalid execution result format. Using default.")
                return self._get_default_result(observation=prompt)
                
            # Ensure all required fields are present
            required_fields = [
                "analysis", "solution", "response", 
                "follow_up_actions", "success", "confidence"
            ]
            
            for field in required_fields:
                if field not in result:
                    if field == "follow_up_actions":
                        result[field] = []
                    elif field == "success":
                        result[field] = True
                    elif field == "confidence":
                        result[field] = 0.8
                    else:
                        result[field] = "Not provided"
                        
            return result
            
        except Exception as e:
            logger.error(f"Error getting execution result: {str(e)}")
            return self._get_default_result(observation=prompt)
            
    def _get_default_result(self, observation: str) -> Dict:
        """Get default execution result."""
        return {
            "analysis": f"Analysis for: {observation[:50]}...",
            "solution": "Standard troubleshooting steps",
            "response": "Thank you for contacting support. We're looking into your issue and will get back to you shortly.",
            "follow_up_actions": [],
            "success": True,
            "confidence": 0.7
        }
        
    def _format_context(self, context: Dict) -> str:
        """Format task context for prompt."""
        if not context:
            return "No additional context provided."
            
        return "\n".join([
            f"- {key}: {value}"
            for key, value in context.items()
        ])
        
    def _format_fragments(self, fragments: List[Dict]) -> str:
        """Format memory fragments for prompt."""
        if not fragments:
            return "No similar past experiences found."
            
        return "\n\n".join([
            f"Fragment {i+1}:"
            f"\nTask: {f['observation']}"
            f"\nResult: {json.dumps(f.get('result', {}), indent=2)}"
            for i, f in enumerate(fragments)
        ]) 