"""
LLM interface for interacting with Ollama models.
"""

import json
import requests
from typing import Dict, List, Optional, Union
from loguru import logger

class OllamaInterface:
    """Interface for interacting with Ollama LLM models."""
    
    def __init__(
        self,
        model_name: str = "llama2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 2048
    ):
        """Initialize Ollama interface.
        
        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL for Ollama API
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.is_connected = self._check_connection()
        
    def _check_connection(self) -> bool:
        """Check if Ollama is accessible.
        
        Returns:
            True if Ollama is accessible, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Failed to connect to Ollama: {str(e)}")
            return False
        
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False
    ) -> str:
        """Generate text using Ollama model.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (overrides default)
            max_tokens: Maximum tokens (overrides default)
            json_mode: Whether to request JSON output
            
        Returns:
            Generated text or fallback if Ollama is not available
        """
        if not self.is_connected:
            return self._fallback_response(prompt, json_mode)
            
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "format": "json" if json_mode else None
        }
        
        if system_prompt:
            payload["system"] = system_prompt
            
        # Filter out None values
        payload = {k: v for k, v in payload.items() if v is not None}
        
        try:
            logger.debug(f"Sending request to Ollama API: {payload}")
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            logger.error(f"Error calling Ollama API: {str(e)}")
            return self._fallback_response(prompt, json_mode)
            
    def _fallback_response(self, prompt: str, json_mode: bool) -> str:
        """Generate a fallback response when Ollama is not available.
        
        Args:
            prompt: User prompt
            json_mode: Whether to generate JSON response
            
        Returns:
            Fallback response
        """
        logger.warning("Using fallback response generator")
        
        if json_mode:
            return json.dumps({
                "response": "I'm currently operating in fallback mode due to LLM connectivity issues. " 
                           "I can provide basic assistance but may not have full capabilities.",
                "confidence": 0.5,
                "fallback": True
            })
        else:
            return "I'm currently operating in fallback mode due to LLM connectivity issues. " \
                   "I can provide basic assistance but may not have full capabilities."
            
    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict:
        """Generate JSON response using Ollama model.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (overrides default)
            max_tokens: Maximum tokens (overrides default)
            
        Returns:
            Parsed JSON response
        """
        response_text = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True
        )
        
        try:
            # Find JSON content in response (looking for curly braces)
            json_start = response_text.find('{')
            json_end = response_text.rfind('}')
            
            if json_start >= 0 and json_end >= 0:
                json_str = response_text[json_start:json_end+1]
                return json.loads(json_str)
            else:
                # Try parsing the whole response
                return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            logger.debug(f"Response text: {response_text}")
            
            # Return a basic fallback structure
            if "fallback" in response_text:
                return {"fallback": True, "confidence": 0.5}
            else:
                return {} 