"""
Main entry point for the AgentNet Customer Support System.
"""

from typing import Dict, List
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from loguru import logger

from .network.network import AgentNetwork

# Initialize FastAPI app
app = FastAPI(
    title="AgentNet Customer Support",
    description="AI-Driven Customer Support System using Multi-Agent Architecture"
)

# Initialize agent network
network = AgentNetwork(
    num_agents=5,
    capability_dim=10,
    memory_size=1000,
    llm_model="ollama/llama2"
)

class SupportRequest(BaseModel):
    """Model for customer support requests."""
    query: str
    context: Dict = {}
    priority: float = 1.0
    capabilities: List[float] = None

@app.post("/support")
async def handle_support_request(request: SupportRequest):
    """Handle incoming customer support requests.
    
    Args:
        request: Customer support request
        
    Returns:
        Dict containing support response and routing information
    """
    try:
        # Prepare task dictionary
        task = {
            "observation": request.query,
            "context": request.context,
            "priority": request.priority
        }
        
        # Add capability requirements if provided
        if request.capabilities:
            task["capabilities"] = np.array(request.capabilities)
            
        # Process task through agent network
        result = network.process_task(
            task=task,
            alpha=0.8,  # Weight decay factor
            max_hops=5  # Maximum routing hops
        )
        
        if result["status"] == "completed":
            return {
                "status": "success",
                "response": result["result"]["response"],
                "analysis": result["result"]["analysis"],
                "confidence": result["result"]["confidence"],
                "routing_path": result["routing_path"]
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Task processing failed: {result['status']}"
            )
            
    except Exception as e:
        logger.error(f"Error processing support request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/network/status")
async def get_network_status():
    """Get current status of the agent network.
    
    Returns:
        Dict containing network state information
    """
    try:
        return {
            "num_agents": network.num_agents,
            "agents": network._get_network_state(),
            "weight_matrix": network.weight_matrix.tolist()
        }
    except Exception as e:
        logger.error(f"Error getting network status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

def main():
    """Run the FastAPI server."""
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

if __name__ == "__main__":
    main() 