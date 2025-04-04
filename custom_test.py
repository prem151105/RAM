#!/usr/bin/env python
"""
Custom test script for the AgentNet Customer Support System.
This script allows you to test the system with your own input.
This version uses a simulation to demonstrate the system's functionality.
"""

import os
import sys
import time
import random
from typing import Dict, List
import numpy as np

class SimulatedAgentNetwork:
    """A simulated version of the AgentNetwork for testing purposes."""
    
    def __init__(
        self,
        num_agents: int = 5,
        capability_dim: int = 10,
        memory_size: int = 1000,
        llm_model: str = "ollama/llama2"
    ):
        """Initialize the simulated agent network."""
        self.num_agents = num_agents
        self.capability_dim = capability_dim
        self.llm_model = llm_model
        
        # Create agent IDs
        self.agents = {f"agent_{i}": {} for i in range(num_agents)}
        
    def process_task(
        self,
        task: Dict,
        alpha: float = 0.8,
        max_hops: int = 5
    ) -> Dict:
        """Simulate processing a task through the agent network."""
        # Extract query from task
        query = task["observation"]
        
        # Simulate some processing time
        time.sleep(1)
        
        # Determine query type based on keywords
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["subscription", "payment", "renew", "charge", "bill"]):
            query_type = "billing"
            routing_path = ["agent_0", "agent_4", "agent_3"]
            confidence = 0.93
        elif any(word in query_lower for word in ["login", "password", "account", "sign in", "access"]):
            query_type = "login"
            routing_path = ["agent_0", "agent_2", "agent_1"]
            confidence = 0.91
        elif any(word in query_lower for word in ["app", "crash", "bug", "error", "feature"]):
            query_type = "technical"
            routing_path = ["agent_0", "agent_1", "agent_2"]
            confidence = 0.88
        elif any(word in query_lower for word in ["refund", "return", "cancel", "damage", "broken", "replacement"]):
            query_type = "product_issue"
            routing_path = ["agent_0", "agent_3", "agent_4"]
            confidence = 0.94
        else:
            query_type = "general"
            # Random routing path for general queries
            num_hops = random.randint(2, 4)
            routing_path = ["agent_0"]
            for _ in range(num_hops):
                next_agent = f"agent_{random.randint(1, self.num_agents-1)}"
                if next_agent not in routing_path:
                    routing_path.append(next_agent)
            confidence = 0.85
        
        # Generate a simulated response based on query type
        if query_type == "billing":
            analysis = "The customer is experiencing an issue with their subscription renewal, resulting in loss of access to premium features. This requires verification of payment status and account status."
            solution = "Verify subscription status, check payment method, offer manual renewal options, and provide temporary access if needed."
            response = "I understand you're having trouble with your subscription renewal and can't access premium features. I've checked your account and see that your last payment attempt on May 15th was declined due to an expired credit card. To fix this: 1) Update your payment method in the Account Settings, 2) Click 'Renew Subscription' to try again, or 3) If you continue to have issues, I can process a manual renewal for you. Would you like me to temporarily enable your premium access while you sort out the payment details?"
        
        elif query_type == "login":
            analysis = "The customer is experiencing issues accessing their account due to authentication problems, possibly related to credentials, account status, or system issues."
            solution = "Check account status, reset credentials if needed, verify email confirmation, and investigate possible system issues."
            response = "I see you're having trouble accessing your premium features due to subscription issues. After checking your account, I can see that your last renewal attempt on the 12th failed due to insufficient funds. To resolve this: 1) Please update your payment method in Account Settings, 2) Click on 'Renew Now', or 3) Contact your bank to ensure there are no holds on transactions to our company. In the meantime, I've activated a 3-day temporary access period for you so you can continue using the premium features while resolving the payment issue."
        
        elif query_type == "technical":
            analysis = "The customer is experiencing a technical issue with application functionality, possibly related to device compatibility, software version, or user permissions."
            solution = "Diagnose specific error type, check for updates, verify device compatibility, and provide workarounds."
            response = "I understand you're encountering technical difficulties with our application. Based on the information you've provided, this could be related to version compatibility or permission settings. Please try the following: 1) Ensure you're using the latest version (v3.2.1) of our app, 2) Clear your app cache through Settings > Storage > Clear Cache, 3) Check that the app has all necessary permissions enabled on your device. If these steps don't resolve your issue, please let me know what specific error messages you're seeing and what device/OS you're using, so I can provide more targeted assistance."
        
        elif query_type == "product_issue":
            analysis = "The customer has received a damaged or defective product and is seeking a replacement. This requires product verification, warranty check, and return processing."
            solution = "Verify purchase details, process return authorization, arrange shipping for replacement, and provide tracking information."
            response = "I'm sorry to hear that you received a damaged product. I'd be happy to arrange a replacement for you right away. I've looked up your order (#ORD-12345) from April 18th, and I see the item is still under our 30-day replacement policy. Here's what we'll do: 1) I've already processed a replacement request (REF-78901), 2) You'll receive a prepaid return label by email within the next hour, 3) Once you ship the damaged item back, we'll send out the replacement immediately - no need to wait for us to receive the return. Your replacement should arrive within 3-5 business days. Is there anything specific about the damage that our quality team should know about?"
        
        else:  # general
            analysis = "The customer has submitted a general inquiry that requires information gathering and possibly routing to a specialized department for complete resolution."
            solution = "Provide general information, gather specific details about the request, and route to appropriate specialized agent if needed."
            response = "Thank you for reaching out to us. I understand your question and would like to help you with this matter. To better assist you, could you please provide a few more details about your specific situation? This will help me ensure you get the most accurate and helpful information. In the meantime, you might find some useful information in our FAQ section at example.com/faq, which covers many common questions on this topic."
            
        # Create the result dictionary
        result = {
            "status": "completed",
            "routing_path": routing_path,
            "result": {
                "analysis": analysis,
                "solution": solution,
                "response": response,
                "confidence": confidence
            }
        }
        
        return result

def run_custom_test(query: str):
    """Run a test with a custom user query."""
    print("\n=== AgentNet Customer Support System Custom Test ===")
    print(f"Processing query: \"{query}\"\n")
    
    print("1. Initializing the agent network...")
    
    # Create the simulated agent network
    network = SimulatedAgentNetwork(
        num_agents=5,
        capability_dim=10,
        memory_size=1000,
        llm_model="ollama/llama2"
    )
    
    print("   * Agent network initialized successfully")
    
    # Create task dictionary
    task = {
        "observation": query,
        "context": {},
        "priority": 1.0
    }
    
    print("\n2. Processing query through agent network...")
    
    # Process task through agent network
    result = network.process_task(
        task=task,
        alpha=0.8,  # Weight decay factor
        max_hops=5  # Maximum routing hops
    )
    
    print("   * Query processed successfully")
    
    # Display results
    print("\n=== Results ===")
    print(f"Routing path: {' -> '.join(result['routing_path'])}")
    
    if result["status"] == "completed":
        print("\nResult:")
        print(f"Analysis: {result['result']['analysis']}")
        print(f"Solution: {result['result']['solution']}")
        print(f"Response: {result['result']['response']}")
        print(f"Confidence: {result['result']['confidence']}")
    else:
        print(f"\nStatus: {result['status']}")
        
    print("\nTest completed.")

if __name__ == "__main__":
    # Check if a query was provided as a command-line argument
    if len(sys.argv) > 1:
        # Join all arguments after the script name as the query
        query = " ".join(sys.argv[1:])
    else:
        # Use a default query if none provided
        query = "I received a damaged product. How can I get a replacement?"
    
    run_custom_test(query) 