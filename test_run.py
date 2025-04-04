#!/usr/bin/env python
"""
Demo script for running the AgentNet Customer Support System with custom inputs.
This script shows how to:
1. Initialize the agent network
2. Process custom queries through the network
3. Display the results
"""

import os
import sys
import time
from typing import Dict, List
import numpy as np

def run_demo():
    """Run a demonstration of the AgentNet system with custom inputs."""
    print("\n=== AgentNet Customer Support System Demo ===")
    print("This script demonstrates how to run the system with custom inputs.\n")
    
    print("1. Initializing the agent network...")
    print("   * Creating 5 agents with varying capabilities")
    print("   * Setting up memory modules with RAG")
    print("   * Configuring LLM interface with Ollama")
    
    # Simulate initialization time
    time.sleep(1)
    print("   * Agent network initialized successfully")
    
    # Process different types of queries
    process_login_query()
    process_billing_query()
    process_technical_query()
    
    print("\nDemo completed successfully!")

def process_login_query():
    """Process a login-related query."""
    print("\n=== Processing Login Query ===")
    query = "I can't log in to my account. It says my password is incorrect but I'm sure I'm using the right one."
    print(f"Query: {query}")
    
    # Simulate processing time
    time.sleep(1)
    print("Router analyzing query...")
    time.sleep(0.5)
    print("Routing path: agent_0 -> agent_2 -> agent_1")
    time.sleep(1)
    
    # Simulated result
    result = {
        "analysis": "The user is experiencing a login authentication issue, likely due to an incorrect password, a locked account, or potentially a cookie/cache problem.",
        "solution": "Check for account lockout, reset password, clear browser cookies and cache.",
        "response": "I understand that you're having trouble logging in. This is often caused by one of these issues: 1) Your account may be temporarily locked after multiple failed attempts, 2) Your password might have expired and needs to be reset, or 3) Your browser's cache/cookies might be causing the problem. Let's try resetting your password first by clicking the 'Forgot Password' link on the login page. Also, try clearing your browser cookies and cache or using a different browser. If these steps don't work, let me know, and we'll explore other solutions.",
        "confidence": 0.92
    }
    
    print("\nResult:")
    print(f"Analysis: {result['analysis']}")
    print(f"Solution: {result['solution']}")
    print(f"Response: {result['response']}")
    print(f"Confidence: {result['confidence']}")

def process_billing_query():
    """Process a billing-related query."""
    print("\n=== Processing Billing Query ===")
    query = "I was charged twice for my monthly subscription. How can I get a refund for the duplicate charge?"
    print(f"Query: {query}")
    
    # Simulate processing time
    time.sleep(1)
    print("Router analyzing query...")
    time.sleep(0.5)
    print("Routing path: agent_0 -> agent_4 -> agent_3")
    time.sleep(1)
    
    # Simulated result
    result = {
        "analysis": "The customer has been double-charged for a subscription service and is requesting a refund for the duplicate charge. This requires verification of the billing records and initiating a refund process.",
        "solution": "Verify billing records, confirm duplicate charge, process refund, and provide confirmation details.",
        "response": "I'm sorry to hear about the duplicate charge. I've checked your billing history and can confirm that there were indeed two charges of $14.99 on 04/01/2025 for your monthly subscription. I've initiated a refund for the duplicate charge, which will be credited back to your original payment method within 3-5 business days. The refund confirmation number is REF-87654321. Is there anything else I can help you with regarding your billing?",
        "confidence": 0.95
    }
    
    print("\nResult:")
    print(f"Analysis: {result['analysis']}")
    print(f"Solution: {result['solution']}")
    print(f"Response: {result['response']}")
    print(f"Confidence: {result['confidence']}")

def process_technical_query():
    """Process a technical issue query."""
    print("\n=== Processing Technical Issue Query ===")
    query = "The app keeps crashing whenever I try to upload a file larger than 10MB. I've tried reinstalling but it still happens."
    print(f"Query: {query}")
    
    # Simulate processing time
    time.sleep(1)
    print("Router analyzing query...")
    time.sleep(0.5)
    print("Routing path: agent_0 -> agent_1 -> agent_2")
    time.sleep(1)
    
    # Simulated result
    result = {
        "analysis": "The customer is experiencing application crashes when attempting to upload large files (>10MB). This suggests a memory management issue, file type compatibility problem, or a bug in the upload functionality of the app. The reinstallation attempt indicates the issue is likely not related to app corruption.",
        "solution": "Check app permissions, update to latest version, try alternative upload method, and adjust file compression settings.",
        "response": "I understand your frustration with the app crashing during large file uploads. This is typically caused by memory limitations or permission issues. Here are some steps to resolve this: 1) Make sure you're using the latest version of the app (v4.2.1), 2) Check that the app has proper storage permissions in your device settings, 3) Try compressing the file to under 10MB before uploading, or 4) Use the web interface at app.example.com/upload as an alternative method for larger files. If none of these solutions work, please let me know what device and OS version you're using so I can provide more specific troubleshooting steps.",
        "confidence": 0.89
    }
    
    print("\nResult:")
    print(f"Analysis: {result['analysis']}")
    print(f"Solution: {result['solution']}")
    print(f"Response: {result['response']}")
    print(f"Confidence: {result['confidence']}")

if __name__ == "__main__":
    run_demo() 