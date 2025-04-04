#!/usr/bin/env python
"""
Historical data test script for the AgentNet Customer Support System.
This script tests the system using historical ticket data and solution templates.
"""

import os
import sys
import time
import random
import csv
import re
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import pandas as pd

class SimulatedAgentNetwork:
    """Simulated agent network for testing."""
    
    def __init__(self, num_agents=5, capability_dim=10, memory_size=1000, 
                 llm_model="ollama/llama2", historical_data_path=None, 
                 solution_templates_dir=None):
        """Initialize simulated agent network.
        
        Args:
            num_agents: Number of agents in the network.
            capability_dim: Dimension of capability vectors.
            memory_size: Size of memory for each agent.
            llm_model: LLM model to use for generation.
            historical_data_path: Path to historical data file.
            solution_templates_dir: Directory with solution templates.
        """
        self.num_agents = num_agents
        self.capability_dim = capability_dim
        self.memory_size = memory_size
        self.llm_model = llm_model
        
        # Track if Ollama is connected
        self.is_ollama_connected = self._check_ollama_connection()
        
        # Initialize agents with different capabilities
        self.agents = {}
        specialties = [
            "Query routing and classification",
            "Technical support and troubleshooting",
            "Account management and data issues",
            "Billing and subscription support",
            "Payment systems and integration"
        ]
        
        for i in range(num_agents):
            specialty = specialties[i] if i < len(specialties) else f"General support {i}"
            self.agents[f"agent_{i}"] = {
                "name": f"Agent {i}",
                "specialty": specialty,
                "capability": np.random.random(capability_dim),
                "memory": [],
                "historical_data": []
            }
        
        # Team assignments for routing
        self.teams = {
            "development": {
                "description": "Handles software development issues, bugs, and technical implementation",
                "contact": "dev-team@agentnet.com",
                "issue_types": ["Software Installation Failure", "Integration Error", "Performance Issues"]
            },
            "operations": {
                "description": "Manages infrastructure, deployments, and operational concerns",
                "contact": "ops-team@agentnet.com",
                "issue_types": ["Network Connectivity Issue", "Server Downtime", "Deployment Problems"]
            },
            "security": {
                "description": "Addresses security concerns, authentication issues, and data protection",
                "contact": "security@agentnet.com",
                "issue_types": ["Authentication Failure", "Data Breach", "Permission Issues"]
            },
            "customer_success": {
                "description": "Provides account management and customer happiness support",
                "contact": "customer-support@agentnet.com",
                "issue_types": ["Account Synchronization Bug", "Account Access", "Feature Requests"]
            },
            "product": {
                "description": "Handles product-specific issues and roadmap items",
                "contact": "product@agentnet.com",
                "issue_types": ["Payment Gateway Integration Failure", "User Interface Issues", "Functionality Gaps"]
            }
        }
        
        # Load historical data if provided
        self.historical_data = []
        if historical_data_path:
            self.historical_data = self._load_historical_data(historical_data_path)
            print(f"Loaded {len(self.historical_data)} historical tickets")
        
        # Load solution templates if provided
        self.solution_templates = []
        if solution_templates_dir:
            self.solution_templates = self._load_solution_templates(solution_templates_dir)
            print(f"Loaded {len(self.solution_templates)} solution templates")
        
        # Build resolution time database
        self.resolution_time_db = self._build_resolution_time_db()
    
    def _check_ollama_connection(self):
        """Check if Ollama is accessible."""
        try:
            # This is a simple check - in a real implementation, you'd check the actual Ollama service
            return True
        except:
            return False
    
    def process_ollama(self, query, context=None):
        """Process a query through Ollama LLM layer.
        
        Args:
            query: The user query to process
            context: Any additional context
            
        Returns:
            Ollama's response or a fallback if Ollama is not available
        """
        # This simulates the Ollama layer - in a real implementation, you'd call the Ollama API
        if not self.is_ollama_connected:
            return "Ollama is not available. Using fallback responses."
            
        try:
            # Simulate Ollama processing time
            time.sleep(0.5)
            
            # In a real implementation, this is where you'd call Ollama's API
            # For this simulation, we'll just use our internal processing
            
            # Simulate Ollama's processing of the query
            processed_query = {
                "original_query": query,
                "processed_text": query,
                "embedding": np.random.random(384),  # Simulate an embedding vector
                "metadata": {
                    "model": self.llm_model,
                    "processing_time": 0.5,
                    "tokens": len(query.split())
                }
            }
            
            return processed_query
            
        except Exception as e:
            print(f"Error with Ollama processing: {str(e)}")
            return {
                "original_query": query,
                "processed_text": query,
                "error": str(e)
            }
    
    def process_task(self, task, alpha=0.8, max_hops=5):
        """Process a task using the agent network.
        
        Args:
            task: Task to process.
            alpha: Weighting parameter for agent selection.
            max_hops: Maximum number of agent hops.
            
        Returns:
            Result of task processing.
        """
        routing_path = []
        current_agent = "agent_0"  # Always start with the router agent
        current_observation = task["observation"]
        
        # First, process through Ollama layer
        ollama_processed = self.process_ollama(current_observation)
        
        # If Ollama returned an error, use the original query
        if isinstance(ollama_processed, dict) and "error" not in ollama_processed:
            processed_query = ollama_processed["processed_text"]
        else:
            processed_query = current_observation
        
        # Now process through our agent network
        for hop in range(max_hops):
            # Add current agent to routing path
            routing_path.append(current_agent)
            
            # Get agent response
            response = self._get_agent_response(current_agent, processed_query, task["context"])
            
            # Check if we should route to another agent
            if "route_to" in response and response["route_to"] and hop < max_hops - 1:
                current_agent = response["route_to"]
                continue
                
            # If we have a final response, return it
            if "result" in response:
                return {
                    "status": "completed",
                    "routing_path": routing_path,
                    "result": response["result"]
                }
        
        # If we reached max hops without a result
        return {
            "status": "incomplete",
            "routing_path": routing_path,
            "message": "Reached maximum number of agent hops without resolution."
        }
    
    def _load_historical_data(self, file_path: str):
        """Load historical ticket data from CSV file."""
        data = []
        try:
            with open(file_path, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Clean up column names by stripping whitespace
                    cleaned_row = {k.strip(): v.strip() for k, v in row.items()}
                    data.append(cleaned_row)
            return data
        except Exception as e:
            print(f"Error loading historical data: {str(e)}")
            # Create dummy data in case of error
            dummy_data = [
                {"Ticket ID": f"TECH_{i:03d}", 
                 "Issue Category": ["Software Installation Failure", "Network Connectivity Issue", 
                              "Device Compatibility Error", "Account Synchronization Bug",
                              "Payment Gateway Integration Failure"][i % 5], 
                 "Priority": ["High", "Medium", "Critical"][i % 3],
                 "Solution": ["Disable antivirus", "Check permissions", "Rollback app version", 
                        "Reset sync token", "Upgrade TLS version"][i % 5],
                 "Resolution Status": "Resolved",
                 "Date of Resolution": "2025-03-15"} 
                for i in range(1, 16)
            ]
            return dummy_data
    
    def _load_solution_templates(self, directory: str):
        """Load solution templates from text files."""
        templates = []
        try:
            for filename in os.listdir(directory):
                if filename.endswith('.txt'):
                    category = os.path.splitext(filename)[0]
                    with open(os.path.join(directory, filename), 'r') as file:
                        content = file.read()
                        templates.append({
                            "category": category,
                            "content": content
                        })
            return templates
        except Exception as e:
            print(f"Error loading solution templates: {str(e)}")
            # Return some dummy templates in case of error
            return [
                {"category": "Software_Installation_Failure", 
                 "content": "1. Disable antivirus temporarily\n2. Run installer as administrator\n3. Check system requirements"},
                {"category": "Network_Connectivity_Issue", 
                 "content": "1. Check app permissions\n2. Verify network connection\n3. Reset network settings"},
                {"category": "Device_Compatibility_Error", 
                 "content": "1. Verify device meets minimum requirements\n2. Update firmware\n3. Try compatibility mode"},
                {"category": "Account_Synchronization_Bug", 
                 "content": "1. Reset sync tokens\n2. Clear cache\n3. Verify account status"},
                {"category": "Payment_Gateway_Integration_Failure", 
                 "content": "1. Check API credentials\n2. Verify SSL settings\n3. Update TLS version"}
            ]
    
    def _build_resolution_time_db(self):
        """Build a database of resolution times based on issue types and priorities."""
        # Default resolution times in minutes
        default_times = {
            "Software Installation Failure": {"High": 45, "Medium": 60, "Critical": 30},
            "Network Connectivity Issue": {"High": 40, "Medium": 50, "Critical": 25},
            "Device Compatibility Error": {"High": 55, "Medium": 70, "Critical": 35},
            "Account Synchronization Bug": {"High": 30, "Medium": 45, "Critical": 20},
            "Payment Gateway Integration Failure": {"High": 60, "Medium": 90, "Critical": 40}
        }
        
        # Enhance with historical data if available
        if self.historical_data:
            # Parse resolution times from historical data if available
            # This would be a real implementation using actual historical resolution times
            pass
            
        return default_times
    
    def _find_matching_ticket(self, query: str) -> Tuple[Dict, float]:
        """Find the most relevant historical ticket for a given query."""
        best_match = None
        best_score = -1
        
        # Simple keyword matching for simulation purposes
        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        
        for ticket in self.historical_data:
            category = ticket.get('Issue Category', '')
            solution = ticket.get('Solution', '')
            
            # Create a combined text to match against
            combined_text = (category + " " + solution).lower()
            
            # Calculate match based on word overlap
            category_words = set(re.findall(r'\b\w+\b', combined_text))
            
            # Find common words
            common_words = query_words.intersection(category_words)
            
            # Calculate score based on overlap
            if len(query_words) > 0:
                score = len(common_words) / len(query_words)
            else:
                score = 0
                
            if score > best_score:
                best_score = score
                best_match = ticket
                
        return best_match, best_score
    
    def _generate_summary(self, query: str) -> str:
        """Generate a concise summary of the customer query."""
        # In a real implementation, this would use an LLM for summarization
        # Here we'll simulate a summarization process
        
        words = query.split()
        if len(words) <= 10:
            return query
            
        # Simple extractive summarization for simulation
        key_phrases = [
            "can't log in", "password incorrect", "forgot password", 
            "subscription", "payment failed", "charge twice", "refund",
            "app crash", "installation fail", "update problem", 
            "error message", "doesn't work", "not compatible",
            "data missing", "sync issue", "can't connect"
        ]
        
        found_phrases = []
        query_lower = query.lower()
        
        for phrase in key_phrases:
            if phrase in query_lower:
                found_phrases.append(phrase)
                
        if found_phrases:
            summary = f"Customer reports issues with: {', '.join(found_phrases)}"
        else:
            # Take first 10 words as summary
            summary = " ".join(words[:10]) + "..."
            
        return summary
    
    def _extract_actions(self, query: str, issue_category: str = None) -> List[Dict]:
        """Extract actionable items from the customer query."""
        # In a real implementation, this would use an LLM to extract actions
        # Here we'll simulate action extraction based on query content and issue category
        
        actions = []
        query_lower = query.lower()
        
        # Check for potential escalation triggers
        escalation_triggers = [
            "urgent", "immediately", "emergency", "critical", 
            "asap", "right now", "lost data", "security breach",
            "production down", "can't work", "deadline"
        ]
        
        for trigger in escalation_triggers:
            if trigger in query_lower:
                actions.append({
                    "type": "escalation",
                    "description": f"Escalate due to urgency indicator: '{trigger}'",
                    "priority": "High"
                })
                break
                
        # Check for potential follow-up requirements
        followup_triggers = [
            "call me", "contact me", "let me know", "update me",
            "follow up", "get back to me", "when will", "how long"
        ]
        
        for trigger in followup_triggers:
            if trigger in query_lower:
                actions.append({
                    "type": "follow_up",
                    "description": f"Schedule follow-up about resolution progress",
                    "timeline": "24 hours"
                })
                break
                
        # Add category-specific actions
        if issue_category:
            if "Software Installation" in issue_category:
                actions.append({
                    "type": "troubleshooting",
                    "description": "Request system specifications and error logs",
                    "team": "technical_support"
                })
            elif "Network Connectivity" in issue_category:
                actions.append({
                    "type": "verification",
                    "description": "Verify network permissions and app configuration",
                    "team": "technical_support"
                })
            elif "Device Compatibility" in issue_category:
                actions.append({
                    "type": "compatibility_check",
                    "description": "Check device against compatibility database",
                    "team": "product"
                })
            elif "Account Synchronization" in issue_category:
                actions.append({
                    "type": "sync_reset",
                    "description": "Reset account synchronization tokens",
                    "team": "development"
                })
            elif "Payment Gateway" in issue_category:
                actions.append({
                    "type": "payment_verification",
                    "description": "Verify transaction logs and payment status",
                    "team": "billing"
                })
                
        return actions
    
    def _determine_responsible_team(self, issue_category: str, query: str) -> Dict:
        """Determine which team should handle this issue."""
        # Map issue categories to responsible teams
        team_mapping = {
            "Software Installation Failure": "development",
            "Network Connectivity Issue": "operations",
            "Device Compatibility Error": "product",
            "Account Synchronization Bug": "development",
            "Payment Gateway Integration Failure": "security"
        }
        
        # First try to map based on issue category
        if issue_category in team_mapping:
            team_id = team_mapping[issue_category]
        else:
            # If no direct mapping, determine based on query content
            query_lower = query.lower()
            
            if any(word in query_lower for word in ["code", "bug", "programming", "developer", "feature"]):
                team_id = "development"
            elif any(word in query_lower for word in ["server", "network", "infrastructure", "down", "slow"]):
                team_id = "operations"
            elif any(word in query_lower for word in ["password", "login", "authentication", "token", "breach"]):
                team_id = "security"
            elif any(word in query_lower for word in ["training", "onboarding", "how to", "tutorial"]):
                team_id = "customer_success"
            elif any(word in query_lower for word in ["feature request", "improvement", "roadmap", "suggestion"]):
                team_id = "product"
            else:
                # Default to customer success if no clear match
                team_id = "customer_success"
                
        return {
            "team_id": team_id,
            "team_info": self.teams.get(team_id, {"description": "Unknown team", "contact": "support@example.com"})
        }
    
    def _estimate_resolution_time(self, issue_category: str, priority: str, complexity_factors: List[str] = None) -> Dict:
        """Estimate the resolution time for this issue."""
        # Get base resolution time from historical data
        base_time = 60  # Default to 60 minutes
        
        if issue_category in self.resolution_time_db and priority in self.resolution_time_db[issue_category]:
            base_time = self.resolution_time_db[issue_category][priority]
            
        # Apply complexity factors
        time_multiplier = 1.0
        
        if complexity_factors:
            # Each complexity factor can adjust the time estimation
            if "multiple_issues" in complexity_factors:
                time_multiplier *= 1.5
            if "legacy_system" in complexity_factors:
                time_multiplier *= 1.3
            if "third_party_dependency" in complexity_factors:
                time_multiplier *= 1.4
            if "data_migration" in complexity_factors:
                time_multiplier *= 1.6
            if "custom_implementation" in complexity_factors:
                time_multiplier *= 1.25
                
        # Calculate estimated resolution time
        estimated_minutes = round(base_time * time_multiplier)
        
        # Generate completion time based on current time
        current_time = datetime.now()
        completion_time = current_time + timedelta(minutes=estimated_minutes)
        
        # Return time estimate details
        return {
            "estimated_minutes": estimated_minutes,
            "estimated_completion": completion_time.strftime("%Y-%m-%d %H:%M"),
            "base_time": base_time,
            "complexity_multiplier": time_multiplier,
            "confidence": 0.85
        }
        
    def _get_agent_response(self, agent_id: str, query: str, context: Dict = None) -> Dict:
        """Get a response from a specific agent."""
        # This is a placeholder implementation. In a real system, this would involve
        # calling the appropriate agent's LLM and returning the response.
        # For now, we'll simulate a response based on the agent's capabilities.
        
        if not agent_id or agent_id not in self.agents:
            agent_id = "agent_0"  # Default to the first agent if invalid
        
        agent = self.agents[agent_id]
        
        # Find matching ticket to simulate historical data lookup
        matching_ticket, match_score = self._find_matching_ticket(query)
        
        # Determine issue category and priority
        category = "General Inquiry"
        priority = "Medium"
        
        if matching_ticket:
            category = matching_ticket.get('Issue Category', 'General Inquiry')
            priority = matching_ticket.get('Priority', 'Medium')
        
        # Create a simulated response
        response = {}
        
        # Determine if this is a final response or should be routed
        agent_idx = int(agent_id.split('_')[1])
        is_specialized = False
        
        # The first agent (router) just routes, other agents may route or execute
        if agent_idx == 0:
            # Router agent - always routes to a specialized agent
            specialized_idx = self._determine_specialist_for_query(query)
            next_agent = f"agent_{specialized_idx}"
            response["route_to"] = next_agent
            
        elif random.random() < 0.8:  # 80% chance to provide a final answer for specialized agents
            # Generate a final response
            sentiment = random.choice(["Frustrated", "Confused", "Anxious", "Urgent", "Neutral"])
            
            # Get analysis and solution
            analysis = self._generate_analysis(query, category)
            solution = self._generate_solution(query, category)
            
            # Create actions
            actions = self._extract_actions(query, category)
            
            # Determine responsible team
            team_assignment = self._determine_responsible_team(query, category)
            
            # Estimate resolution time
            resolution_time = self._estimate_resolution_time(query, priority)
            
            response["result"] = {
                "summary": self._generate_summary(query),
                "analysis": analysis,
                "solution": solution,
                "response": self._generate_response(query, solution, sentiment),
                "confidence": 0.7 + random.random() * 0.3,  # Random confidence between 0.7 and 1.0
                "matching_ticket": matching_ticket,
                "match_score": match_score,
                "actions": actions,
                "team_assignment": team_assignment,
                "resolution_time_estimate": resolution_time,
                "category": category,
                "priority": priority,
                "sentiment": sentiment
            }
        else:
            # Route to another agent
            current_idx = int(agent_id.split('_')[1])
            other_agents = [i for i in range(1, self.num_agents) if i != current_idx]
            if other_agents:
                next_agent_idx = random.choice(other_agents)
                response["route_to"] = f"agent_{next_agent_idx}"
            else:
                # If somehow there are no other agents, provide a final response
                response["result"] = {
                    "summary": self._generate_summary(query),
                    "analysis": f"Agent {agent_id} is processing the query.",
                    "solution": "Standard solution applied based on query analysis.",
                    "response": "I've analyzed your issue and applied our standard resolution procedure. Please let me know if this resolves your problem.",
                    "confidence": 0.7,
                    "matching_ticket": None,
                    "match_score": 0.0,
                    "actions": [],
                    "team_assignment": self._determine_responsible_team(query, "General Inquiry"),
                    "resolution_time_estimate": self._estimate_resolution_time(query, "Medium"),
                    "category": "General Inquiry",
                    "priority": "Medium",
                    "sentiment": "Neutral"
                }
        
        return response
        
    def _determine_specialist_for_query(self, query: str) -> int:
        """Determine which specialist agent should handle this query."""
        query_lower = query.lower()
        
        # Simple keyword matching for routing
        if any(word in query_lower for word in ["install", "software", "download", "setup", "update"]):
            return 1  # Technical support specialist
        elif any(word in query_lower for word in ["account", "login", "password", "username", "profile"]):
            return 2  # Account management specialist
        elif any(word in query_lower for word in ["bill", "payment", "subscription", "charge", "refund"]):
            return 3  # Billing specialist
        elif any(word in query_lower for word in ["api", "integration", "gateway", "connect", "webhook"]):
            return 4  # Integration specialist
        else:
            # Default to a random specialist if no clear match
            return random.randint(1, min(4, self.num_agents - 1))
    
    def _generate_analysis(self, query: str, category: str) -> str:
        """Generate an analysis of the customer query."""
        # In a real implementation, this would use an LLM to generate an analysis
        # Here we'll provide simulated analyses based on the category
        
        analyses = {
            "Software Installation Failure": 
                "The customer is experiencing issues with software installation, which appears to be failing during the process. " 
                "This could be due to antivirus blocking, insufficient permissions, or compatibility issues with their system.",
            
            "Network Connectivity Issue":
                "The customer is having trouble connecting to our service through the network. "
                "This is likely related to local network configurations, firewalls, or app permissions.",
            
            "Device Compatibility Error":
                "The customer is experiencing compatibility issues between our software and their device. "
                "This suggests either outdated firmware, unsupported hardware, or software version mismatch.",
            
            "Account Synchronization Bug":
                "The customer is facing problems with account data synchronization across devices. "
                "This is typically caused by authentication token issues, cached data conflicts, or server sync delays.",
            
            "Payment Gateway Integration Failure":
                "The customer is encountering errors when attempting to integrate our payment gateway. "
                "This could be related to API credentials, SSL configuration, or outdated TLS settings."
        }
        
        if category in analyses:
            return analyses[category]
        else:
            # Generic analysis for other categories
            return ("After analyzing your query, I understand you're experiencing an issue that requires attention. "
                   f"Based on the details provided, this appears to be related to {category}. "
                   "I'll help identify the root cause and provide appropriate solutions.")
    
    def _generate_solution(self, query: str, category: str) -> str:
        """Generate a solution based on the query and category."""
        # In a real implementation, this would use an LLM and historical data
        # Here we'll provide simulated solutions based on the category
        
        solutions = {
            "Software Installation Failure": 
                "1. Temporarily disable your antivirus software\n"
                "2. Run the installer as administrator\n"
                "3. Clear the temporary installation files\n"
                "4. Make sure your system meets the minimum requirements\n"
                "5. Try using the offline installer from our website",
            
            "Network Connectivity Issue":
                "1. Check that your app has necessary permissions\n"
                "2. Verify your network connection is stable\n"
                "3. Temporarily disable any VPN or firewall\n"
                "4. Clear the app cache and data\n"
                "5. Reinstall the application if the issue persists",
            
            "Device Compatibility Error":
                "1. Update your device firmware to the latest version\n"
                "2. Try using compatibility mode if available\n"
                "3. Check our website for a list of supported devices\n"
                "4. Roll back to an earlier version of our software\n"
                "5. Contact support for device-specific solutions",
            
            "Account Synchronization Bug":
                "1. Sign out and back into your account on all devices\n"
                "2. Clear cache and cookies in your browser/app\n"
                "3. Force a manual sync through account settings\n"
                "4. Verify your account status is active\n"
                "5. Reset your sync token via customer support",
            
            "Payment Gateway Integration Failure":
                "1. Verify your API credentials are correct\n"
                "2. Ensure your server supports TLS 1.2 or higher\n"
                "3. Check your SSL certificate is valid and properly installed\n"
                "4. Confirm your server's IP is whitelisted in our dashboard\n"
                "5. Test with our sandbox environment before going live"
        }
        
        if category in solutions:
            return solutions[category]
        else:
            # Generic solution for other categories
            return ("Based on your issue, I recommend the following steps:\n"
                   "1. Review our documentation for common issues\n"
                   "2. Clear your cache and restart the application\n"
                   "3. Verify you're using the latest version\n"
                   "4. Check your account permissions\n"
                   "5. Contact our support team if the issue persists")
    
    def _generate_response(self, query: str, solution: str, sentiment: str) -> str:
        """Generate a customer-friendly response based on the solution and detected sentiment."""
        # In a real implementation, this would use an LLM to generate a response
        # Here we'll provide simulated responses based on sentiment and solution
        
        # Adjust tone based on customer sentiment
        if sentiment == "Frustrated" or sentiment == "Urgent":
            intro = ("I understand how frustrating this situation is, and I'm going to help you resolve it right away. "
                    "I appreciate your patience as we work through this issue together.")
        elif sentiment == "Confused":
            intro = ("I can see how this might be confusing, and I'm here to clear things up for you. "
                    "Let me guide you through this step by step.")
        elif sentiment == "Anxious":
            intro = ("I understand this issue is causing you concern, and I want to assure you that we'll get it resolved. "
                    "Let me help you through this with a straightforward solution.")
        else:
            intro = ("Thank you for reaching out about this issue. "
                    "I'm happy to help you resolve this situation.")
        
        # Create response by combining intro, problem acknowledgment, and solution
        problem_acknowledgment = f"Based on your description, I can see that you're experiencing an issue with {query[:30].lower()}..."
        
        solution_intro = "Here's what I recommend to resolve this:"
        
        conclusion = ("If you need any clarification or have questions about these steps, "
                     "please don't hesitate to ask. "
                     "I'm here to help you get this resolved as quickly as possible.")
        
        # Combine all parts
        response = f"{intro} {problem_acknowledgment}\n\n{solution_intro}\n\n{solution}\n\n{conclusion}"
        
        return response

def run_historical_test(query: str):
    """Run a test with historical data."""
    print("\n=== AgentNet Customer Support System Historical Data Test ===")
    print(f"Processing query: \"{query}\"\n")
    
    print("1. Initializing the agent network...")
    
    # Create the simulated agent network with historical data
    network = SimulatedAgentNetwork(
        num_agents=5,
        capability_dim=10,
        memory_size=1000,
        llm_model="ollama/llama2",
        historical_data_path="data/raw/Historical_ticket_data.csv",
        solution_templates_dir="data/raw"
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
    print(f"Summary: {result['result']['summary']}")
    print(f"Category: {result['result']['category']}")
    print(f"Priority: {result['result']['priority']}")
    print(f"Sentiment: {result['result']['sentiment']}")
    print(f"Routing path: {' -> '.join(result['routing_path'])}")
    
    if result["status"] == "completed":
        print("\nAnalysis:")
        print(f"{result['result']['analysis']}")
        
        print("\nSolution:")
        print(f"{result['result']['solution']}")
        
        print("\nResponse:")
        print(f"{result['result']['response']}")
        
        print("\nConfidence:")
        print(f"{result['result']['confidence']:.2f}")
        
        if result["result"]["actions"]:
            print("\nRequired Actions:")
            for action in result["result"]["actions"]:
                print(f"* {action['type']}: {action['description']}")
                
        print("\nTeam Assignment:")
        team = result["result"]["team_assignment"]
        print(f"* Assigned to: {team['team_id']} team")
        print(f"* Team description: {team['team_info']['description']}")
        print(f"* Contact: {team['team_info']['contact']}")
        
        print("\nResolution Time Estimate:")
        time_est = result["result"]["resolution_time_estimate"]
        print(f"* Estimated resolution time: {time_est['estimated_minutes']} minutes")
        print(f"* Expected completion by: {time_est['estimated_completion']}")
        print(f"* Confidence in estimate: {time_est['confidence']:.2f}")
        
        # Display matching ticket if available
        if result["result"]["matching_ticket"]:
            print("\nMatching historical ticket:")
            ticket = result["result"]["matching_ticket"]
            print(f"Ticket ID: {ticket.get('Ticket ID', 'N/A')}")
            print(f"Issue Category: {ticket.get('Issue Category', 'N/A')}")
            print(f"Priority: {ticket.get('Priority', 'N/A')}")
            print(f"Solution: {ticket.get('Solution', 'N/A')}")
            print(f"Match Score: {result['result']['match_score']:.2f}")
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
        query = "I'm having trouble installing your software. It keeps failing at 75% with an unknown error."
    
    run_historical_test(query) 