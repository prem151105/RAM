"""
Data generator for mock customer support data.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from loguru import logger

class MockDataGenerator:
    """Generate mock customer support data for testing purposes."""
    
    def __init__(self, output_dir: str = "data/raw"):
        """Initialize the data generator.
        
        Args:
            output_dir: Directory to save generated data
        """
        self.output_dir = output_dir
        self._ensure_output_dir()
        
        # Define query categories and templates
        self.categories = {
            "account": [
                "I can't log in to my account",
                "How do I reset my password?",
                "I need to update my billing information",
                "My account is locked, how can I unlock it?",
                "I want to delete my account"
            ],
            "billing": [
                "I was charged incorrectly",
                "When will my next payment be due?",
                "How do I update my payment method?",
                "Can I get a refund for my purchase?",
                "I don't recognize a charge on my account"
            ],
            "product": [
                "The product doesn't work as expected",
                "How do I use feature X?",
                "Is there a way to customize the settings?",
                "The app keeps crashing when I try to use it",
                "I need help setting up the product"
            ],
            "technical": [
                "I'm getting an error message: {error_code}",
                "The website isn't loading properly",
                "I can't connect to the server",
                "The system is running very slowly",
                "How do I troubleshoot connection issues?"
            ],
            "general": [
                "What are your business hours?",
                "How can I contact a human representative?",
                "Do you offer bulk discounts?",
                "What's your return policy?",
                "How long does shipping take?"
            ]
        }
        
        # Define error codes for technical issues
        self.error_codes = [
            "ERR-1001", "ERR-2034", "ERR-4004", "ERR-5050", 
            "ERR-6023", "ERR-7891", "ERR-8080", "ERR-9001"
        ]
        
        # Define priority levels
        self.priorities = [1.0, 1.5, 2.0, 2.5, 3.0]
        
    def generate_support_queries(
        self,
        n_queries: int = 100,
        filename: str = "support_queries.csv"
    ) -> pd.DataFrame:
        """Generate mock customer support queries.
        
        Args:
            n_queries: Number of queries to generate
            filename: Output filename
            
        Returns:
            DataFrame with generated queries
        """
        queries = []
        categories = list(self.categories.keys())
        
        for i in range(n_queries):
            # Select random category
            category = np.random.choice(categories)
            
            # Select random template from category
            template = np.random.choice(self.categories[category])
            
            # Generate query from template
            query = template
            if "{error_code}" in query:
                query = query.replace("{error_code}", np.random.choice(self.error_codes))
                
            # Generate random user ID
            user_id = f"user_{np.random.randint(1000, 9999)}"
            
            # Generate random priority
            priority = np.random.choice(self.priorities)
            
            # Create query entry
            query_entry = {
                "query_id": f"q_{i+1}",
                "category": category,
                "query": query,
                "user_id": user_id,
                "priority": priority,
                "timestamp": pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 30))
            }
            
            queries.append(query_entry)
            
        # Create DataFrame and save to CSV
        df = pd.DataFrame(queries)
        filepath = os.path.join(self.output_dir, filename)
        
        df.to_csv(filepath, index=False)
        logger.info(f"Generated {n_queries} mock queries to {filepath}")
        
        return df
        
    def generate_capability_dimensions(self) -> Dict[str, np.ndarray]:
        """Generate capability dimensions for each category.
        
        Returns:
            Dictionary mapping categories to capability vectors
        """
        categories = list(self.categories.keys())
        n_categories = len(categories)
        capability_dim = 10
        
        # Create capability vectors
        capabilities = {}
        
        for i, category in enumerate(categories):
            # Base vector with some noise
            vec = np.zeros(capability_dim)
            vec[i] = 0.7  # Primary dimension
            
            # Add some capability in other dimensions
            for j in range(capability_dim):
                if j != i and j < n_categories:
                    vec[j] = np.random.uniform(0, 0.2)
                elif j >= n_categories:
                    vec[j] = np.random.uniform(0, 0.1)
                    
            # Normalize
            vec = vec / np.sum(vec)
            capabilities[category] = vec
            
        return capabilities
        
    def _ensure_output_dir(self):
        """Ensure the output directory exists."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True) 