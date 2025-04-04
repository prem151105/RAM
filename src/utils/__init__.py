"""
Utility functions for the AgentNet Customer Support System.

This package contains utility functions for:
- Data loading and preprocessing
- Evaluation and metrics
- Embedding generation
- LLM interface
- Data access
"""

from .data_loader import DataLoader
from .evaluation import Evaluator
from .embedding import EmbeddingGenerator
from .llm_interface import LLMInterface, OllamaInterface
from .data_access import DataAccess
from .data_loader import CustomerDataLoader
from .data_generator import MockDataGenerator

# Version info
__version__ = "0.1.0"

# Export functions and classes
__all__ = [
    "DataLoader",
    "Evaluator",
    "EmbeddingGenerator",
    "LLMInterface",
    "DataAccess",
    "OllamaInterface",
    "CustomerDataLoader",
    "MockDataGenerator"
]

# Convenience functions
def get_llm_interface(model_name="llama2"):
    """Get an Ollama LLM interface with default settings.
    
    Args:
        model_name: Name of the model to use
        
    Returns:
        OllamaInterface: Configured LLM interface
    """
    return OllamaInterface(model_name=model_name)
    
def load_customer_data(data_file="support_queries.csv", data_dir="data/raw"):
    """Load customer data from a CSV file.
    
    Args:
        data_file: Name of the CSV file
        data_dir: Directory containing the file
        
    Returns:
        pandas.DataFrame: Loaded data
    """
    loader = CustomerDataLoader(data_dir=data_dir)
    return loader.load_csv(data_file)
    
def generate_mock_data(n_queries=100, output_file="support_queries.csv"):
    """Generate mock customer support data for testing.
    
    Args:
        n_queries: Number of queries to generate
        output_file: Name of the output file
        
    Returns:
        pandas.DataFrame: Generated data
    """
    generator = MockDataGenerator()
    return generator.generate_support_queries(n_queries=n_queries, filename=output_file) 