"""
Command-line interface for testing the AgentNet Customer Support System.
"""

import os
import sys
import argparse
import time
import json
from typing import Dict, List, Optional
import numpy as np
from loguru import logger

from .network.network import AgentNetwork
from .utils.data_generator import MockDataGenerator
from .utils.data_loader import CustomerDataLoader

def setup_logger():
    """Configure the logger."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Add file logging
    os.makedirs("logs", exist_ok=True)
    logger.add(
        "logs/agentnet_{time}.log",
        rotation="500 MB",
        level="DEBUG"
    )

def generate_test_data(n_queries: int = 100) -> None:
    """Generate test data for testing the system."""
    logger.info("Generating test data...")
    
    # Ensure data directory exists
    os.makedirs("data/raw", exist_ok=True)
    
    # Generate mock queries
    generator = MockDataGenerator(output_dir="data/raw")
    df = generator.generate_support_queries(n_queries=n_queries)
    
    logger.info(f"Generated {len(df)} test queries")
    
    # Display sample queries
    for category, group in df.groupby("category"):
        sample = group.sample(1).iloc[0]
        logger.info(f"Sample {category} query: '{sample['query']}'")

def process_single_query(query: str, context: Optional[Dict] = None) -> None:
    """Process a single query through the agent network."""
    logger.info(f"Processing query: '{query}'")
    
    # Initialize agent network
    network = AgentNetwork(
        num_agents=5,
        capability_dim=10,
        memory_size=1000,
        llm_model="llama2"
    )
    
    # Prepare task
    task = {
        "observation": query,
        "context": context or {},
        "priority": 1.0
    }
    
    # Start timer
    start_time = time.time()
    
    # Process task
    result = network.process_task(
        task=task,
        alpha=0.8,
        max_hops=5
    )
    
    # Calculate processing time
    elapsed_time = time.time() - start_time
    
    # Display results
    logger.info(f"Task processing completed in {elapsed_time:.2f} seconds")
    logger.info(f"Routing path: {' -> '.join(result['routing_path'])}")
    
    if result["status"] == "completed":
        logger.info(f"Analysis: {result['result']['analysis']}")
        logger.info(f"Solution: {result['result']['solution']}")
        logger.info(f"Response: {result['result']['response']}")
        logger.info(f"Confidence: {result['result']['confidence']}")
        
        if result['result'].get('follow_up_actions'):
            logger.info(f"Follow-up actions: {result['result']['follow_up_actions']}")
    else:
        logger.warning(f"Task processing failed: {result['status']}")

def process_batch_queries(data_file: str, limit: int = 5) -> None:
    """Process a batch of queries from a data file."""
    logger.info(f"Processing queries from {data_file}")
    
    # Load data
    loader = CustomerDataLoader(data_dir="data/raw")
    df = loader.load_csv(data_file)
    
    if df.empty:
        logger.error(f"Failed to load data from {data_file}")
        return
        
    # Limit number of queries to process
    if limit > 0:
        df = df.head(limit)
        
    # Initialize agent network
    network = AgentNetwork(
        num_agents=5,
        capability_dim=10,
        memory_size=1000,
        llm_model="llama2"
    )
    
    # Process each query
    for i, row in df.iterrows():
        query = row["query"]
        logger.info(f"Processing query {i+1}/{len(df)}: '{query}'")
        
        # Prepare task
        task = {
            "observation": query,
            "context": {"user_id": row.get("user_id", "unknown")},
            "priority": row.get("priority", 1.0)
        }
        
        # Process task
        result = network.process_task(task)
        
        if result["status"] == "completed":
            logger.info(f"Completed. Confidence: {result['result']['confidence']}")
        else:
            logger.warning(f"Failed: {result['status']}")
            
        # Add a small delay between queries
        time.sleep(1)

def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="AgentNet Customer Support CLI")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Generate test data command
    generate_parser = subparsers.add_parser("generate", help="Generate test data")
    generate_parser.add_argument(
        "--count", type=int, default=100,
        help="Number of test queries to generate"
    )
    
    # Process single query command
    query_parser = subparsers.add_parser("query", help="Process a single query")
    query_parser.add_argument("text", help="Query text")
    query_parser.add_argument(
        "--context", type=str, default="{}",
        help="Query context as JSON string"
    )
    
    # Process batch queries command
    batch_parser = subparsers.add_parser("batch", help="Process queries from a data file")
    batch_parser.add_argument(
        "--file", type=str, default="support_queries.csv",
        help="CSV file containing queries"
    )
    batch_parser.add_argument(
        "--limit", type=int, default=5,
        help="Maximum number of queries to process"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logger
    setup_logger()
    
    # Handle commands
    if args.command == "generate":
        generate_test_data(n_queries=args.count)
    elif args.command == "query":
        try:
            context = json.loads(args.context)
            process_single_query(query=args.text, context=context)
        except json.JSONDecodeError:
            logger.error(f"Invalid context JSON: {args.context}")
    elif args.command == "batch":
        process_batch_queries(data_file=args.file, limit=args.limit)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 