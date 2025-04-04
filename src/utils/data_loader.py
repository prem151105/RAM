"""
Data loader for customer support datasets.
"""

import os
import pandas as pd
from typing import Dict, List, Optional, Union
from loguru import logger

class CustomerDataLoader:
    """Data loader for customer support datasets."""
    
    def __init__(
        self,
        data_dir: str = "data/raw"
    ):
        """Initialize data loader.
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = data_dir
        self.datasets = {}
        
    def load_csv(
        self,
        filename: str,
        dataset_name: Optional[str] = None
    ) -> pd.DataFrame:
        """Load a CSV file and store it in memory.
        
        Args:
            filename: Name of the CSV file (with or without .csv extension)
            dataset_name: Name to refer to this dataset (defaults to filename without extension)
            
        Returns:
            Loaded DataFrame
        """
        # Add .csv extension if not present
        if not filename.lower().endswith('.csv'):
            filename = f"{filename}.csv"
            
        # Use filename as dataset_name if not provided
        if dataset_name is None:
            dataset_name = os.path.splitext(filename)[0]
            
        # Construct full path
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            logger.info(f"Loading dataset from {filepath}")
            df = pd.read_csv(filepath)
            
            # Store dataset in memory
            self.datasets[dataset_name] = df
            
            logger.info(f"Loaded {len(df)} records into dataset '{dataset_name}'")
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset {filepath}: {str(e)}")
            return pd.DataFrame()
            
    def get_dataset(
        self,
        dataset_name: str
    ) -> pd.DataFrame:
        """Retrieve a loaded dataset by name.
        
        Args:
            dataset_name: Name of the dataset to retrieve
            
        Returns:
            DataFrame or empty DataFrame if not found
        """
        if dataset_name in self.datasets:
            return self.datasets[dataset_name]
            
        logger.warning(f"Dataset '{dataset_name}' not found. Available datasets: {list(self.datasets.keys())}")
        return pd.DataFrame()
        
    def list_datasets(self) -> List[str]:
        """List all available datasets.
        
        Returns:
            List of dataset names
        """
        return list(self.datasets.keys())
        
    def get_sample_queries(
        self,
        dataset_name: str,
        n_samples: int = 5,
        query_column: str = "query"
    ) -> List[str]:
        """Get sample queries from a dataset.
        
        Args:
            dataset_name: Name of the dataset
            n_samples: Number of samples to retrieve
            query_column: Column containing the queries
            
        Returns:
            List of query strings
        """
        df = self.get_dataset(dataset_name)
        
        if df.empty or query_column not in df.columns:
            return []
            
        return df[query_column].sample(min(n_samples, len(df))).tolist()
        
    def prepare_task_from_query(
        self,
        query: str,
        context: Optional[Dict] = None,
        priority: float = 1.0
    ) -> Dict:
        """Prepare a task dictionary from a query string.
        
        Args:
            query: Customer query string
            context: Additional context
            priority: Task priority
            
        Returns:
            Task dictionary
        """
        task = {
            "observation": query,
            "priority": priority
        }
        
        if context:
            task["context"] = context
            
        return task 