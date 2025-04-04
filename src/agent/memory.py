"""
Memory components for the AgentNet Customer Support System.
"""

import time
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from collections import deque
from datetime import datetime
import json
import heapq

class Memory:
    """Base memory interface."""
    
    def __init__(self, max_size: int = 1000):
        """Initialize memory with maximum size.
        
        Args:
            max_size: Maximum number of memories to store
        """
        self.max_size = max_size
        self.memory = {}
        self.access_counts = {}
        self.last_accessed = {}
        self.creation_time = {}
    
    def add(self, key: str, value: Any) -> bool:
        """Add a memory item.
        
        Args:
            key: Unique identifier for the memory
            value: Value to store
            
        Returns:
            True if successful, False otherwise
        """
        if len(self.memory) >= self.max_size and key not in self.memory:
            # Evict least recently used memory
            self._evict()
            
        self.memory[key] = value
        self.access_counts[key] = 0
        self.last_accessed[key] = time.time()
        self.creation_time[key] = time.time()
        return True
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve a memory item.
        
        Args:
            key: Identifier for the memory to retrieve
            
        Returns:
            The memory value or None if not found
        """
        if key in self.memory:
            self.access_counts[key] += 1
            self.last_accessed[key] = time.time()
            return self.memory[key]
        return None
    
    def update(self, key: str, value: Any) -> bool:
        """Update an existing memory item.
        
        Args:
            key: Identifier for the memory to update
            value: New value to store
            
        Returns:
            True if successful, False if key not found
        """
        if key in self.memory:
            self.memory[key] = value
            self.access_counts[key] += 1
            self.last_accessed[key] = time.time()
            return True
        return False
    
    def delete(self, key: str) -> bool:
        """Delete a memory item.
        
        Args:
            key: Identifier for the memory to delete
            
        Returns:
            True if successful, False if key not found
        """
        if key in self.memory:
            del self.memory[key]
            del self.access_counts[key]
            del self.last_accessed[key]
            del self.creation_time[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all memory."""
        self.memory.clear()
        self.access_counts.clear()
        self.last_accessed.clear()
        self.creation_time.clear()
    
    def size(self) -> int:
        """Get current memory size."""
        return len(self.memory)
    
    def _evict(self) -> None:
        """Evict the least valuable memory item."""
        if not self.memory:
            return
            
        # Default strategy: evict least recently used
        least_recent_key = min(self.last_accessed, key=self.last_accessed.get)
        self.delete(least_recent_key)
        
    def get_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        if not self.memory:
            return {"size": 0, "utilization": 0}
            
        return {
            "size": len(self.memory),
            "max_size": self.max_size,
            "utilization": len(self.memory) / self.max_size,
            "avg_access_count": sum(self.access_counts.values()) / len(self.access_counts) if self.access_counts else 0
        }

class VectorMemory(Memory):
    """Memory that supports vector embeddings and similarity search."""
    
    def __init__(self, max_size: int = 1000, embedding_dim: int = 384):
        """Initialize vector memory.
        
        Args:
            max_size: Maximum number of memories to store
            embedding_dim: Dimension of embedding vectors
        """
        super().__init__(max_size)
        self.embedding_dim = embedding_dim
        self.embeddings = {}
        
    def add(self, key: str, value: Any, embedding: np.ndarray) -> bool:
        """Add a memory item with embedding.
        
        Args:
            key: Unique identifier for the memory
            value: Value to store
            embedding: Vector embedding for similarity search
            
        Returns:
            True if successful, False otherwise
        """
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embedding.shape[0]}")
            
        # Normalize embedding for cosine similarity
        embedding = embedding / (np.linalg.norm(embedding) + 1e-9)
        
        result = super().add(key, value)
        if result:
            self.embeddings[key] = embedding
        return result
    
    def update(self, key: str, value: Any, embedding: Optional[np.ndarray] = None) -> bool:
        """Update an existing memory item and optionally its embedding.
        
        Args:
            key: Identifier for the memory to update
            value: New value to store
            embedding: Optional new vector embedding
            
        Returns:
            True if successful, False if key not found
        """
        result = super().update(key, value)
        
        if result and embedding is not None:
            if embedding.shape[0] != self.embedding_dim:
                raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embedding.shape[0]}")
                
            # Normalize embedding for cosine similarity
            embedding = embedding / (np.linalg.norm(embedding) + 1e-9)
            self.embeddings[key] = embedding
            
        return result
    
    def delete(self, key: str) -> bool:
        """Delete a memory item and its embedding.
        
        Args:
            key: Identifier for the memory to delete
            
        Returns:
            True if successful, False if key not found
        """
        if key in self.embeddings:
            del self.embeddings[key]
        return super().delete(key)
    
    def clear(self) -> None:
        """Clear all memory and embeddings."""
        super().clear()
        self.embeddings.clear()
    
    def search(self, embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar items based on embedding.
        
        Args:
            embedding: Query vector embedding
            top_k: Number of top results to return
            
        Returns:
            List of (key, similarity) pairs, sorted by similarity (highest first)
        """
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embedding.shape[0]}")
            
        # Normalize query embedding
        embedding = embedding / (np.linalg.norm(embedding) + 1e-9)
        
        # Calculate similarities with all stored embeddings
        similarities = {}
        for key, stored_embedding in self.embeddings.items():
            # Cosine similarity
            similarity = np.dot(embedding, stored_embedding)
            similarities[key] = similarity
            
            # Update access metadata
            self.access_counts[key] += 1
            self.last_accessed[key] = time.time()
        
        # Sort by similarity (descending) and return top_k results
        top_results = heapq.nlargest(top_k, similarities.items(), key=lambda x: x[1])
        return top_results
    
    def search_by_metadata(self, metadata_filter: Dict[str, Any], top_k: int = 5) -> List[Tuple[str, Any]]:
        """Search for items based on metadata filters.
        
        Args:
            metadata_filter: Dictionary of metadata field-value pairs to match
            top_k: Maximum number of results to return
            
        Returns:
            List of (key, value) pairs that match the filter
        """
        results = []
        
        for key, value in self.memory.items():
            # Skip non-dictionary values
            if not isinstance(value, dict):
                continue
                
            # Check if all filter criteria match
            matches = True
            for filter_key, filter_value in metadata_filter.items():
                if filter_key not in value or value[filter_key] != filter_value:
                    matches = False
                    break
                    
            if matches:
                # Update access metadata
                self.access_counts[key] += 1
                self.last_accessed[key] = time.time()
                results.append((key, value))
                
                if len(results) >= top_k:
                    break
        
        return results
    
    def _evict(self) -> None:
        """Evict the least valuable memory item based on LRU and access count."""
        if not self.memory:
            return
            
        # Combined strategy: consider both recency and access frequency
        # Lower score = more likely to be evicted
        eviction_scores = {}
        current_time = time.time()
        
        for key in self.memory:
            time_factor = 1.0 / (1.0 + current_time - self.last_accessed[key])
            access_factor = self.access_counts[key]
            
            # Combine factors with weights
            eviction_scores[key] = (0.7 * time_factor) + (0.3 * access_factor)
        
        # Evict the item with the lowest score
        evict_key = min(eviction_scores, key=eviction_scores.get)
        self.delete(evict_key)

class EpisodicMemory(VectorMemory):
    """Episodic memory stores temporally ordered, context-rich experiences."""
    
    def __init__(self, max_size: int = 1000, embedding_dim: int = 384):
        """Initialize episodic memory.
        
        Args:
            max_size: Maximum number of memories to store
            embedding_dim: Dimension of embedding vectors
        """
        super().__init__(max_size, embedding_dim)
        self.temporal_index = []  # List of keys in temporal order
    
    def add(self, key: str, value: Any, embedding: np.ndarray) -> bool:
        """Add an episodic memory with timestamp and embedding.
        
        Args:
            key: Unique identifier for the memory
            value: Value to store (should include a timestamp)
            embedding: Vector embedding for similarity search
            
        Returns:
            True if successful, False otherwise
        """
        # Ensure the memory includes a timestamp
        if isinstance(value, dict) and "timestamp" not in value:
            value["timestamp"] = datetime.now().isoformat()
            
        result = super().add(key, value, embedding)
        if result:
            self.temporal_index.append(key)
        return result
    
    def delete(self, key: str) -> bool:
        """Delete an episodic memory.
        
        Args:
            key: Identifier for the memory to delete
            
        Returns:
            True if successful, False if key not found
        """
        if key in self.temporal_index:
            self.temporal_index.remove(key)
        return super().delete(key)
    
    def clear(self) -> None:
        """Clear all episodic memory."""
        super().clear()
        self.temporal_index = []
    
    def get_recent(self, n: int = 5) -> List[Tuple[str, Any]]:
        """Get the n most recent memories.
        
        Args:
            n: Number of recent memories to retrieve
            
        Returns:
            List of (key, value) pairs, most recent first
        """
        if not self.temporal_index:
            return []
            
        # Get the n most recent keys
        recent_keys = self.temporal_index[-n:]
        recent_keys.reverse()  # Most recent first
        
        # Update access metadata for these keys
        current_time = time.time()
        for key in recent_keys:
            self.access_counts[key] += 1
            self.last_accessed[key] = current_time
        
        # Return key-value pairs
        return [(key, self.memory[key]) for key in recent_keys if key in self.memory]
    
    def get_time_range(self, start_time: str, end_time: str) -> List[Tuple[str, Any]]:
        """Get memories within a time range.
        
        Args:
            start_time: ISO format start time
            end_time: ISO format end time
            
        Returns:
            List of (key, value) pairs within the time range
        """
        results = []
        
        for key in self.temporal_index:
            value = self.memory.get(key)
            if not value or not isinstance(value, dict) or "timestamp" not in value:
                continue
                
            memory_time = value["timestamp"]
            if start_time <= memory_time <= end_time:
                # Update access metadata
                self.access_counts[key] += 1
                self.last_accessed[key] = time.time()
                results.append((key, value))
        
        return results

class SemanticMemory(VectorMemory):
    """Semantic memory stores concept-level, generalized knowledge."""
    
    def __init__(self, max_size: int = 2000, embedding_dim: int = 384):
        """Initialize semantic memory.
        
        Args:
            max_size: Maximum number of memories to store
            embedding_dim: Dimension of embedding vectors
        """
        super().__init__(max_size, embedding_dim)
        self.categories = {}  # Maps keys to categories
    
    def add(self, key: str, value: Any, embedding: np.ndarray, category: str = "general") -> bool:
        """Add a semantic memory with category and embedding.
        
        Args:
            key: Unique identifier for the memory
            value: Value to store
            embedding: Vector embedding for similarity search
            category: Category label for the memory
            
        Returns:
            True if successful, False otherwise
        """
        result = super().add(key, value, embedding)
        if result:
            self.categories[key] = category
        return result
    
    def delete(self, key: str) -> bool:
        """Delete a semantic memory.
        
        Args:
            key: Identifier for the memory to delete
            
        Returns:
            True if successful, False if key not found
        """
        if key in self.categories:
            del self.categories[key]
        return super().delete(key)
    
    def clear(self) -> None:
        """Clear all semantic memory."""
        super().clear()
        self.categories.clear()
    
    def get_by_category(self, category: str, max_results: int = 10) -> List[Tuple[str, Any]]:
        """Get memories by category.
        
        Args:
            category: Category to filter by
            max_results: Maximum number of results to return
            
        Returns:
            List of (key, value) pairs in the category
        """
        results = []
        
        # Find all keys in the specified category
        category_keys = [key for key, cat in self.categories.items() if cat == category]
        
        # Update access metadata and get values
        current_time = time.time()
        for key in category_keys[:max_results]:
            if key in self.memory:
                self.access_counts[key] += 1
                self.last_accessed[key] = current_time
                results.append((key, self.memory[key]))
        
        return results
    
    def search_hybrid(self, embedding: np.ndarray, category: Optional[str] = None, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar items based on embedding and optionally filter by category.
        
        Args:
            embedding: Query vector embedding
            category: Optional category to filter by
            top_k: Number of top results to return
            
        Returns:
            List of (key, similarity) pairs, sorted by similarity (highest first)
        """
        # First, get candidate keys based on category filter
        if category is not None:
            candidate_keys = set(key for key, cat in self.categories.items() if cat == category)
        else:
            candidate_keys = set(self.embeddings.keys())
        
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embedding.shape[0]}")
            
        # Normalize query embedding
        embedding = embedding / (np.linalg.norm(embedding) + 1e-9)
        
        # Calculate similarities with filtered embeddings
        similarities = {}
        for key in candidate_keys:
            if key in self.embeddings:
                # Cosine similarity
                similarity = np.dot(embedding, self.embeddings[key])
                similarities[key] = similarity
                
                # Update access metadata
                self.access_counts[key] += 1
                self.last_accessed[key] = time.time()
        
        # Sort by similarity (descending) and return top_k results
        top_results = heapq.nlargest(top_k, similarities.items(), key=lambda x: x[1])
        return top_results
    
    def _evict(self) -> None:
        """Evict the least valuable memory item based on access patterns and age."""
        if not self.memory:
            return
            
        # More sophisticated eviction strategy for semantic memory
        eviction_scores = {}
        current_time = time.time()
        
        for key in self.memory:
            # Time factor: how recently was it accessed
            recency = 1.0 / (1.0 + current_time - self.last_accessed[key])
            
            # Usage factor: how often it's accessed
            usage = self.access_counts[key]
            
            # Age factor: how old is the memory
            age = current_time - self.creation_time[key]
            age_factor = 1.0 / (1.0 + age / (3600 * 24))  # Scale by days
            
            # Combine factors with weights
            # Semantic memories that are accessed often should be kept
            eviction_scores[key] = (0.5 * recency) + (0.4 * usage) + (0.1 * age_factor)
        
        # Evict the item with the lowest score
        evict_key = min(eviction_scores, key=eviction_scores.get)
        self.delete(evict_key)

class QueueMemory(Memory):
    """Simple queue-based memory for temporary storage."""
    
    def __init__(self, max_size: int = 100):
        """Initialize queue memory.
        
        Args:
            max_size: Maximum number of items in the queue
        """
        super().__init__(max_size)
        self.queue = deque(maxlen=max_size)
    
    def add(self, key: str, value: Any) -> bool:
        """Add an item to the queue.
        
        Args:
            key: Unique identifier for the item
            value: Value to store
            
        Returns:
            True if successful, False otherwise
        """
        result = super().add(key, value)
        if result:
            self.queue.append(key)
        return result
    
    def delete(self, key: str) -> bool:
        """Delete an item from the queue.
        
        Args:
            key: Identifier for the item to delete
            
        Returns:
            True if successful, False if key not found
        """
        if key in self.queue:
            self.queue.remove(key)
        return super().delete(key)
    
    def clear(self) -> None:
        """Clear the queue."""
        super().clear()
        self.queue.clear()
    
    def pop(self) -> Optional[Tuple[str, Any]]:
        """Pop the oldest item from the queue.
        
        Returns:
            (key, value) pair or None if queue is empty
        """
        if not self.queue:
            return None
            
        key = self.queue.popleft()
        value = self.memory.get(key)
        
        if key in self.memory:
            self.delete(key)
            
        return key, value
    
    def peek(self) -> Optional[Tuple[str, Any]]:
        """Peek at the oldest item without removing it.
        
        Returns:
            (key, value) pair or None if queue is empty
        """
        if not self.queue:
            return None
            
        key = self.queue[0]
        value = self.memory.get(key)
        
        # Update access metadata
        if key in self.memory:
            self.access_counts[key] += 1
            self.last_accessed[key] = time.time()
            
        return key, value
    
    def _evict(self) -> None:
        """No custom eviction needed for queue memory."""
        pass  # Queue automatically handles eviction 